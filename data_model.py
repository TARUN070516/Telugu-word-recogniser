import os
import argparse
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter
import csv


class WordDataset(Dataset):
	def __init__(self, images_dir, gt_path, char_map=None, imgH=32, transform=None, augment=False):
		self.images_dir = images_dir
		self.gt = []
		with open(gt_path, 'r', encoding='utf-8') as f:
			for line in f:
				parts = line.strip().split('\t')
				if len(parts) >= 2:
					imgname, text = parts[0], parts[1]
					self.gt.append((imgname, text))
		self.transform = transform
		self.imgH = imgH
		self.augment = augment
		self.char_map = char_map or self._build_charset()
		# Build augmentation pipeline once (only used when augment=True).
		# ColorJitter on grayscale affects brightness/contrast only.
		# RandomAffine models the slight tilt and scale variation common in
		# scanned word images; fill=255 pads with white (paper background).
		self._augment_tf = transforms.Compose([
			transforms.ColorJitter(brightness=0.3, contrast=0.3),
			transforms.RandomAffine(
				degrees=2,
				translate=(0.02, 0.02),
				scale=(0.95, 1.05),
				fill=255,
			),
		])

	def _build_charset(self):
		chars = set()
		for _, txt in self.gt:
			for ch in txt:
				chars.add(ch)
		# Reserve 0 for CTC blank
		sorted_chars = sorted(list(chars))
		return {ch: i + 1 for i, ch in enumerate(sorted_chars)}

	def __len__(self):
		return len(self.gt)

	def __getitem__(self, idx):
		imgname, txt = self.gt[idx]
		# Resolve image path robustly: GT may contain relative prefixes like 'images/..'
		candidates = []
		# if absolute path provided, try it first
		if os.path.isabs(imgname):
			candidates.append(imgname)
		# direct join with images_dir
		candidates.append(os.path.join(self.images_dir, imgname))
		# basename only
		candidates.append(os.path.join(self.images_dir, os.path.basename(imgname)))
		# strip leading 'images/' or 'images\\' segments
		if imgname.startswith('images/') or imgname.startswith('images\\'):
			candidates.append(os.path.join(self.images_dir, imgname.split('/', 1)[-1]))
			candidates.append(os.path.join(self.images_dir, imgname.split('\\', 1)[-1]))

		imgpath = None
		for c in candidates:
			if c and os.path.exists(c):
				imgpath = c
				break
		if imgpath is None:
			raise FileNotFoundError(f"Image not found for GT entry '{imgname}'. Tried: {candidates}")
		img = Image.open(imgpath).convert('L')
		# resize keeping aspect ratio to height imgH
		w, h = img.size
		new_w = max(1, int(w * (self.imgH / float(h))))
		# CTC requires T >= 2*L - 1 time steps. T ≈ new_w // 4 - 1, so we need
		# new_w >= (2*len(target) + 2) * 4  to guarantee a valid CTC path.
		# Enforce a per-sample minimum width so short images never violate this.
		n_chars = sum(1 for ch in txt if ch in (self.char_map or {}))
		min_w = max(16, (2 * n_chars + 2) * 4)
		new_w = max(new_w, min_w)
		img = img.resize((new_w, self.imgH), Image.BILINEAR)
		# Apply augmentation BEFORE converting to tensor so PIL transforms work.
		# Only active for the training set (augment=True); val is always clean.
		if self.augment:
			img = self._augment_tf(img)
		# Convert to tensor then normalize to [-1, 1].
		# Always use ToTensor() here; do NOT also pass transforms.ToTensor() as the
		# external transform — that would double-apply and is no longer needed.
		img = transforms.ToTensor()(img)
		img = img.sub(0.5).div(0.5)

		# encode text to ints
		target = [self.char_map[ch] for ch in txt if ch in self.char_map]
		target = torch.tensor(target, dtype=torch.long)
		return img, target, txt


def collate_fn(batch):
	# batch: list of (img, target, txt)
	imgs = [b[0] for b in batch]
	targets = [b[1] for b in batch]
	texts = [b[2] for b in batch]
	batch_sizes = [im.size(2) for im in imgs]
	maxW = max(batch_sizes)
	# pad images to maxW
	padded = torch.zeros(len(imgs), 1, imgs[0].size(1), maxW)
	for i, im in enumerate(imgs):
		padded[i, :, :, :im.size(2)] = im

	# concat targets
	target_lengths = [t.numel() for t in targets]
	if len(targets) > 0:
		targets_concat = torch.cat(targets)
	else:
		targets_concat = torch.tensor([], dtype=torch.long)
	# return padded images along with concatenated targets, target lengths, texts,
	# and the original per-sample widths (in pixels) before padding.
	return padded, targets_concat, torch.tensor(target_lengths, dtype=torch.long), texts, torch.tensor(batch_sizes, dtype=torch.long)


class CRNN(nn.Module):
	def __init__(self, imgH, nc, nclass, nh=256):
		super().__init__()
		self.cnn = nn.Sequential(
			nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
			nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
			nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
			nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
			nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True), nn.BatchNorm2d(512),
			nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
			nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True)
		)
		self.rnn = nn.LSTM(512, nh, bidirectional=True, num_layers=2,
		                   batch_first=False, dropout=0.3)
		# Dropout before the final projection reduces over-reliance on specific
		# LSTM units and is the primary regularisation against overfitting.
		self.dropout = nn.Dropout(0.3)
		self.embedding = nn.Linear(nh * 2, nclass)

	def forward(self, x):
		# x: (B, C=1, H, W)
		conv = self.cnn(x)
		b, c, h, w = conv.size()
		assert h == 1, "The height after conv must be 1"
		conv = conv.squeeze(2)  # (B, C, W)
		conv = conv.permute(2, 0, 1)  # (W, B, C)
		recurrent, _ = self.rnn(conv)
		T, B, H = recurrent.size()
		output = self.embedding(self.dropout(recurrent.view(T * B, H)))
		output = output.view(T, B, -1)
		return output.log_softmax(2)


def compute_input_lengths(widths, T):
	"""
	Compute per-sample CTC input lengths from original (pre-padding) pixel widths.

	CNN width reduction:
	  MaxPool2d(2,2)          -> w // 2
	  MaxPool2d(2,2)          -> w // 4
	  MaxPool2d((2,1),(2,1))  -> unchanged
	  MaxPool2d((2,1),(2,1))  -> unchanged
	  Conv2d(512,512,2,1,0)   -> w - 1   (kernel=2, stride=1, pad=0)

	So: T_sample = (w // 4) - 1

	We clamp to [1, T] where T is the actual model output length for the batch
	(determined by the widest/padded image). Per-sample lengths must be <= T;
	they should also be >= target_length for valid CTC paths.
	"""
	lengths = [max(1, int(w // 4) - 1) for w in widths]
	return torch.tensor(lengths, dtype=torch.long).clamp(max=T)


def train_epoch(model, device, dataloader, criterion, optimizer):
	model.train()
	total_loss = 0.0
	running_loss = 0.0
	for batch_idx, batch in enumerate(dataloader, 1):
		imgs, targets, target_lengths = batch[0].to(device), batch[1].to(device), batch[2].to(device)
		widths = batch[4].tolist()
		optimizer.zero_grad()
		logits = model(imgs)  # (T, B, C)
		T, B, C = logits.size()
		# FIX: use shared helper so training and validation use identical length logic
		input_lengths = compute_input_lengths(widths, T).to(device)
		# Sanity check on first batch only
		if batch_idx == 1:
			violations = (input_lengths < target_lengths).sum().item()
			if violations > 0:
				print(f"  [WARN] Batch 1: {violations}/{B} samples have input_length < target_length "
				      f"— those CTC losses are zeroed. Check for very narrow images.")
		loss = criterion(logits, targets, input_lengths, target_lengths)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
		optimizer.step()
		total_loss += loss.item() * B
		running_loss += loss.item()
		if batch_idx % 50 == 0:
			recent = running_loss / 50
			running_loss = 0.0
			print(f"  Batch {batch_idx}/{len(dataloader)} - recent_loss={recent:.4f}", flush=True)
	return total_loss / len(dataloader.dataset)


def validate(model, device, dataloader, criterion, map_idx2char):
	model.eval()
	total_loss = 0.0
	correct = 0
	total = 0
	# metrics counters for character-level precision/recall
	tp_total = 0
	pred_total = 0
	gt_total = 0
	# extra metrics: edit distances / CER
	total_edit = 0
	total_chars = 0
	# blank token counts
	total_blanks = 0
	total_time_steps = 0
	total_invalid = 0
	with torch.no_grad():
		for batch_idx, batch in enumerate(dataloader, 1):
			imgs, targets, target_lengths, texts, widths = (
				batch[0].to(device), batch[1].to(device), batch[2].to(device),
				batch[3], batch[4].tolist()
			)
			logits = model(imgs)  # (T, B, C)
			T, B, C = logits.size()
			# FIX: use shared helper — same formula as train_epoch
			input_lengths = compute_input_lengths(widths, T).to(device)
			if batch_idx == 1:
				print(f"  [DEBUG] logits shape (T,B,C): {logits.size()}")
				print(f"  [DEBUG] example widths:        {widths[:8]}")
				print(f"  [DEBUG] computed input_lengths:{input_lengths.tolist()[:8]}")
				print(f"  [DEBUG] target_lengths:        {target_lengths.tolist()[:8]}")
			invalid = (input_lengths < target_lengths).sum().item()
			total_invalid += invalid
			if batch_idx == 1:
				print(f"  [DEBUG] invalid (target>input) in first batch: {invalid}/{B}")
			loss = criterion(logits, targets, input_lengths, target_lengths)
			total_time_steps += B * T
			total_loss += loss.item() * B
			# simple argmax decoder
			preds = logits.argmax(2).permute(1, 0).cpu().numpy()  # (B, T)
			# count blanks
			total_blanks += int((preds == 0).sum())
			for i, p in enumerate(preds):
				# collapse repeats and remove blanks (0)
				prev = -1
				out = []
				for ch in p:
					if ch != prev and ch != 0:
						out.append(map_idx2char.get(ch, ''))
					prev = ch
				pred_str = ''.join(out)
				if pred_str == texts[i]:
					correct += 1
				total += 1
				# character-level counts
				pred_cnt = Counter(pred_str)
				gt_cnt = Counter(texts[i])
				for ch, c in pred_cnt.items():
					if ch in gt_cnt:
						tp_total += min(c, gt_cnt[ch])
				pred_total += sum(pred_cnt.values())
				gt_total += sum(gt_cnt.values())
				# edit distance / CER
				ed = edit_distance(pred_str, texts[i])
				total_edit += ed
				total_chars += len(texts[i])
			# print a few example predictions for diagnostics
			if batch_idx == 1 and i < 5:
				print(f"  Val sample {i}: GT='{texts[i]}', PRED='{pred_str}'")
			if batch_idx % 50 == 0:
				print(f"  Val batch {batch_idx}/{len(dataloader)} processed", flush=True)

	# compute precision/recall/f1
	precision = tp_total / pred_total if pred_total > 0 else 0.0
	recall = tp_total / gt_total if gt_total > 0 else 0.0
	f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
	cer = total_edit / total_chars if total_chars > 0 else 0.0
	avg_edit = total_edit / total if total > 0 else 0.0
	blank_freq = total_blanks / total_time_steps if total_time_steps > 0 else 0.0
	print(f"  [DEBUG] total invalid target>input across val: {total_invalid}")
	# return the extra metrics in addition to previous ones
	return total_loss / len(dataloader.dataset), correct, total, precision, recall, f1, cer, avg_edit, blank_freq


def build_charset_from_file(gt_path):
	chars = set()
	with open(gt_path, 'r', encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split('\t')
			if len(parts) >= 2:
				for ch in parts[1]:
					chars.add(ch)
	sorted_chars = sorted(list(chars))
	char_map = {ch: i + 1 for i, ch in enumerate(sorted_chars)}
	idx2char = {i + 1: ch for i, ch in enumerate(sorted_chars)}
	return char_map, idx2char


def edit_distance(s1, s2):
	# simple Levenshtein distance
	if s1 == s2:
		return 0
	len1 = len(s1)
	len2 = len(s2)
	dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
	for i in range(len1 + 1):
		dp[i][0] = i
	for j in range(len2 + 1):
		dp[0][j] = j
	for i in range(1, len1 + 1):
		for j in range(1, len2 + 1):
			cost = 0 if s1[i - 1] == s2[j - 1] else 1
			dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
	return dp[len1][len2]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', default='IIT_data_set', help='dataset folder')
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--batch', type=int, default=32)
	parser.add_argument('--lr', type=float, default=1e-4,
	                    help='peak learning rate (warmed up over --warmup_epochs)')
	parser.add_argument('--warmup_epochs', type=int, default=5,
	                    help='number of epochs to linearly warm LR from lr/10 up to lr')
	parser.add_argument('--imgH', type=int, default=32)
	parser.add_argument('--num_workers', type=int, default=2, help='DataLoader num_workers (set 0 to debug)')
	parser.add_argument('--early_stop_patience', type=int, default=5,
	                    help='stop training if val_loss does not improve for this many epochs (0 = disabled)')
	parser.add_argument('--save', default='checkpoints')
	parser.add_argument('--resume', default=None, help='checkpoint path to resume training from')
	parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	args = parser.parse_args()

	data_dir = args.data
	train_images = os.path.join(data_dir, 'train', 'images')
	val_images = os.path.join(data_dir, 'val', 'images')
	train_gt = os.path.join(data_dir, 'train', 'train_gt.txt')
	val_gt = os.path.join(data_dir, 'val', 'val_gt.txt')

	char_map, idx2char = build_charset_from_file(train_gt)
	nclass = len(char_map) + 1  # +1 for blank(0)

	# FIX: augment=True only on training set — val always uses clean images.
	train_ds = WordDataset(train_images, train_gt, char_map, imgH=args.imgH, augment=True)
	val_ds   = WordDataset(val_images,   val_gt,   char_map, imgH=args.imgH, augment=False)

	# Warn about val characters not seen during training (they are silently dropped
	# from targets, corrupting target_lengths and causing phantom CTC violations).
	unknown_chars = {ch for _, txt in val_ds.gt for ch in txt if ch not in char_map}
	if unknown_chars:
		print(f"[WARN] {len(unknown_chars)} character(s) in val GT not in char_map "
		      f"(will be dropped from targets): {sorted(unknown_chars)[:20]}")

	train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
	val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

	print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, batch_size: {args.batch}, num_workers: {args.num_workers}")

	device = torch.device(args.device)
	model = CRNN(args.imgH, 1, nclass).to(device)
	criterion = nn.CTCLoss(blank=0, zero_infinity=True)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	# FIX: warm up LR linearly for warmup_epochs, then reduce on plateau.
	# Starting too high (1e-3) causes the model to collapse to all-blanks in epoch 1.
	warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
		optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
	)
	plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode='min', factor=0.5, patience=2
	)

	start_epoch = 1
	if args.resume:
		if not os.path.exists(args.resume):
			raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
		ckpt = torch.load(args.resume, map_location=device)
		model.load_state_dict(ckpt['model_state'])
		if 'optimizer_state' in ckpt:
			optimizer.load_state_dict(ckpt['optimizer_state'])
			print(f"Loaded optimizer state from checkpoint: {args.resume}")
		else:
			print(f"Loaded checkpoint {args.resume} without optimizer state; optimizer will start fresh")
		# Restore scheduler states so the LR trajectory continues exactly where
		# it left off. Without this, LinearLR warmup restarts from step 0 and
		# ramps the LR upward — causing the training loss explosion seen in
		# the resume run (lr jumped from ~2e-3 to ~3.6e-3 on epoch 10).
		if 'warmup_scheduler_state' in ckpt:
			warmup_scheduler.load_state_dict(ckpt['warmup_scheduler_state'])
			print(f"Restored warmup scheduler state from checkpoint.")
		else:
			# Old checkpoint with no saved scheduler state.
			# We must NOT call warmup_scheduler.step() here — doing so before
			# any optimizer.step() triggers PyTorch's ordering warning AND
			# silently overwrites the LR that was just restored from
			# optimizer_state with LinearLR's own computation (which uses
			# base_lr=args.lr, not the restored value).
			# Instead, directly set last_epoch on the scheduler so it considers
			# warmup complete and acts as identity for all future .step() calls.
			resumed_epoch = ckpt.get('epoch', 0)
			steps_done = min(resumed_epoch, args.warmup_epochs)
			warmup_scheduler.last_epoch = steps_done
			# Recompute _last_lr to match last_epoch without touching the optimizer.
			warmup_scheduler._last_lr = [
				group['lr'] for group in optimizer.param_groups
			]
			print(f"Old checkpoint (no scheduler state): set warmup last_epoch={steps_done} "
			      f"(warmup {'complete' if steps_done >= args.warmup_epochs else 'in progress'}).")
		if 'plateau_scheduler_state' in ckpt:
			plateau_scheduler.load_state_dict(ckpt['plateau_scheduler_state'])
			print(f"Restored plateau scheduler state from checkpoint.")
		else:
			print(f"Old checkpoint: plateau scheduler starts fresh (no saved state).")
		checkpoint_char_map = ckpt.get('char_map')
		if checkpoint_char_map is not None and checkpoint_char_map != char_map:
			raise ValueError('Checkpoint char_map does not match current dataset charset.')
		start_epoch = ckpt.get('epoch', 0) + 1
		print(f"Resuming training from epoch {start_epoch}")

	os.makedirs(args.save, exist_ok=True)

	metrics_file = os.path.join(args.save, 'evaluation_metrics.csv')
	# write header if not exists
	if not os.path.exists(metrics_file):
		with open(metrics_file, 'w', newline='', encoding='utf-8') as mf:
			w = csv.writer(mf)
			w.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'precision', 'recall', 'f1', 'cer', 'avg_edit', 'blank_freq'])

	if start_epoch > args.epochs:
		print(f"Checkpoint already trained through epoch {start_epoch - 1}. Set --epochs greater than {start_epoch - 1} to continue training.")
		return

	# Early stopping state — restored from checkpoint if resuming, so that
	# best_val_loss reflects the true historical best rather than resetting to
	# inf (which caused epoch 9's inflated val_loss to be wrongly saved as best).
	best_val_loss = float('inf')
	early_stop_counter = 0
	best_model_path = os.path.join(args.save, 'best_model.pt')
	if args.resume and 'best_val_loss' in ckpt:
		best_val_loss = ckpt['best_val_loss']
		early_stop_counter = ckpt.get('early_stop_counter', 0)
		print(f"Restored early stopping state: best_val_loss={best_val_loss:.4f}, "
		      f"counter={early_stop_counter}/{args.early_stop_patience}")

	for epoch in range(start_epoch, args.epochs + 1):
		train_loss = train_epoch(model, device, train_loader, criterion, optimizer)
		val_loss, correct, total, precision, recall, f1, cer, avg_edit, blank_freq = validate(model, device, val_loader, criterion, idx2char)
		# Step warmup scheduler every epoch; plateau scheduler uses val_loss
		warmup_scheduler.step()
		plateau_scheduler.step(val_loss)
		current_lr = optimizer.param_groups[0]['lr']
		acc = correct / total if total > 0 else 0.0
		print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={acc:.4f} "
		      f"prec={precision:.4f} rec={recall:.4f} f1={f1:.4f} cer={cer:.4f} "
		      f"avg_edit={avg_edit:.3f} blank_freq={blank_freq:.4f} lr={current_lr:.2e}")
		# Update early-stopping state FIRST so the checkpoint captures the
		# correct values. Previously ckpt was built before this block, meaning
		# a "best" epoch's checkpoint still stored the old best_val_loss.
		is_best = val_loss < best_val_loss - 1e-4
		if is_best:
			best_val_loss = val_loss
			early_stop_counter = 0
		else:
			early_stop_counter += 1

		# Build checkpoint AFTER updating state — best_val_loss and
		# early_stop_counter are now current for this epoch.
		ckpt = {
			'epoch': epoch,
			'model_state': model.state_dict(),
			'optimizer_state': optimizer.state_dict(),
			'warmup_scheduler_state': warmup_scheduler.state_dict(),
			'plateau_scheduler_state': plateau_scheduler.state_dict(),
			'best_val_loss': best_val_loss,
			'early_stop_counter': early_stop_counter,
			'char_map': char_map
		}
		# save to the configured save folder
		save_path = os.path.join(args.save, f'model_epoch_{epoch}.pt')
		torch.save(ckpt, save_path)
		# also save a copy to the workspace root (Telugu word recogniser root)
		root_save_path = os.path.join(os.getcwd(), f'model_epoch_{epoch}.pt')
		torch.save(ckpt, root_save_path)
		# also update a latest pointer file at root for easy loading
		latest_root = os.path.join(os.getcwd(), 'crnn_latest.pt')
		torch.save(ckpt, latest_root)
		with open(metrics_file, 'a', newline='', encoding='utf-8') as mf:
			w = csv.writer(mf)
			w.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{acc:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{cer:.4f}", f"{avg_edit:.3f}", f"{blank_freq:.4f}"])
		print(f"Saved checkpoint to: {save_path} and workspace root: {root_save_path}")

		if is_best:
			torch.save(ckpt, best_model_path)
			print(f"  [BEST] New best val_loss={val_loss:.4f} — saved to {best_model_path}")
		else:
			print(f"  [ES]   No improvement for {early_stop_counter}/{args.early_stop_patience} epoch(s). "
			      f"Best val_loss={best_val_loss:.4f}")
			if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
				print(f"Early stopping triggered at epoch {epoch}. "
				      f"Best model saved at: {best_model_path}")
				break


if __name__ == '__main__':
	main()