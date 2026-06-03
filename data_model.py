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
	def __init__(self, images_dir, gt_path, char_map=None, imgH=32, transform=None):
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
		self.char_map = char_map or self._build_charset()

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
		img = img.resize((new_w, self.imgH), Image.BILINEAR)
		if self.transform:
			img = self.transform(img)
		else:
			img = transforms.ToTensor()(img)
		# normalize to -1..1
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
	return padded, targets_concat, torch.tensor(target_lengths, dtype=torch.long), texts


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
		self.rnn = nn.Sequential(
			nn.LSTM(512, nh, bidirectional=True, num_layers=2, batch_first=False),
			# output from LSTM will be (T, B, nh*2)
		)
		self.embedding = nn.Linear(nh * 2, nclass)

	def forward(self, x):
		# x: (B, C=1, H, W)
		conv = self.cnn(x)
		b, c, h, w = conv.size()
		assert h == 1, "The height after conv must be 1"
		conv = conv.squeeze(2)  # (B, C, W)
		conv = conv.permute(2, 0, 1)  # (W, B, C)
		recurrent, _ = self.rnn[0](conv)
		T, B, H = recurrent.size()
		output = self.embedding(recurrent.view(T * B, H))
		output = output.view(T, B, -1)
		return output.log_softmax(2)


def train_epoch(model, device, dataloader, criterion, optimizer):
	model.train()
	total_loss = 0.0
	running_loss = 0.0
	for batch_idx, (imgs, targets, target_lengths, _) in enumerate(dataloader, 1):
		imgs = imgs.to(device)
		targets = targets.to(device)
		optimizer.zero_grad()
		logits = model(imgs)  # (T, B, C)
		T, B, C = logits.size()
		input_lengths = torch.full((B,), T, dtype=torch.long)
		loss = criterion(logits, targets, input_lengths, target_lengths)
		loss.backward()
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
	with torch.no_grad():
		for batch_idx, (imgs, targets, target_lengths, texts) in enumerate(dataloader, 1):
			imgs = imgs.to(device)
			targets = targets.to(device)
			logits = model(imgs)
			T, B, C = logits.size()
			input_lengths = torch.full((B,), T, dtype=torch.long)
			loss = criterion(logits, targets, input_lengths, target_lengths)
			total_loss += loss.item() * B
			# simple argmax decoder
			preds = logits.argmax(2).permute(1, 0).cpu().numpy()  # (B, T)
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
			if batch_idx % 50 == 0:
				print(f"  Val batch {batch_idx}/{len(dataloader)} processed", flush=True)

	# compute precision/recall/f1
	precision = tp_total / pred_total if pred_total > 0 else 0.0
	recall = tp_total / gt_total if gt_total > 0 else 0.0
	f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
	return total_loss / len(dataloader.dataset), correct, total, precision, recall, f1


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


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', default='IIT_data_set', help='dataset folder')
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--batch', type=int, default=32)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--imgH', type=int, default=32)
	parser.add_argument('--num_workers', type=int, default=2, help='DataLoader num_workers (set 0 to debug)')
	parser.add_argument('--save', default='checkpoints')
	parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	args = parser.parse_args()

	data_dir = args.data
	train_images = os.path.join(data_dir, 'train', 'images')
	val_images = os.path.join(data_dir, 'val', 'images')
	train_gt = os.path.join(data_dir, 'train', 'train_gt.txt')
	val_gt = os.path.join(data_dir, 'val', 'val_gt.txt')

	char_map, idx2char = build_charset_from_file(train_gt)
	nclass = len(char_map) + 1  # +1 for blank(0)

	transform = transforms.Compose([transforms.ToTensor()])

	train_ds = WordDataset(train_images, train_gt, char_map, imgH=args.imgH, transform=transform)
	val_ds = WordDataset(val_images, val_gt, char_map, imgH=args.imgH, transform=transform)

	train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
	val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

	print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, batch_size: {args.batch}, num_workers: {args.num_workers}")

	device = torch.device(args.device)
	model = CRNN(args.imgH, 1, nclass).to(device)
	criterion = nn.CTCLoss(blank=0, zero_infinity=True)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	os.makedirs(args.save, exist_ok=True)

	metrics_file = os.path.join(args.save, 'evaluation_metrics.csv')
	# write header if not exists
	if not os.path.exists(metrics_file):
		with open(metrics_file, 'w', newline='', encoding='utf-8') as mf:
			w = csv.writer(mf)
			w.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'precision', 'recall', 'f1'])

	for epoch in range(1, args.epochs + 1):
		train_loss = train_epoch(model, device, train_loader, criterion, optimizer)
		val_loss, correct, total, precision, recall, f1 = validate(model, device, val_loader, criterion, idx2char)
		acc = correct / total if total > 0 else 0.0
		print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={acc:.4f} prec={precision:.4f} rec={recall:.4f} f1={f1:.4f}")
		# prepare checkpoint dict
		ckpt = {'epoch': epoch, 'model_state': model.state_dict(), 'char_map': char_map}
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
			w.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{acc:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])
		print(f"Saved checkpoint to: {save_path} and workspace root: {root_save_path}")


if __name__ == '__main__':
	main()

