import os
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from collections import Counter


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
        self.rnn = nn.LSTM(512, nh, bidirectional=True, num_layers=2, batch_first=False)
        self.embedding = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        recurrent, _ = self.rnn(conv)
        T, B, H = recurrent.size()
        output = self.embedding(recurrent.view(T * B, H))
        output = output.view(T, B, -1)
        return output.log_softmax(2)


def decode_prediction(logits, idx2char):
    # logits: (T,1,C)
    preds = logits.argmax(2).permute(1, 0).cpu().numpy()  # (B, T)
    p = preds[0]
    prev = -1
    out = []
    for ch in p:
        if ch != prev and ch != 0:
            out.append(idx2char.get(ch, ''))
        prev = ch
    return ''.join(out)


def preprocess_image(img_path, imgH):
    img = Image.open(img_path).convert('L')
    w, h = img.size
    new_w = max(1, int(w * (imgH / float(h))))
    img = img.resize((new_w, imgH), Image.BILINEAR)
    tensor = transforms.ToTensor()(img)
    tensor = tensor.sub(0.5).div(0.5)
    return tensor.unsqueeze(0)  # (1,1,H,W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=os.path.join(os.getcwd(), 'crnn_latest.pt'), help='checkpoint path')
    parser.add_argument('--input', default=os.path.join(os.getcwd(), 'check'), help='input folder with images')
    parser.add_argument('--imgH', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default=os.path.join(os.getcwd(), 'check_results.csv'))
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    char_map = ckpt.get('char_map')
    if char_map is None:
        raise KeyError('char_map not found in checkpoint')
    idx2char = {v: k for k, v in char_map.items()}
    nclass = len(char_map) + 1

    model = CRNN(args.imgH, 1, nclass)
    model.load_state_dict(ckpt['model_state'])
    model.to(args.device)
    model.eval()

    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    files = [os.path.join(args.input, f) for f in os.listdir(args.input) if os.path.splitext(f)[1].lower() in exts]
    results = []
    for p in files:
        try:
            inp = preprocess_image(p, args.imgH).to(args.device)
            with torch.no_grad():
                logits = model(inp)  # (T, B, C)
                pred = decode_prediction(logits, idx2char)
            print(f"{os.path.basename(p)} -> {pred}")
            results.append((os.path.basename(p), pred))
        except Exception as e:
            print(f"Error processing {p}: {e}")

    # write CSV
    import csv
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image', 'prediction'])
        for row in results:
            w.writerow(row)
    print(f"Wrote results to {args.out}")


if __name__ == '__main__':
    main()
