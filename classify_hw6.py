# -*- coding: utf-8 -*-
# MNIST-like batch classifier without CLI.
# Edit CONFIG below and run: python classify_mnist_auto.py

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import csv
import sys

# =========================
# CONFIG (edit these)
# =========================
WEIGHTS_PATH = r"LeNet.pt"                 # path to your trained weights (.pt)
ARCH = "lenet"                             # "lenet" or "mlp"
INPUT_PATH = r"data\photos"                # directory with images (scanned recursively)
INVERT = True                              # True if images are black digits on white background
BATCH_SIZE = 128
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUTPUT_CSV = None                          # None -> save to "<INPUT_PATH>/predictions.csv"
NUM_WORKERS = 0                            # 0 is safest on Windows; try 2 if you want parallel loading
PIN_MEMORY = (torch.cuda.is_available())   # pin_memory speeds up host->GPU transfer
# =========================


# -------------------------
# Models (must match training)
# -------------------------
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def build_model(arch: str) -> nn.Module:
    a = arch.lower()
    if a == "lenet":
        return LeNet()
    if a == "mlp":
        return MLPNet()
    raise ValueError(f"Unknown arch: {arch}. Choose 'lenet' or 'mlp'.")


# -------------------------
# Transforms and dataset
# -------------------------
class InvertTransform:
    """Top-level callable (pickle-safe) to replace lambda for Windows DataLoader."""
    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.invert(img)


def make_transform(invert: bool):
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28), interpolation=InterpolationMode.BILINEAR),
    ]
    if invert:
        ops.append(InvertTransform())
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # must match training
    ])
    return transforms.Compose(ops)


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]


class FileImageDataset(Dataset):
    def __init__(self, files: List[Path], transform):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception as e:
            # Raise with path info for easier debugging
            raise RuntimeError(f"Failed to load/transform image: {path} ({e})")
        return tensor, str(path)


# -------------------------
# Inference helpers
# -------------------------
@torch.no_grad()
def infer_folder(model: nn.Module, device: torch.device, root: Path, transform, batch_size=128) -> List[Tuple[str, int, list]]:
    files = list_images(root)
    if not files:
        print(f"[ERROR] No images found in: {root}")
        return []

    ds = FileImageDataset(files, transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=False if NUM_WORKERS == 0 else True,
    )

    model.eval()
    results = []
    for imgs, paths in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu()
        preds = probs.argmax(1).tolist()
        for pth, pred, prob_vec in zip(paths, preds, probs.tolist()):
            results.append((pth, pred, prob_vec))
    return results


def save_csv(rows: List[Tuple[str, int, list]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["path", "pred"] + [f"prob_{i}" for i in range(10)]
        w.writerow(header)
        for path, pred, prob_vec in rows:
            w.writerow([path, pred] + [f"{p:.6f}" for p in prob_vec])
    print(f"[OK] Saved predictions to: {out_csv}")


# -------------------------
# Main
# -------------------------
def main():
    weights = Path(WEIGHTS_PATH)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights}")
        sys.exit(1)

    in_path = Path(INPUT_PATH)
    if not in_path.exists() or not in_path.is_dir():
        print(f"[ERROR] INPUT_PATH must be an existing directory: {in_path}")
        sys.exit(1)

    device = torch.device(DEVICE)
    print(f"device: {device}")
    print(f"arch: {ARCH}")
    print(f"weights: {weights}")
    print(f"input dir: {in_path}")
    print(f"invert: {INVERT}")

    # Build and load model
    model = build_model(ARCH).to(device)
    try:
        state = torch.load(str(weights), map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        sys.exit(1)

    transform = make_transform(INVERT)

    # Inference
    rows = infer_folder(model, device, in_path, transform, batch_size=BATCH_SIZE)
    if not rows:
        return

    # Print first 20
    print(f"Scanned {len(rows)} images. First 20 predictions:")
    for path, pred, prob_vec in rows[:20]:
        print(f"{Path(path).name}: pred={pred}, conf={max(prob_vec):.3f}")

    # Save CSV
    out_csv = Path(OUTPUT_CSV) if OUTPUT_CSV else (in_path / "predictions.csv")
    save_csv(rows, out_csv)


if __name__ == "__main__":
    main()