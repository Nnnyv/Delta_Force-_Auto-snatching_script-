import argparse, os, time, json, random, struct
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


# ---------------- Model ----------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16->8
            nn.Dropout(0.10),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------- Utils --------------
def set_seed(seed: int = 42, deterministic: bool = False):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def stratified_split_indices(targets: List[int], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[int], List[int]]:
    from collections import defaultdict
    by_cls = defaultdict(list)
    for idx, y in enumerate(targets):
        by_cls[int(y)].append(idx)
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for _, idxs in by_cls.items():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio)) if len(idxs) > 1 else 1
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx


@torch.no_grad()
def evaluate(model: nn.Module, loader, device):
    # 使用双精度累计与返回，以获得更高精度的 acc（适合 10 万样本）
    model.eval()
    ce = nn.CrossEntropyLoss()

    loss_sum = np.float64(0.0)
    correct = np.int64(0)
    total   = np.int64(0)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)

        bsz = x.size(0)
        loss_sum += np.float64(loss.item()) * np.int64(bsz)
        pred = logits.argmax(1)
        correct += np.int64((pred == y).sum().item())
        total   += np.int64(bsz)

    avg_loss = (loss_sum / total) if total else np.float64(0.0)
    acc      = (np.float64(correct) / np.float64(total)) if total else np.float64(0.0)
    return avg_loss, acc  # numpy.float64（双精度）


# --------- Top-level transforms (picklable) ---------
class ToBinary:
    def __init__(self, thr: float = 0.75):
        self.thr = float(thr)
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is tensor in [0,1]
        return (x > self.thr).float()

class Invert:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is tensor in [0,1]
        return 1.0 - x


# --------- IDX (.idx3-ubyte / .idx1-ubyte) Loader ---------
def load_idx_images(path: str) -> np.ndarray:
    # images: magic=2051, dims: [num, rows, cols]
    with open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2051:
            raise ValueError(f"Invalid magic number for images: {magic} (expect 2051). File={path}")
        num = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size != num * rows * cols:
        raise ValueError(f"Image data size mismatch: got {arr.size}, expected {num*rows*cols}.")
    arr = arr.reshape(num, rows, cols)
    return arr


def load_idx_labels(path: str) -> np.ndarray:
    # labels: magic=2049, dims: [num]
    with open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2049:
            raise ValueError(f"Invalid magic number for labels: {magic} (expect 2049). File={path}")
        num = struct.unpack(">I", f.read(4))[0]
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size != num:
        raise ValueError(f"Label data size mismatch: got {arr.size}, expected {num}.")
    return arr


class IdxDataset(Dataset):
    def __init__(
        self,
        images_path: str,
        labels_path: str,
        transform=None,
        class_to_idx: Optional[Dict[int, int]] = None,
    ):
        super().__init__()
        self.images = load_idx_images(images_path)
        self.raw_labels = load_idx_labels(labels_path).astype(np.int64)
        if self.images.shape[0] != self.raw_labels.shape[0]:
            raise ValueError(f"Number of images {self.images.shape[0]} != labels {self.raw_labels.shape[0]}")
        # 由训练集生成的映射确保标签连续（0..C-1）
        if class_to_idx is None:
            uniq = sorted(np.unique(self.raw_labels).tolist())
            self.class_to_idx = {int(c): i for i, c in enumerate(uniq)}
        else:
            self.class_to_idx = dict(class_to_idx)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        img = self.images[idx]  # (H,W) uint8
        y_raw = int(self.raw_labels[idx])
        y = self.class_to_idx[y_raw]
        # 转 PIL 后交给 torchvision transforms
        pil = Image.fromarray(img, mode="L")
        if self.transform is not None:
            x = self.transform(pil)
        else:
            x = transforms.ToTensor()(pil)  # Fallback
        return x, y


# -------------- Transforms --------------
def build_transforms(
    img_size: int = 32,
    use_aug: bool = False,
    binarize: bool = False,
    bin_thr: float = 0.75,
    invert: bool = False,
    to_grayscale: bool = False,
):
    norm = transforms.Normalize(mean=[0.5], std=[0.5])
    aug = []
    if use_aug:
        aug = [transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))]

    # Pipeline: PIL -> (Grayscale) -> Resize -> (Aug) -> ToTensor[0,1] -> (Invert) -> (Binarize) -> Normalize
    base = []
    if to_grayscale:
        base += [transforms.Grayscale(num_output_channels=1)]
    base += [
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        *aug,
        transforms.ToTensor(),  # [0,1]
    ]
    if invert:
        base += [Invert()]
    if binarize:
        base += [ToBinary(bin_thr)]
    base += [norm]

    # 验证集不做增广
    val_tf = []
    if to_grayscale:
        val_tf += [transforms.Grayscale(num_output_channels=1)]
    val_tf += [
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
    if invert:
        val_tf += [Invert()]
    if binarize:
        val_tf += [ToBinary(bin_thr)]
    val_tf += [norm]
    return transforms.Compose(base), transforms.Compose(val_tf)


# --------- ONNX export helper (legacy exporter) ---------
def export_onnx(model: nn.Module, out_dir: Path, img_size: int, onnx_name: str = "model.onnx"):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # 记录原设备并切换到 CPU
    try:
        prev_device = next(model.parameters()).device
    except StopIteration:
        prev_device = torch.device("cpu")
    model.to("cpu")

    dummy = torch.randn(1, 1, img_size, img_size, device="cpu")
    onnx_path = out_dir / onnx_name

    torch.onnx.export(
        model, dummy, onnx_path.as_posix(),
        input_names=["input"], output_names=["logits"],
        opset_version=12, do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
    )
    print(f"[info] Exported ONNX (legacy, cpu): {onnx_path}")

    model.to(prev_device)
    return onnx_path


# -------------- DataLoader helpers for IDX --------------
def make_loaders_idx_split(
    train_images: str, train_labels: str,
    img_size: int, batch_size: int, num_workers: int,
    val_split: float, seed: int, use_aug: bool, binarize: bool, bin_thr: float, invert: bool
):
    train_tf, val_tf = build_transforms(img_size, use_aug=use_aug, binarize=binarize, bin_thr=bin_thr, invert=invert, to_grayscale=False)

    # 先构造临时数据集以拿到标签与类别映射
    tmp_ds = IdxDataset(train_images, train_labels, transform=None, class_to_idx=None)
    targets = tmp_ds.raw_labels.tolist()
    class_to_idx = tmp_ds.class_to_idx
    classes = [str(c) for c in sorted(class_to_idx.keys())]

    # 重新实例化带 transform 的数据集
    full_ds = IdxDataset(train_images, train_labels, transform=train_tf, class_to_idx=class_to_idx)
    train_idx, val_idx = stratified_split_indices(
        [class_to_idx[int(y)] for y in targets], val_ratio=val_split, seed=seed
    )

    train_subset = Subset(full_ds, train_idx)
    val_subset = Subset(IdxDataset(train_images, train_labels, transform=val_tf, class_to_idx=class_to_idx), val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"[info] train images: {Path(train_images)}")
    print(f"[info] train labels: {Path(train_labels)}")
    print(f"[info] samples: train={len(train_subset)}, val={len(val_subset)}, total={len(full_ds)}")
    print(f"[info] classes: {classes}")
    return train_loader, val_loader, classes, class_to_idx


def make_loaders_idx_with_val(
    train_images: str, train_labels: str,
    val_images: str,   val_labels: str,
    img_size: int, batch_size: int, num_workers: int,
    use_aug: bool, binarize: bool, bin_thr: float, invert: bool
):
    train_tf, val_tf = build_transforms(img_size, use_aug=use_aug, binarize=binarize, bin_thr=bin_thr, invert=invert, to_grayscale=False)

    # 用训练集确定类别映射
    tmp_train = IdxDataset(train_images, train_labels, transform=None, class_to_idx=None)
    class_to_idx = tmp_train.class_to_idx
    classes = [str(c) for c in sorted(class_to_idx.keys())]

    train_ds = IdxDataset(train_images, train_labels, transform=train_tf, class_to_idx=class_to_idx)
    val_ds   = IdxDataset(val_images,   val_labels,   transform=val_tf,   class_to_idx=class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"[info] train images: {Path(train_images)}, samples={len(train_ds)}")
    print(f"[info] train labels: {Path(train_labels)}")
    print(f"[info] val   images: {Path(val_images)}, samples={len(val_ds)}")
    print(f"[info] val   labels: {Path(val_labels)}")
    print(f"[info] classes: {classes}")
    return train_loader, val_loader, classes, class_to_idx


# -------------- DataLoader helpers for Folder (ImageFolder) --------------
def make_loaders_folder_split(
    train_root: str,
    img_size: int, batch_size: int, num_workers: int,
    val_split: float, seed: int, use_aug: bool, binarize: bool, bin_thr: float, invert: bool
):
    # 文件夹模式转灰度，确保单通道
    train_tf, val_tf = build_transforms(
        img_size, use_aug=use_aug, binarize=binarize, bin_thr=bin_thr, invert=invert, to_grayscale=True
    )

    tmp_ds = ImageFolder(train_root)
    targets = list(tmp_ds.targets)  # 已经是 0..C-1
    classes = list(tmp_ds.classes)  # 例如 ['0','1',...,'9']，按字典序排序
    class_to_idx = dict(tmp_ds.class_to_idx)

    full_ds = ImageFolder(train_root, transform=train_tf)
    train_idx, val_idx = stratified_split_indices(targets, val_ratio=val_split, seed=seed)

    train_subset = Subset(full_ds, train_idx)
    val_subset   = Subset(ImageFolder(train_root, transform=val_tf), val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"[info] train root: {Path(train_root)}")
    print(f"[info] samples: train={len(train_subset)}, val={len(val_subset)}, total={len(full_ds)}")
    print(f"[info] classes: {classes}")
    return train_loader, val_loader, classes, class_to_idx


def make_loaders_folder_with_val(
    train_root: str, val_root: str,
    img_size: int, batch_size: int, num_workers: int,
    use_aug: bool, binarize: bool, bin_thr: float, invert: bool
):
    train_tf, val_tf = build_transforms(
        img_size, use_aug=use_aug, binarize=binarize, bin_thr=bin_thr, invert=invert, to_grayscale=True
    )

    train_ds = ImageFolder(train_root, transform=train_tf)
    val_ds   = ImageFolder(val_root,   transform=val_tf)

    # 保证类别一致
    if train_ds.classes != val_ds.classes:
        raise ValueError(
            f"Train/Val classes mismatch.\n"
            f"train classes={train_ds.classes}\nval classes={val_ds.classes}\n"
            f"请确保两个根目录下的子文件夹（类别名）一致。"
        )
    classes = list(train_ds.classes)
    class_to_idx = dict(train_ds.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"[info] train root: {Path(train_root)}, samples={len(train_ds)}")
    print(f"[info] val   root: {Path(val_root)}, samples={len(val_ds)}")
    print(f"[info] classes: {classes}")
    return train_loader, val_loader, classes, class_to_idx


# -------------- Main train --------------
def main():
    ap = argparse.ArgumentParser(
        description="Train 32x32 printed digit classifier from IDX or folder dataset and export ONNX."
    )

    # 数据集类型
    ap.add_argument("--dataset", choices=["idx", "folder"], default="folder", help="dataset type: idx or folder")

    # 文件夹数据集参数
    ap.add_argument("--train-root", type=str, default=r"D:/sanjiao_data/train_data", help="root dir for training images (folder dataset)")
    ap.add_argument("--val-root",   type=str, default="", help="optional root dir for validation images (folder dataset)")

    # IDX 数据集参数（保留以兼容原有数据）
    ap.add_argument("--train-images", type=str, default=r"D:/sanjiao_data/idx/train-images.idx3-ubyte",
                    help="path to train-images.idx3-ubyte (IDX dataset)")
    ap.add_argument("--train-labels", type=str, default=r"D:/sanjiao_data/idx/train-labels.idx1-ubyte",
                    help="path to train-labels.idx1-ubyte (IDX dataset)")
    ap.add_argument("--val-images",   type=str, default=r"D:/sanjiao_data/idx/val-images.idx3-ubyte",
                    help="optional path to val-images.idx3-ubyte (IDX dataset)")
    ap.add_argument("--val-labels",   type=str, default=r"D:/sanjiao_data/idx/val-labels.idx1-ubyte",
                    help="optional path to val-labels.idx1-ubyte (IDX dataset)")

    ap.add_argument("--use-split-if-missing", action="store_true",
                    help="if val set missing or empty, split a val set from train (stratified)")

    ap.add_argument("--val-split", type=float, default=0.1, help="val ratio when splitting from train")
    ap.add_argument("--out-dir", type=str, default="build_artifacts")
    ap.add_argument("--epochs", type=int, default=45)

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")

    # 轻度增广（平移）
    ap.add_argument("--aug", action="store_true", help="enable mild translation augmentation")
    # 训练中可选二值化（白底黑字）
    ap.add_argument("--binarize", action="store_true")
    ap.add_argument("--bin-thr", type=float, default=0.75)

    # 反色：黑底白字 -> 白底黑字；你的数据是白底黑字，默认关闭
    ap.add_argument("--invert", dest="invert", action="store_true", help="invert to white background, black digits")
    ap.add_argument("--no-invert", dest="invert", action="store_false", help="disable inversion (recommended for white background)")
    ap.set_defaults(invert=False)

    # 可选：自定义导出的 ONNX 文件名
    ap.add_argument("--onnx-name", type=str, default="model.onnx")

    args = ap.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 构建数据加载器（不会加载任何旧权重，始终从零开始训练）
    if args.dataset == "folder":
        train_root = str(Path(args.train_root))
        if not os.path.isdir(train_root):
            raise SystemExit(f"[error] train root not found: {train_root}")

        val_root = str(Path(args.val_root)) if args.val_root else ""
        have_val = bool(val_root) and os.path.isdir(val_root)
        use_split = args.use_split_if_missing or (not have_val)

        if use_split:
            print("[info] using stratified split from training folder (no separate val root found or forced by flag).")
            train_loader, val_loader, classes, class_to_idx = make_loaders_folder_split(
                train_root,
                args.img_size, args.batch_size, args.num_workers,
                args.val_split, seed=args.seed, use_aug=args.aug,
                binarize=args.binarize, bin_thr=args.bin_thr, invert=args.invert
            )
        else:
            print("[info] using explicit validation folder.")
            train_loader, val_loader, classes, class_to_idx = make_loaders_folder_with_val(
                train_root, val_root,
                args.img_size, args.batch_size, args.num_workers,
                use_aug=args.aug, binarize=args.binarize, bin_thr=args.bin_thr, invert=args.invert
            )

    else:  # args.dataset == "idx"
        train_images = str(Path(args.train_images))
        train_labels = str(Path(args.train_labels))
        if not (os.path.isfile(train_images) and os.path.isfile(train_labels)):
            raise SystemExit(f"[error] train IDX files not found: images={train_images}, labels={train_labels}")

        val_images = str(Path(args.val_images)) if args.val_images else ""
        val_labels = str(Path(args.val_labels)) if args.val_labels else ""
        have_val = (val_images and os.path.isfile(val_images)) and (val_labels and os.path.isfile(val_labels))
        use_split = args.use_split_if_missing or (not have_val)

        if use_split:
            print("[info] using stratified split from training IDX (no separate val files found or forced by flag).")
            train_loader, val_loader, classes, class_to_idx = make_loaders_idx_split(
                train_images, train_labels,
                args.img_size, args.batch_size, args.num_workers,
                args.val_split, seed=args.seed, use_aug=args.aug,
                binarize=args.binarize, bin_thr=args.bin_thr, invert=args.invert
            )
        else:
            print("[info] using explicit validation IDX files.")
            train_loader, val_loader, classes, class_to_idx = make_loaders_idx_with_val(
                train_images, train_labels, val_images, val_labels,
                args.img_size, args.batch_size, args.num_workers,
                use_aug=args.aug, binarize=args.binarize, bin_thr=args.bin_thr, invert=args.invert
            )

    num_classes = len(classes)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.txt").write_text("\n".join(classes), encoding="utf-8")
    # 保存类别到索引的映射（按 classes 列表的顺序）
    (out_dir / "class_to_idx.json").write_text(
        json.dumps({c: i for i, c in enumerate(classes)}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model = SmallCNN(num_classes=num_classes).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_path = out_dir / "best.pth"

    for epoch in range(1, args.epochs+1):
        model.train()
        t0 = time.time()
        run_loss, total = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
            total += x.size(0)
        sch.step()

        train_loss = (run_loss / total) if total else 0.0
        val_loss, val_acc = evaluate(model, val_loader, device)  # numpy.float64
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} - train {train_loss:.6f}  val {val_loss:.6f}  acc {val_acc:.8f}  ({dt:.1f}s)")

        if val_acc > best_acc:
            best_acc = float(val_acc)  # 保持数值可序列化
            torch.save(model.state_dict(), best_path.as_posix())
            print(f"  Saved best weights: {best_path}  (acc={best_acc:.8f})")

            # 同步导出 ONNX（用刚保存的最佳权重）
            model.load_state_dict(torch.load(best_path, map_location=device))
            export_onnx(model, out_dir, args.img_size, onnx_name=args.onnx_name)

    # 训练结束，确保再次导出 ONNX（与最佳权重一致）
    model.load_state_dict(torch.load(best_path, map_location=device))
    export_onnx(model, out_dir, args.img_size, onnx_name=args.onnx_name)

    print(f"Labels saved to: {(out_dir / 'labels.txt').resolve()}")
    print("Preprocessing: Grayscale(1 for folder) -> Resize to 32x32 -> ToTensor([0,1]) -> optional Invert -> optional Binarize -> Normalize(0.5,0.5).")


if __name__ == "__main__":
    main()