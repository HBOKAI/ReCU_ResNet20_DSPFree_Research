import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_r5t_loaders(cfg, smoke=False):
    """Raw RGB [0,1] loaders. No CIFAR Normalize: thermometer encoding uses pixel intensity."""
    root = cfg["dataset"].get("root", "./data")
    download = bool(cfg["dataset"].get("download", True))

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.CIFAR10(root=root, train=True, transform=train_tf, download=download)
    test_ds = datasets.CIFAR10(root=root, train=False, transform=test_tf, download=download)

    if smoke:
        train_ds = torch.utils.data.Subset(train_ds, range(min(1024, len(train_ds))))
        test_ds = torch.utils.data.Subset(test_ds, range(min(512, len(test_ds))))

    tr = cfg["training"]
    workers = 0 if (
        sys.platform == "win32"
        or (smoke and tr.get("windows_safe_smoke", True))
    ) else int(tr.get("num_workers", 8))
    train_loader = DataLoader(
        train_ds, batch_size=int(tr.get("batch_size", 128)), shuffle=True,
        num_workers=workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=int(tr.get("test_batch_size", 128)), shuffle=False,
        num_workers=workers, pin_memory=True,
    )
    return train_loader, test_loader
