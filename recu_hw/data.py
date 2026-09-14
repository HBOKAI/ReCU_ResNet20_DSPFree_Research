import sys

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Official ReCU CIFAR-10 normalization uses ImageNet mean/std.
OFFICIAL_RECU_MEAN = (0.485, 0.456, 0.406)
OFFICIAL_RECU_STD = (0.229, 0.224, 0.225)


def build_recu_loaders(config, smoke=False):
    dcfg = config["dataset"]
    tcfg = config["training"]

    root = dcfg.get("root", "./data")
    download = bool(dcfg.get("download", True))
    workers = int(tcfg.get("num_workers", 8))
    # The Windows execution environment used for these runs can reject
    # multiprocessing DataLoader pipes (WinError 5).  Keep the data protocol
    # identical while using a single loader process on Windows.
    if sys.platform == "win32":
        workers = 0
    elif smoke and bool(tcfg.get("windows_safe_smoke", True)):
        workers = 0

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(OFFICIAL_RECU_MEAN, OFFICIAL_RECU_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(OFFICIAL_RECU_MEAN, OFFICIAL_RECU_STD),
    ])

    train_ds = datasets.CIFAR10(
        root=root, train=True, download=download, transform=train_tf
    )
    test_ds = datasets.CIFAR10(
        root=root, train=False, download=download, transform=test_tf
    )

    if smoke:
        train_ds = Subset(train_ds, list(range(min(1024, len(train_ds)))))
        test_ds = Subset(test_ds, list(range(min(512, len(test_ds)))))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg.get("batch_size", 256)),
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=int(tcfg.get("test_batch_size", 128)),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader
