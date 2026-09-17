import sys
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .data import OFFICIAL_RECU_MEAN, OFFICIAL_RECU_STD


def _workers(training_cfg: Dict, smoke: bool) -> int:
    workers = int(training_cfg.get("num_workers", 8))
    if sys.platform == "win32":
        workers = 0
    elif smoke and bool(training_cfg.get("windows_safe_smoke", True)):
        workers = 0
    return workers


def _train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(OFFICIAL_RECU_MEAN, OFFICIAL_RECU_STD),
    ])


def _eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(OFFICIAL_RECU_MEAN, OFFICIAL_RECU_STD),
    ])


def deterministic_train_split_indices(
    total: int,
    val_size: int,
    calibration_size: int,
    seed: int,
) -> Dict[str, list]:
    total = int(total)
    val_size = int(val_size)
    calibration_size = int(calibration_size)
    if val_size <= 0 or val_size >= total:
        raise ValueError("val_size must be in (0, total)")
    if calibration_size <= 0 or calibration_size > total - val_size:
        raise ValueError("calibration_size must fit inside the TRAIN-derived training partition")

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(total, generator=g).tolist()
    val = perm[:val_size]
    train = perm[val_size:]
    calibration = train[:calibration_size]
    return {
        "seed": int(seed),
        "total": total,
        "val_indices": val,
        "train_indices": train,
        "calibration_indices": calibration,
    }


def build_train_val_calibration_loaders(
    config: Dict,
    *,
    smoke: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, list]]:
    """Build TRAIN-only train/validation/calibration loaders.

    The official CIFAR-10 test set is intentionally not instantiated here, so
    coverage/ranking/checkpoint selection cannot accidentally inspect TEST.
    Validation and calibration use deterministic non-augmented transforms;
    training uses the historical ReCU augmentation.
    """
    dcfg = config["dataset"]
    tcfg = config["training"]
    scfg = config.get("study", {})

    root = dcfg.get("root", "./data")
    download = bool(dcfg.get("download", True))
    workers = _workers(tcfg, smoke)
    seed = int(scfg.get("split_seed", config.get("experiment", {}).get("seed", 123)))

    train_aug_ds = datasets.CIFAR10(
        root=root, train=True, download=download, transform=_train_transform()
    )
    train_eval_ds = datasets.CIFAR10(
        root=root, train=True, download=download, transform=_eval_transform()
    )

    total = len(train_aug_ds)
    if smoke:
        # Keep all three partitions disjoint while making local smoke tests fast.
        val_size = min(int(scfg.get("smoke_val_size", 256)), max(1, total // 10))
        calibration_size = min(int(scfg.get("smoke_calibration_size", 256)), total - val_size)
        train_cap = min(int(scfg.get("smoke_train_size", 1024)), total - val_size)
    else:
        val_size = int(scfg.get("val_size", 5000))
        calibration_size = int(scfg.get("calibration_size", 4096))
        train_cap = None

    split = deterministic_train_split_indices(
        total=total,
        val_size=val_size,
        calibration_size=calibration_size,
        seed=seed,
    )
    train_indices = split["train_indices"]
    if train_cap is not None:
        train_indices = train_indices[:train_cap]

    train_ds = Subset(train_aug_ds, train_indices)
    val_ds = Subset(train_eval_ds, split["val_indices"])
    calibration_ds = Subset(train_eval_ds, split["calibration_indices"])

    shuffle_gen = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg.get("batch_size", 256)),
        shuffle=True,
        generator=shuffle_gen,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tcfg.get("test_batch_size", 128)),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    calibration_loader = DataLoader(
        calibration_ds,
        batch_size=int(tcfg.get("test_batch_size", 128)),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    metadata = dict(split)
    metadata["train_indices_used"] = list(train_indices)
    metadata["is_smoke"] = bool(smoke)
    return train_loader, val_loader, calibration_loader, metadata


def build_official_test_loader(config: Dict, *, smoke: bool = False) -> DataLoader:
    """Build official TEST separately; call only after a checkpoint is frozen."""
    dcfg = config["dataset"]
    tcfg = config["training"]
    root = dcfg.get("root", "./data")
    download = bool(dcfg.get("download", True))
    workers = _workers(tcfg, smoke)

    test_ds = datasets.CIFAR10(
        root=root, train=False, download=download, transform=_eval_transform()
    )
    if smoke:
        test_ds = Subset(test_ds, list(range(min(512, len(test_ds)))))

    return DataLoader(
        test_ds,
        batch_size=int(tcfg.get("test_batch_size", 128)),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
