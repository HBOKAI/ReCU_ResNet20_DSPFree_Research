"""Workspace-specific glue for the H1-S/H1-MP search pack."""

import copy
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .h1_model_adapter import H1ModelAdapter
from .r5t_data import build_r5t_loaders


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_r8b_h0_model(r8b_checkpoint: str, h0_results_json: str, device):
    # Reuse the already verified R8B loader and H0 q+shift reconstruction.
    from tools.run_h1_integer_sweep import apply_h0_int6_biases, load_r8b

    model, _ = load_r8b(_resolve(r8b_checkpoint), device)
    apply_h0_int6_biases(model, _resolve(h0_results_json))
    model.eval()
    return model


def build_h1_search_loaders(
    seed: int,
    calibration_samples: int,
    search_validation_samples: int,
):
    """Build fixed, disjoint train-derived calibration/search/test loaders."""
    if calibration_samples <= 0 or search_validation_samples <= 0:
        raise ValueError("calibration and search-validation sizes must be positive")
    if calibration_samples + search_validation_samples > 50000:
        raise ValueError("train subsets exceed CIFAR-10 train size")

    random.seed(seed)
    torch.manual_seed(seed)
    root = ROOT / "data"
    raw_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(
        root=str(root), train=True, transform=raw_transform, download=True
    )
    test_dataset = datasets.CIFAR10(
        root=str(root), train=False, transform=raw_transform, download=True
    )
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(len(train_dataset), generator=generator).tolist()
    calibration_indices = permutation[:calibration_samples]
    search_indices = permutation[calibration_samples:calibration_samples + search_validation_samples]
    if set(calibration_indices).intersection(search_indices):
        raise RuntimeError("calibration and search-validation subsets overlap")

    calibration = Subset(train_dataset, calibration_indices)
    search_validation = Subset(train_dataset, search_indices)
    kwargs = {
        "batch_size": 128,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": True,
    }
    calibration_loader = DataLoader(calibration, **kwargs)
    search_loader = DataLoader(search_validation, **kwargs)
    test_loader = DataLoader(test_dataset, **kwargs)
    # Expose the split for reproducibility tests/reporting without changing
    # the standard DataLoader API.
    calibration_loader.h1_indices = calibration_indices
    search_loader.h1_indices = search_indices
    test_loader.h1_indices = list(range(len(test_dataset)))
    return calibration_loader, search_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    predictor = getattr(model, "_h1_adapter", model)
    predictor.model.eval() if isinstance(predictor, H1ModelAdapter) else model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        correct += int((predictor(x).argmax(1) == y).sum().item())
        total += int(y.size(0))
    return 100.0 * correct / max(total, 1)


def make_h1_adapter(model):
    adapter = H1ModelAdapter(model)
    # Existing extension runners call evaluate(model), so attach the adapter
    # as a non-module runtime view; it is not part of state_dict/checkpoints.
    model._h1_adapter = adapter
    return adapter
