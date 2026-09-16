"""Finite-width integer operations. No PyTorch dependency."""

import numpy as np


def signed_range(bits):
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def finite(q, bits, name, overflow):
    lo, hi = signed_range(int(bits))
    bad = int(np.count_nonzero((q < lo) | (q > hi)))
    overflow[name] = overflow.get(name, 0) + bad
    return np.clip(q, lo, hi).astype(np.int64, copy=False)


def right_shift_away(q, amount):
    """H2 residual alignment: nearest integer, ties away from zero."""
    q = np.asarray(q, dtype=np.int64)
    if amount <= 0:
        return q
    offset = 1 << (amount - 1)
    magnitude = (np.abs(q) + offset) >> amount
    return np.where(q < 0, -magnitude, magnitude)


def right_shift_even(q, amount):
    """H1 boundary: torch.round's nearest integer, ties to even."""
    q = np.asarray(q, dtype=np.int64)
    if amount <= 0:
        return q
    magnitude = np.abs(q)
    base = magnitude >> amount
    remainder = magnitude & ((1 << amount) - 1)
    half = 1 << (amount - 1)
    rounded = base + ((remainder > half) | ((remainder == half) & ((base & 1) != 0)))
    return np.where(q < 0, -rounded, rounded)


def align_away(q, source_shift, target_shift):
    delta = int(target_shift) - int(source_shift)
    q = np.asarray(q, dtype=np.int64)
    return q << delta if delta >= 0 else right_shift_away(q, -delta)


def align_even(q, source_shift, target_shift):
    delta = int(target_shift) - int(source_shift)
    q = np.asarray(q, dtype=np.int64)
    return q << delta if delta >= 0 else right_shift_even(q, -delta)


def hardtanh_integer(q, shift):
    limit = 1 << int(shift)
    return np.clip(q, -limit, limit)
