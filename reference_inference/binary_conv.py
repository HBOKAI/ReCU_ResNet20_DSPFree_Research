"""Ternary-aware XNOR/popcount convolution over exported binary weights.

The two stored binary values are 0 = -1 and 1 = +1. Zero input activations
(torch.sign(0)) and spatial padding are masked; for each dot product
S = 2*P - N, where N is the number of nonzero participating terms.
"""

import numpy as np


def binary_conv(input_sign, weight_sign, stride=1, padding=1):
    x = np.asarray(input_sign, dtype=np.int8)
    w = np.asarray(weight_sign, dtype=np.int8)
    if x.ndim != 4 or w.ndim != 4 or x.shape[1] != w.shape[1] or w.shape[2:] != (3, 3):
        raise ValueError("Expected NCHW input and OI33 weights")
    batch, channels, height, width = x.shape
    out_channels = w.shape[0]
    out_h = (height + 2 * padding - 3) // stride + 1
    out_w = (width + 2 * padding - 3) // stride + 1
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out = np.zeros((batch, out_channels, out_h, out_w), dtype=np.int32)
    # Each kernel point is a separate XNOR/popcount bank. Packing the channel
    # axis changes no addition, threshold, or rounding semantics.
    for ky in range(3):
        for kx in range(3):
            patch = padded[:, :, ky:ky + stride * out_h:stride, kx:kx + stride * out_w:stride]
            patch = patch.transpose(0, 2, 3, 1).reshape(-1, channels)
            xb = np.packbits(patch > 0, axis=1, bitorder="little")
            xm = np.packbits(patch != 0, axis=1, bitorder="little")
            wb = np.packbits(w[:, :, ky, kx] > 0, axis=1, bitorder="little")
            wm = np.packbits(w[:, :, ky, kx] != 0, axis=1, bitorder="little")
            valid = np.bitwise_and(xm[:, None, :], wm[None, :, :])
            matched = np.bitwise_and(np.bitwise_not(np.bitwise_xor(xb[:, None, :], wb[None, :, :])), valid)
            pop = np.bitwise_count(matched).sum(axis=2, dtype=np.int32)
            terms = np.bitwise_count(valid).sum(axis=2, dtype=np.int32)
            out += (2 * pop - terms).reshape(batch, out_h, out_w, out_channels).transpose(0, 3, 1, 2)
    return out.astype(np.int64)
