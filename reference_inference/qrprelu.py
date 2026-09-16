"""Exact integer QRPReLU branch from H2TraceAdapter._qrprelu_integer."""

import numpy as np

from .integer_ops import finite


def qrprelu_integer(q_in, node, arrays, record, overflow, trace):
    trace_prefix = "qrprelu." + node.removesuffix(".qrprelu_output")
    input_shift = int(record["input_shift"])
    inner_shift = int(record["param_shift"])
    output_shift = int(record["output_shift"])
    a_exp = arrays[node + ".slope_exponent"].astype(np.int64)
    xi1 = arrays[node + ".xi1_q"].astype(np.int64)
    xi2 = arrays[node + ".xi2_q"].astype(np.int64)
    shape = (1, len(a_exp), 1, 1)
    q_x = q_in.astype(np.int64) << (inner_shift - input_shift)
    q_inner = q_x + xi1.reshape(shape)
    q_inner = finite(q_inner, record["inner_bits"], trace_prefix + ".inner_add", overflow)
    deltas = output_shift + a_exp - inner_shift
    powers = np.left_shift(np.int64(1), deltas).reshape(shape)
    q_neg = q_inner * powers + (xi2.reshape(shape) << (output_shift - inner_shift))
    q_pos = q_in.astype(np.int64) << (output_shift - input_shift)
    branch = q_in >= 0
    q_out = np.where(branch, q_pos, q_neg)
    q_neg = finite(q_neg, record["bits"], trace_prefix + ".negative_branch", overflow)
    q_out = finite(q_out, record["bits"], trace_prefix + ".output", overflow)
    trace(trace_prefix + ".branch_positive", branch.astype(np.uint8))
    trace(trace_prefix + ".inner_add", q_inner)
    trace(trace_prefix + ".negative_branch", q_neg)
    trace(trace_prefix + ".output", q_out)
    return q_out, output_shift
