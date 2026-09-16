"""Pure NumPy/Python integer inference for the frozen H2A-v2 profile.

After parameter export this module never imports PyTorch or the model package.
Input is unsigned CIFAR-10 RGB uint8 in CHW or NCHW order. Each output is an
INT24 logit at the FC bias-add shift; argmax is the prediction.
"""

import json
from pathlib import Path

import numpy as np

from .binary_conv import binary_conv
from .integer_ops import align_away, align_even, finite, hardtanh_integer
from .qrprelu import qrprelu_integer


HERE = Path(__file__).resolve().parent


def thermometer_uint8(rgb):
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim == 3:
        rgb = rgb[None, ...]
    if rgb.ndim != 4 or rgb.shape[1:] != (3, 32, 32):
        raise ValueError("Expected uint8 RGB [N,3,32,32]")
    # torch.round(p/8): p is uint8; ties go to an even quotient.
    quotient = rgb.astype(np.int16) >> 3
    half = (rgb & 7) == 4
    n = quotient + ((rgb & 7) > 4) + (half & ((quotient & 1) != 0))
    index = np.arange(32, dtype=np.int16).reshape(1, 1, 32, 1, 1)
    bits = index >= (32 - n[:, :, None, :, :])
    return (bits.astype(np.int8) * 2 - 1).reshape(rgb.shape[0], 96, 32, 32)


class IntegerReference:
    def __init__(self, config_path=HERE / "model_config.json", params_path=HERE / "params_export/inference_params.npz"):
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        with np.load(params_path, allow_pickle=False) as archive:
            self.params = {key: archive[key] for key in archive.files}
        self.overflow = {}
        self.trace = {}
        self.keep_trace = False

    def _trace(self, name, array):
        if self.keep_trace:
            self.trace[name] = np.asarray(array).copy()

    def _boundary(self, name, raw, source_shift):
        h1 = self.config["h1mp"]
        shift = int(h1["shift_by_node"][name])
        bits = int(h1["bits_by_node"][name])
        q = align_even(raw, source_shift, shift)
        q = finite(q, bits, "h1." + name, self.overflow)
        self._trace("h1." + name, q)
        return q, shift

    def _binary(self, name, x, weight_name, stride, width_key):
        q = binary_conv(np.sign(x).astype(np.int8), self.params[weight_name + ".weight_sign"], stride=stride)
        width = self.config["h2a"]["binary_convolution"][width_key]["signed_acc_bits"]
        q = finite(q, width, "binary." + name, self.overflow)
        self._trace("binary." + name, q)
        return q

    def _affine(self, node, module, input_q, input_shift=0):
        record = self.config["h2a"]["affine_intermediates"][node]
        bias_info = self.config["h0_bias"][module]
        common_shift = int(record["shift"])
        bias_shift = int(bias_info["shift"])
        signs = self.params[module + ".sign"].astype(np.int64)
        exponents = self.params[module + ".exponent"].astype(np.int64)
        bias = self.params[module + ".bias_q"].astype(np.int64)
        channel_shape = (1, len(signs)) + (1,) * (input_q.ndim - 2)
        deltas = common_shift + exponents - int(input_shift)
        if np.any(deltas < 0):
            raise ValueError("Negative affine left shift: " + node)
        shifted_term = np.left_shift(input_q.astype(np.int64), deltas.reshape(channel_shape)) * signs.reshape(channel_shape)
        aligned_bias = (bias << (common_shift - bias_shift)).reshape(channel_shape)
        output = finite(shifted_term + aligned_bias, record["bits"], "affine." + node + ".add_output", self.overflow)
        self._trace("affine." + node + ".shifted_term", shifted_term)
        self._trace("affine." + node + ".bias_aligned", aligned_bias)
        self._trace("affine." + node + ".add_output", output)
        return output, common_shift

    def _option_a(self, q):
        channels = q.shape[1]
        result = np.zeros((q.shape[0], channels * 2, q.shape[2] // 2, q.shape[3] // 2), dtype=np.int64)
        result[:, channels // 2:channels // 2 + channels] = q[:, :, ::2, ::2]
        return result

    def _residual(self, node, branch_a, branch_b):
        q_a, shift_a = branch_a
        q_b, shift_b = branch_b
        record = self.config["h2a"]["residuals"][node]
        shift = int(record["output_shift"])
        a = align_away(q_a, shift_a, shift)
        b = align_away(q_b, shift_b, shift)
        self._trace("residual." + node + ".align_a", a)
        self._trace("residual." + node + ".align_b", b)
        summed = finite(a + b, record["bits"], "residual." + node + ".add_output", self.overflow)
        self._trace("residual." + node + ".add_output", summed)
        return self._boundary(node, summed, shift)

    def _fc(self, head):
        head_q, head_shift = head
        record = self.config["h2a"]["fc"]
        common_shift = int(record["common_shift"])
        signs = self.params["linear.weight_sign"]
        exps = self.params["linear.weight_exponent"]
        terms = np.zeros((head_q.shape[0], 10, 64), dtype=np.int64)
        acc = np.zeros((head_q.shape[0], 10), dtype=np.int64)
        for cls in range(10):
            for feature in range(64):
                delta = common_shift + int(exps[cls, feature]) - int(head_shift)
                term = head_q[:, feature].astype(np.int64) << delta
                if signs[cls, feature] < 0:
                    term = -term
                terms[:, cls, feature] = term
                acc[:, cls] += term
        self._trace("fc.shifted_products", terms)
        acc = finite(acc, record["accumulator_bits"], "fc.accumulator", self.overflow)
        self._trace("fc.accumulator", acc)
        bias_shift = self.config["h0_bias"]["linear"]["shift"]
        out_shift = int(record["bias_add_shift"])
        aligned = align_away(acc, common_shift, out_shift)
        bias = self.params["linear.bias_q"].astype(np.int64) << (out_shift - bias_shift)
        logits = finite(aligned + bias[None, :], record["bias_add_bits"], "fc.bias_add", self.overflow)
        self._trace("fc.bias_add", logits)
        logits = finite(logits, record["final_logits_bits"], "final_logits", self.overflow)
        self._trace("final_logits", logits)
        return logits

    def forward_uint8(self, rgb, trace=False):
        self.overflow = {}
        self.trace = {}
        self.keep_trace = bool(trace)
        encoded = thermometer_uint8(rgb)
        self._trace("thermometer", encoded)
        stem_acc = self._binary("stem.signed_acc", encoded, "conv1", 1, "stem_cin96")
        stem_affine = self._affine("stem.post_hardtanh", "stem_affine", stem_acc)
        stem = hardtanh_integer(*stem_affine)
        state = self._boundary("stem.post_hardtanh", stem, stem_affine[1])
        for prefix in self.config["block_names"]:
            stage = int(prefix[5])
            index = int(prefix[7])
            transition = stage > 1 and index == 0
            cin = state[0].shape[1]
            width = "cin" + str(cin)
            stride = 2 if transition else 1
            conv1 = self._binary(prefix + ".conv1_signed_acc", state[0], prefix + ".conv1", stride, width)
            first = self._boundary(prefix + ".first_affine", *self._affine(prefix + ".first_affine", prefix + ".aff1", conv1))
            if transition:
                shortcut_q = self._option_a(state[0])
                self._trace("residual." + prefix + ".stage_shortcut_option_a", shortcut_q)
                shortcut = self._boundary(prefix + ".stage_shortcut", shortcut_q, state[1])
            else:
                shortcut = state
            x1 = self._residual(prefix + ".x1", first, shortcut)
            pre = self._boundary(prefix + ".pre_bconv2_hardtanh", hardtanh_integer(*x1), x1[1])
            conv2 = self._binary(prefix + ".conv2_signed_acc", pre[0], prefix + ".conv2", 1, "cin" + str(pre[0].shape[1]))
            second = self._boundary(prefix + ".second_affine", *self._affine(prefix + ".second_affine", prefix + ".aff2", conv2))
            second_add = self._residual(prefix + ".second_add", second, x1)
            node = prefix + ".qrprelu_output"
            qrp_raw = qrprelu_integer(second_add[0], node, self.params, self.config["h2a"]["qrprelu"][node], self.overflow, self._trace)
            state = self._boundary(node, *qrp_raw)
            self._trace(prefix + ".stage_output", state[0])
        last_q, last_shift = state
        gap_plan = self.config["h2a"]["gap"]
        gap = last_q.reshape(last_q.shape[0], 64, 64).sum(axis=2, dtype=np.int64)
        gap = finite(gap, gap_plan["sum_bits"], "gap.sum", self.overflow)
        self._trace("gap.sum", gap)
        self._trace("gap.output", gap)
        gap_shift = last_shift + 6
        head_raw = self._affine("head.affine_output", "head_affine", gap, gap_shift)
        head = self._boundary("head.affine_output", *head_raw)
        logits = self._fc(head)
        return logits

    def predict_uint8(self, rgb):
        return self.forward_uint8(rgb).argmax(axis=1)


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    with (HERE.parent / "data/cifar-10-batches-py/test_batch").open("rb") as handle:
        batch = pickle.load(handle, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32)
    labels = np.asarray(batch[b"labels"])
    reference = IntegerReference()
    correct = 0
    for index in range(args.count):
        pred = int(reference.predict_uint8(images[index])[0])
        correct += pred == int(labels[index])
    print({"count": args.count, "correct": correct, "accuracy": 100 * correct / args.count})
