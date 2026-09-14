"""Workspace-specific H1 adapter contract.

Codex should implement `recu_hw/h1_model_adapter.py` against the actual R8B/R4
classes in the user's workspace.

Required API:

class H1ModelAdapter:
    def __init__(self, r8b_model): ...
    def list_quant_nodes(self) -> list[str]: ...
    def set_observer(self, observer_or_none): ...
    def set_fake_quant(self, fake_quant_or_none): ...
    def verify_integer_residual_alignment(self, shift_map, bits) -> dict: ...

H1 quantization nodes should include true multi-bit hardware states, especially:
- stem post-affine/Hardtanh state used by shortcut/BConv input
- each block residual save x1
- each block second residual-add output
- each QRPReLU/block output
- stage-transition residual states
- head affine output / FC input

Do NOT quantize in H1:
- binary-convolution accumulators
- GAP accumulator

Keep H0 INT6 affine/FC bias fixed.
Residual branches must be scale-aligned by integer shifts before integer add.
Generic forward hooks are insufficient if residual adds occur inline; adapt the
actual block forward path or use a faithful wrapper.
"""
