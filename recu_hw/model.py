import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .layers import ReCUBinaryConv2d
from .qrprelu import QuantizedRPReLU


class LambdaLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def option_a_shortcut(in_planes, planes, stride):
    if stride == 1 and in_planes == planes:
        return nn.Identity()

    # Exact CIFAR Option-A behavior used by official ReCU:
    # spatial decimation and symmetric channel zero padding.
    return LambdaLayer(
        lambda x: F.pad(
            x[:, :, ::2, ::2],
            (0, 0, 0, 0, planes // 4, planes // 4),
            "constant",
            0,
        )
    )


class ReCUBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        activation_mode="prelu",
        alpha_mode="float",
    ):
        super().__init__()
        self.conv1 = ReCUBinaryConv2d(
            in_planes, planes, 3, stride=stride, padding=1, bias=False,
            alpha_mode=alpha_mode,
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = ReCUBinaryConv2d(
            planes, planes, 3, stride=1, padding=1, bias=False,
            alpha_mode=alpha_mode,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if activation_mode == "prelu":
            self.post_act = nn.PReLU(planes)
        elif activation_mode == "qrprelu":
            self.post_act = QuantizedRPReLU(planes)
        else:
            raise ValueError(f"Unknown activation_mode: {activation_mode}")

        self.shortcut = option_a_shortcut(in_planes, planes, stride)

    def forward(self, x):
        # Exact official ReCU block topology:
        # BConv1 -> BN1 -> add shortcut -> save x1 -> Hardtanh
        # -> BConv2 -> BN2 -> add x1 -> PReLU
        out = self.bn1(self.conv1(x))
        out = out + self.shortcut(x)
        x1 = out

        out = F.hardtanh(out)
        out = self.bn2(self.conv2(out))
        out = out + x1
        return self.post_act(out)


class ReCUResNet20(nn.Module):
    def __init__(
        self,
        num_classes=10,
        activation_mode="prelu",
        alpha_mode="float",
    ):
        super().__init__()
        self.in_planes = 16
        self.activation_mode = activation_mode
        self.alpha_mode = alpha_mode

        # Official ReCU first layer is full-precision.
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 3, 1)
        self.layer2 = self._make_layer(32, 3, 2)
        self.layer3 = self._make_layer(64, 3, 2)

        # Official ReCU head: GAP -> BN1d -> full-precision Linear.
        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, num_classes)

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(
                ReCUBasicBlock(
                    self.in_planes,
                    planes,
                    stride=s,
                    activation_mode=self.activation_mode,
                    alpha_mode=self.alpha_mode,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)), inplace=False)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        return self.linear(out)


def build_recu_resnet20(config):
    m = config.get("model", {})
    return ReCUResNet20(
        num_classes=int(m.get("num_classes", 10)),
        activation_mode=m.get("activation_mode", "prelu"),
        alpha_mode=m.get("alpha_mode", "float"),
    )


def parameter_breakdown(model):
    groups = {
        "total": 0,
        "conv_weight": 0,
        "bn_trainable": 0,
        "linear": 0,
        "alpha": 0,
        "prelu": 0,
        "qrprelu": 0,
        "other": 0,
    }

    seen = set()

    for name, module in model.named_modules():
        if isinstance(module, ReCUBinaryConv2d):
            if id(module.weight) not in seen:
                groups["conv_weight"] += module.weight.numel()
                seen.add(id(module.weight))
            if module.bias is not None and id(module.bias) not in seen:
                groups["conv_weight"] += module.bias.numel()
                seen.add(id(module.bias))
            groups["alpha"] += module.alpha.numel()
            seen.add(id(module.alpha))
        elif isinstance(module, nn.Conv2d):
            for p in module.parameters(recurse=False):
                if id(p) not in seen:
                    groups["conv_weight"] += p.numel()
                    seen.add(id(p))
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for p in module.parameters(recurse=False):
                if id(p) not in seen:
                    groups["bn_trainable"] += p.numel()
                    seen.add(id(p))
        elif isinstance(module, nn.Linear):
            for p in module.parameters(recurse=False):
                if id(p) not in seen:
                    groups["linear"] += p.numel()
                    seen.add(id(p))
        elif isinstance(module, nn.PReLU):
            for p in module.parameters(recurse=False):
                if id(p) not in seen:
                    groups["prelu"] += p.numel()
                    seen.add(id(p))
        elif isinstance(module, QuantizedRPReLU):
            for p in module.parameters(recurse=False):
                if id(p) not in seen:
                    groups["qrprelu"] += p.numel()
                    seen.add(id(p))

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups["total"] = total
    accounted = sum(v for k, v in groups.items() if k not in {"total", "other"})
    groups["other"] = total - accounted
    return groups
