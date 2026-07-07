from __future__ import annotations

import time

import torch
import torch.nn as nn


def time_synchronized() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    ).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))

    if conv.bias is None:
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device)
    else:
        b_conv = conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fused


def model_info(model: nn.Module, verbose: bool = False, img_size: int = 640) -> None:
    # Intentionally lightweight for embedded inference-only runtime.
    _ = (model, verbose, img_size)


__all__ = ["time_synchronized", "fuse_conv_and_bn", "model_info"]
