from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from .common import Bottleneck, C3, Concat, Conv, SPP, StemBlock
from ..utils.torch_utils import fuse_conv_and_bn, model_info


class Detect(nn.Module):
    stride = None
    export_cat = False

    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5 + 8
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2 if anchors else 0
        self.grid = [torch.zeros(1)] * self.nl
        if anchors:
            a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        else:
            a = torch.empty(0)
        self.register_buffer("anchors", a)
        self.register_buffer(
            "anchor_grid",
            a.clone().view(self.nl, 1, -1, 1, 1, 2) if anchors else torch.empty(0),
        )
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch) if ch else nn.ModuleList()

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = torch.full_like(x[i], 0)
                class_range = list(range(5)) + list(range(13, 13 + self.nc))
                y[..., class_range] = x[i][..., class_range].sigmoid()
                y[..., 5:13] = x[i][..., 5:13]
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2.0) ** 2 * self.anchor_grid[i]
                y[..., 5:7] = y[..., 5:7] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]
                y[..., 7:9] = y[..., 7:9] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]
                y[..., 9:11] = y[..., 9:11] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]
                y[..., 11:13] = y[..., 11:13] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        try:
            yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
        except TypeError:
            yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx))
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg=None, ch=3, nc=None):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else {}
        self.model = nn.Sequential()
        self.save = []
        self.names = [str(i) for i in range(int(nc or 0))]
        self.stride = torch.tensor([8.0, 16.0, 32.0])

    def forward(self, x, augment=False, profile=False):
        if augment:
            return self.forward_once(x, profile), None
        return self.forward_once(x, profile)

    def forward_once(self, x, profile=False):
        _ = profile
        y = []
        for m in self.model:
            if getattr(m, "f", -1) != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if getattr(m, "i", -1) in self.save else None)
        return x

    def fuse(self):
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.fuseforward
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)

    def nms(self, mode=True):
        _ = mode
        return self

    def autoshape(self):
        return self


__all__ = [
    "Conv",
    "StemBlock",
    "Bottleneck",
    "C3",
    "SPP",
    "Concat",
    "Detect",
    "Model",
]
