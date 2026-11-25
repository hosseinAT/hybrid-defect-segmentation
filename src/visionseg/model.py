"""A small residual encoder with explicit masked window and cross-scale attention."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

VARIANTS = {"cnn": (False, False), "attention": (True, False),
            "fusion": (False, True), "hybrid": (True, True)}


def partition(x, window):
    """BHWC -> (B * number_of_windows, window**2, C)."""
    b, h, w, c = x.shape
    return x.reshape(b, h // window, window, w // window, window, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, c)


def reverse(tokens, window, height, width, batch):
    c = tokens.shape[-1]
    return tokens.reshape(batch, height // window, width // window, window, window, c).permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, c)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                 nn.GroupNorm(4, cout), nn.GELU(),
                                 nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                 nn.GroupNorm(4, cout), nn.GELU())
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class WindowAttention(nn.Module):
    def __init__(self, dim=32, heads=4, window=4, shift=0):
        super().__init__()
        if dim % heads or not 0 <= shift < window:
            raise ValueError("Invalid head dimension or shift")
        self.dim, self.heads, self.window, self.shift = dim, heads, window, shift
        self.norm = nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(torch.zeros(heads, (2 * window - 1) ** 2))
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)
        delta = coords[:, :, None] - coords[:, None, :] + window - 1
        self.register_buffer("relative_index", delta[0] * (2 * window - 1) + delta[1])
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def _mask(self, h, w, hp, wp, device):
        k, s = self.window, self.shift
        valid = torch.zeros((1, hp, wp, 1), dtype=torch.bool, device=device)
        valid[:, :h, :w] = True
        region = torch.zeros((1, hp, wp, 1), device=device)
        if s:
            valid = torch.roll(valid, shifts=(-s, -s), dims=(1, 2))
            # The three boundary regions separate cyclic wrap-around tokens.
            hs, ws = (slice(0, -k), slice(-k, -s), slice(-s, None)), (slice(0, -k), slice(-k, -s), slice(-s, None))
            for i, hi in enumerate(hs):
                for j, wi in enumerate(ws):
                    region[:, hi, wi] = 3 * i + j
        labels = partition(region, k).squeeze(-1)
        keys = partition(valid, k).squeeze(-1)
        return (labels[:, :, None] == labels[:, None, :]) & keys[:, None, :]

    def forward(self, x):
        b, c, h, w = x.shape
        k, s = self.window, self.shift
        hp, wp = ((h + k - 1) // k) * k, ((w + k - 1) // k) * k
        padded = F.pad(x, (0, wp - w, 0, hp - h)).permute(0, 2, 3, 1)
        if s:
            padded = torch.roll(padded, shifts=(-s, -s), dims=(1, 2))
        tokens = partition(padded, k)
        n, length, _ = tokens.shape
        q, key, value = self.qkv(self.norm(tokens)).reshape(n, length, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        logits = (q @ key.transpose(-1, -2)) / (c // self.heads) ** .5
        logits = logits + self.relative_bias[:, self.relative_index][None]
        allowed = self._mask(h, w, hp, wp, x.device).repeat(b, 1, 1)
        # Finite sentinel keeps all-padding query rows finite; those rows are cropped.
        logits = logits.masked_fill(~allowed[:, None], torch.finfo(logits.dtype).min)
        update = (logits.softmax(-1) @ value).transpose(1, 2).reshape(n, length, c)
        tokens = tokens + self.proj(update)
        tokens = tokens + self.mlp(tokens)
        out = reverse(tokens, k, hp, wp, b)
        if s:
            out = torch.roll(out, shifts=(s, s), dims=(1, 2))
        return out[:, :h, :w].permute(0, 3, 1, 2).contiguous()


class CrossScaleFusion(nn.Module):
    """Fine-grid queries attend to 16 pooled context tokens."""
    def __init__(self, dim=32):
        super().__init__()
        self.q, self.k, self.v, self.out = [nn.Conv2d(dim, dim, 1) for _ in range(4)]

    def forward(self, high, low):
        b, c, h, w = high.shape
        context = F.adaptive_avg_pool2d(low, (4, 4))
        q = self.q(high).flatten(2).transpose(1, 2)
        k = self.k(context).flatten(2)
        v = self.v(context).flatten(2).transpose(1, 2)
        attention = (q @ k / c ** .5).softmax(-1)
        return high + self.out((attention @ v).transpose(1, 2).reshape(b, c, h, w))


class DefectNet(nn.Module):
    def __init__(self, variant="hybrid"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        attention, fusion = VARIANTS[variant]
        self.stem, self.down, self.bottleneck = ConvBlock(1, 16), ConvBlock(16, 32), ConvBlock(32, 32)
        self.attention = nn.Sequential(WindowAttention(shift=0), WindowAttention(shift=2)) if attention else nn.Identity()
        self.fusion = CrossScaleFusion() if fusion else None
        self.decode, self.head = ConvBlock(48, 16), nn.Conv2d(16, 1, 1)

    def forward(self, x):
        a = self.stem(x)
        b = self.down(F.avg_pool2d(a, 2))
        c = self.attention(self.bottleneck(F.avg_pool2d(b, 2)))
        b = b + F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False)
        if self.fusion is not None:
            b = self.fusion(b, c)
        return self.head(self.decode(torch.cat([a, F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)], 1)))


def defect_loss(logits, target, dice_weight=.5, gamma=2.):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-bce)
    alpha = torch.where(target > .5, .75, .25)
    focal = (alpha * (1 - pt) ** gamma * bce).mean()
    p = logits.sigmoid()
    # Per-image Dice prevents a larger defective component dominating a batch.
    dims = (1, 2, 3)
    dice = 1 - ((2 * (p * target).sum(dims) + 1) / (p.sum(dims) + target.sum(dims) + 1)).mean()
    return (1 - dice_weight) * focal + dice_weight * dice
