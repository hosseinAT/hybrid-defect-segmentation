"""Procedural surfaces and an audited CSV contract for private image data."""
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

SPLITS = {"train", "val", "cal", "test"}


class SyntheticSurface(Dataset):
    """Index-stable samples; independent split seeds prevent shared base images."""

    def __init__(self, n: int, seed: int, size: int = 64, normal_only: bool = False):
        if n < 1 or size < 32:
            raise ValueError("Require n >= 1 and size >= 32")
        self.n, self.seed, self.size, self.normal_only = n, seed, size, normal_only

    def __len__(self):
        return self.n

    def kind(self, i):
        return "good" if self.normal_only or i % 2 == 0 else ("scratch", "pits", "patch")[(i // 2) % 3]

    def sample_id(self, i):
        return f"procedural-{self.seed}-{i:05d}"

    def __getitem__(self, i):
        if not 0 <= i < self.n:
            raise IndexError(i)
        r = np.random.default_rng(np.random.SeedSequence([self.seed, i]))
        s = self.size
        y, x = np.mgrid[:s, :s]
        theta = r.uniform(-.2, .2)
        base = .5 + .09 * np.sin((x * np.sin(theta) + y * np.cos(theta)) * r.uniform(.4, 1.1) + r.uniform(0, 6))
        base += r.normal(0, .025, (s, s)) + r.uniform(-.08, .08) + r.uniform(-.08, .08) * x / s
        mask = Image.new("L", (s, s))
        draw = ImageDraw.Draw(mask)
        a, b = map(int, r.integers(8, s - 8, 2))
        kind = self.kind(i)
        if kind == "scratch":
            c = int(np.clip(a + r.integers(-20, 21), 1, s - 2))
            e = int(np.clip(b + r.integers(8, 24), 1, s - 2))
            draw.line([(a, b), (c, e)], fill=255, width=int(r.integers(1, 4)))
        elif kind == "pits":
            for _ in range(3):
                a, b = map(int, r.integers(5, s - 5, 2))
                q = int(r.integers(2, 5))
                draw.ellipse((a - q, b - q, a + q, b + q), fill=255)
        elif kind == "patch":
            draw.rectangle((a, b, min(s - 1, a + 7), min(s - 1, b + 3)), fill=255)
        target = np.asarray(mask, dtype=np.float32) / 255
        base += target * r.choice([-1, 1]) * r.uniform(.15, .35)
        image = np.clip(base, 0, 1).astype(np.float32)
        return torch.from_numpy(image[None]), torch.from_numpy(target[None].copy())


def _relative_file(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or Path(value).is_absolute():
        raise ValueError("Manifest paths must stay inside the dataset root")
    if not path.is_file():
        raise ValueError(f"Missing data file: {value}")
    return path


def audit_manifest(root: str | Path) -> dict:
    """Reject duplicate IDs, image content and component groups across splits.

    A group must identify the physical component/batch, not an arbitrary image.
    That semantic requirement cannot be inferred automatically from a CSV.
    """
    root = Path(root)
    with (root / "manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        expected = {"sample_id", "image", "mask", "split", "group", "label"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"CSV columns must be {sorted(expected)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    ids, groups, content = set(), {}, {}
    for row in rows:
        if row["split"] not in SPLITS or row["label"] not in {"0", "1"}:
            raise ValueError("Invalid split or label")
        if not row["sample_id"] or row["sample_id"] in ids or not row["group"]:
            raise ValueError("Require unique sample IDs and a non-empty group")
        ids.add(row["sample_id"])
        group = row["group"]
        if groups.setdefault(group, row["split"]) != row["split"]:
            raise ValueError(f"Component group crosses splits: {group}")
        image_path = _relative_file(root, row["image"])
        with Image.open(image_path) as handle:
            image = np.asarray(handle.convert("L"))
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if content.setdefault(digest, row["split"]) != row["split"]:
            raise ValueError("Identical decoded image content crosses splits")
        row["decoded_sha256"] = digest
        if row["mask"]:
            with Image.open(_relative_file(root, row["mask"])) as handle:
                target = np.asarray(handle.convert("L"))
            if target.shape != image.shape or not set(np.unique(target)).issubset({0, 1, 255}):
                raise ValueError("Mask must match image shape and contain binary values")
            if bool(target.any()) != (row["label"] == "1"):
                raise ValueError("Image label disagrees with mask")
        elif row["label"] != "0":
            raise ValueError("A defective image needs a localization mask")
    return {"rows": rows, "counts": dict(Counter(r["split"] for r in rows)),
            "manifest_sha256": hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest()}


class ManifestSurface(Dataset):
    def __init__(self, root, split, size=64, audit=None):
        self.root, self.size = Path(root), size
        self.rows = [r for r in (audit or audit_manifest(root))["rows"] if r["split"] == split]
        if not self.rows:
            raise ValueError(f"Empty split: {split}")

    def __len__(self):
        return len(self.rows)

    def kind(self, i):
        return "good" if self.rows[i]["label"] == "0" else "defect"

    def sample_id(self, i):
        return self.rows[i]["sample_id"]

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self.root / row["image"]) as handle:
            image = handle.convert("L").resize((self.size, self.size), Image.Resampling.BILINEAR)
            x = np.asarray(image, dtype=np.float32) / 255
        if row["mask"]:
            with Image.open(self.root / row["mask"]) as handle:
                target = handle.convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
                y = (np.asarray(target) > 0).astype(np.float32)
        else:
            y = np.zeros_like(x)
        if row["label"] == "1" and not y.any():
            raise ValueError("Defect vanished during resizing; increase image size")
        return torch.from_numpy(x[None].copy()), torch.from_numpy(y[None].copy())


def stack_dataset(dataset):
    pairs = [dataset[i] for i in range(len(dataset))]
    return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])
