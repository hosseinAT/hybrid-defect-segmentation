"""Four-cell architecture ablation with validation-only checkpoint selection."""
from __future__ import annotations
import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SyntheticSurface, ManifestSurface, audit_manifest, stack_dataset
from .metrics import confusion
from .model import DefectNet, VARIANTS, defect_loss
from .repro import environment, seed_all, write_json


@dataclass
class Config:
    seeds: list[int] = field(default_factory=lambda: [7, 17, 27])
    variants: list[str] = field(default_factory=lambda: list(VARIANTS))
    epochs: int = 6
    train: int = 128
    val: int = 48
    test: int = 80
    size: int = 64
    batch_size: int = 8
    lr: float = .001
    thresholds: list[float] = field(default_factory=lambda: [.2, .3, .4, .5, .6, .7])


@torch.inference_mode()
def predict(model, x, batch_size=8):
    model.eval()
    return torch.cat([model(batch).sigmoid() for batch in x.split(batch_size)]).numpy()


def selection(probabilities, targets, thresholds):
    candidates = [(confusion(probabilities >= t, targets)["dice"], t) for t in thresholds]
    # Stable tie: prefer the lower configured threshold, preserving recall.
    return max(candidates, key=lambda item: item[0])


@torch.inference_mode()
def latency(model, size, repeats=30):
    model.eval()
    x = torch.zeros(1, 1, size, size)
    for _ in range(5):
        model(x)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(x)
        timings.append((time.perf_counter() - start) * 1000)
    return {"batch_size": 1, "warmup": 5, "repeats": repeats,
            "median_ms": statistics.median(timings), "p95_ms": float(np.quantile(timings, .95))}


def failure_figure(x, y, probs, threshold, dataset, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scores = [confusion(probs[i] >= threshold, y[i])["dice"] for i in range(len(y))]
    idx = np.argsort(scores)[:4]
    fig, axes = plt.subplots(len(idx), 4, figsize=(10, 2.3 * len(idx)), squeeze=False)
    for row, i in enumerate(idx):
        for col, image in enumerate([x[i, 0], y[i, 0], probs[i, 0], probs[i, 0] >= threshold]):
            axes[row, col].imshow(image, cmap="magma" if col == 2 else "gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[row, 0].set_title(f"{dataset.sample_id(int(i))}\nDice {scores[i]:.3f}", fontsize=8)
    for ax, title in zip(axes[0], ["Input / failure cases", "Ground truth", "Probability", "Prediction"]):
        ax.set_title(title + ("\n" + ax.get_title() if ax.get_title() else ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(config, out, root=None):
    c, out = config, Path(out)
    if c.epochs < 1 or not c.seeds or not c.variants or c.size < 32 or c.batch_size < 1 or not c.thresholds or any(not 0 < t < 1 for t in c.thresholds):
        raise ValueError("Invalid experiment configuration")
    out.mkdir(parents=True, exist_ok=True)
    audit = audit_manifest(root) if root else None
    datasets = {split: ManifestSurface(root, split, c.size, audit) if root else SyntheticSurface(n, seed, c.size)
                for split, n, seed in [("train", c.train, 11), ("val", c.val, 22), ("test", c.test, 33)]}
    vx, vy = stack_dataset(datasets["val"])
    tx, ty = stack_dataset(datasets["test"])
    write_json(out / "config.json", asdict(c))
    write_json(out / "data_audit.json", audit or {"source": "procedural surfaces", "split_seeds": {"train": 11, "val": 22, "test": 33}, "counts": {k: len(v) for k, v in datasets.items()}})
    rows, curves = [], []
    for seed in c.seeds:
        for variant in c.variants:
            seed_all(seed)
            model = DefectNet(variant)
            optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr)
            loader = DataLoader(datasets["train"], batch_size=c.batch_size, shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            best, best_state, threshold, best_epoch = -1., None, .5, 0
            for epoch in range(1, c.epochs + 1):
                model.train()
                total = 0.
                for x, y in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = defect_loss(model(x), y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                    total += loss.item() * len(x)
                score, candidate = selection(predict(model, vx), vy.numpy(), c.thresholds)
                curves.append({"seed": seed, "variant": variant, "epoch": epoch, "loss": total / len(datasets["train"]), "val_dice": score})
                if score > best:
                    best, threshold, best_epoch = score, candidate, epoch
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            probs = predict(model, tx)
            target = ty.numpy()
            counts = confusion(probs >= threshold, target)
            normal = ~target.astype(bool).reshape(len(target), -1).any(1)
            false_alarm = (probs >= threshold).reshape(len(target), -1).any(1)
            per_kind = {kind: confusion(probs[[datasets["test"].kind(i) == kind for i in range(len(tx))]] >= threshold,
                                       target[[datasets["test"].kind(i) == kind for i in range(len(tx))]])
                        for kind in sorted({datasets["test"].kind(i) for i in range(len(tx))})}
            row = {"seed": seed, "variant": variant, "parameters": sum(p.numel() for p in model.parameters()),
                   "best_epoch": best_epoch, "threshold": threshold, "val_dice": best, "test": counts,
                   "normal_image_false_alarm_rate": float(false_alarm[normal].mean()) if normal.any() else None,
                   "per_kind": per_kind, "latency_cpu": latency(model, c.size)}
            rows.append(row)
            checkpoint = out / f"{variant}_seed{seed}.pt"
            torch.save({"weights": best_state, "variant": variant, "threshold": threshold, "size": c.size}, checkpoint)
            with (out / f"predictions_{variant}_seed{seed}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "kind", "tp", "fp", "fn", "tn", "precision", "recall", "dice", "iou", "accuracy"])
                writer.writeheader()
                for i in range(len(tx)):
                    writer.writerow({"sample_id": datasets["test"].sample_id(i), "kind": datasets["test"].kind(i), **confusion(probs[i] >= threshold, target[i])})
            if seed == c.seeds[0]:
                failure_figure(tx.numpy(), target, probs, threshold, datasets["test"], out / f"failures_{variant}.png")
            print(f"seed={seed} variant={variant} test_dice={counts['dice']:.4f}", flush=True)
    summary = {}
    for variant in c.variants:
        values = [row["test"]["dice"] for row in rows if row["variant"] == variant]
        summary[variant] = {"dice_mean": float(np.mean(values)), "dice_sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None, "n_seeds": len(values)}
    result = {"data": "private manifest" if root else "procedural synthetic only", "config": asdict(c), "runs": rows, "summary": summary, "environment": environment(Path(__file__).parent)}
    write_json(out / "metrics.json", result)
    write_json(out / "learning_curves.json", curves)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/demo.json")
    p.add_argument("--out", default="runs/demo")
    p.add_argument("--root", help="Dataset root containing audited manifest.csv")
    args = p.parse_args()
    run(Config(**json.loads(Path(args.config).read_text())), args.out, args.root)


if __name__ == "__main__":
    main()
