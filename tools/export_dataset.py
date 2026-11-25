"""Write a small procedural CSV dataset to exercise the local-data workflow."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from visionseg.data import SyntheticSurface


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/procedural_example")
    p.add_argument("--normal-only-fit", action="store_true", help="For anomaly fitting/calibration")
    args = p.parse_args()
    root = Path(args.out)
    if root.exists():
        raise ValueError("Choose a new output directory; existing datasets are preserved")
    root.mkdir(parents=True)
    rows = []
    for split, n, seed in [("train", 40, 61), ("val", 16, 62), ("cal", 32, 63), ("test", 24, 64)]:
        normal = args.normal_only_fit and split in {"train", "cal"}
        ds = SyntheticSurface(n, seed, normal_only=normal)
        (root / split).mkdir()
        for i in range(n):
            x, y = ds[i]
            name = f"{split}/{i:04d}"
            Image.fromarray(np.rint(x[0].numpy() * 255).astype("uint8")).save(root / (name + ".png"))
            Image.fromarray((y[0].numpy() * 255).astype("uint8")).save(root / (name + "_mask.png"))
            rows.append({"sample_id": ds.sample_id(i), "image": name + ".png", "mask": name + "_mask.png",
                         "split": split, "group": ds.sample_id(i), "label": int(y.any())})
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "image", "mask", "split", "group", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(root / "manifest.csv")


if __name__ == "__main__":
    main()
