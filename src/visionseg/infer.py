"""Apply a saved CPU checkpoint to one image and save probability/mask PNGs."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .model import DefectNet
from .repro import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="runs/inference")
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DefectNet(checkpoint["variant"]).eval()
    model.load_state_dict(checkpoint["weights"])
    image = Image.open(args.image).convert("L")
    resized = np.asarray(image.resize((checkpoint["size"], checkpoint["size"]), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    with torch.inference_mode():
        probability = model(torch.from_numpy(resized[None, None])).sigmoid()[0, 0].numpy()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(probability * 255).astype("uint8")).save(out / "probability.png")
    Image.fromarray(((probability >= checkpoint["threshold"]) * 255).astype("uint8")).save(out / "mask.png")
    write_json(out / "prediction.json", {"variant": checkpoint["variant"], "threshold": checkpoint["threshold"], "input_size": list(image.size), "prediction_size": list(probability.shape), "positive_pixels": int((probability >= checkpoint["threshold"]).sum())})


if __name__ == "__main__":
    main()
