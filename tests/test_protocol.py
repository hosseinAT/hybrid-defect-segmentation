import csv
import numpy as np
import pytest
import torch
from PIL import Image
from visionseg.data import SyntheticSurface, audit_manifest, ManifestSurface
from visionseg.metrics import confusion, ranking, wilson_interval


def make_manifest(root, rows):
    with (root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "image", "mask", "split", "group", "label"])
        writer.writeheader()
        writer.writerows(rows)


def test_procedural_samples_are_index_stable_and_splits_differ():
    a, b = SyntheticSurface(4, 11), SyntheticSurface(4, 22)
    assert torch.equal(a[1][0], a[1][0])
    assert not torch.equal(a[1][0], b[1][0])
    assert a[0][1].sum() == 0 and a[1][1].sum() > 0


def test_manifest_rejects_component_and_content_leakage(tmp_path):
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(tmp_path / "a.png")
    Image.fromarray(np.ones((32, 32), dtype=np.uint8)).save(tmp_path / "b.png")
    rows = [dict(sample_id="a", image="a.png", mask="", split="train", group="batch1", label="0"),
            dict(sample_id="b", image="b.png", mask="", split="test", group="batch1", label="0")]
    make_manifest(tmp_path, rows)
    with pytest.raises(ValueError, match="group crosses"):
        audit_manifest(tmp_path)
    rows[1]["group"] = "batch2"
    rows[1]["image"] = "a.png"
    make_manifest(tmp_path, rows)
    with pytest.raises(ValueError, match="content crosses"):
        audit_manifest(tmp_path)


def test_manifest_accepts_disjoint_groups_and_constructs_zero_mask(tmp_path):
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(tmp_path / "a.png")
    make_manifest(tmp_path, [dict(sample_id="a", image="a.png", mask="", split="train", group="batch1", label="0")])
    audit = audit_manifest(tmp_path)
    x, y = ManifestSurface(tmp_path, "train", audit=audit)[0]
    assert x.shape == (1, 64, 64) and not y.any()


def test_empty_masks_and_single_class_metrics_have_defined_behavior():
    metrics = confusion([0, 0], [0, 0])
    assert metrics["dice"] == 1 and metrics["tn"] == 2
    assert ranking([0, 0], [.1, .2])["auroc"] is None
    lo, hi = wilson_interval(0, 20)
    assert lo == 0 and 0 < hi < .2
