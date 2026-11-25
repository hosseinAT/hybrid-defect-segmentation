# Data contract

The default experiments generate procedural grayscale images. They download no dataset and use no customer images. Real-image results require a separate local dataset and a new recorded run.

## Local CSV format

Place `manifest.csv` at the dataset root. It has exactly six columns:

```csv
sample_id,image,mask,split,group,label
part-0001,images/part-0001.png,masks/part-0001.png,train,batch-A,1
part-0002,images/part-0002.png,,cal,batch-B,0
part-0003,images/part-0003.png,masks/part-0003.png,test,batch-C,1
```

This is a format example, not a complete dataset. Supply additional samples and all splits needed by the selected experiment. Paths are relative to the CSV root. Absolute paths and paths escaping the root are rejected. Supported images are those readable by Pillow; they are converted to a single grayscale channel in `[0, 1]`.

| Field | Meaning | Constraint |
|---|---|---|
| `sample_id` | Stable image identifier | Unique and non-empty |
| `image` | Relative image path | File must exist inside root |
| `mask` | Relative binary localization mask | Optional only for normal images |
| `split` | Dataset partition | `train`, `val`, `cal`, or `test` |
| `group` | Physical component or production batch | Every group belongs to exactly one split |
| `label` | Image class | `0` normal, `1` defective |

Use group IDs that express the real source of correlation. Images of the same part taken from different angles belong to one group. If near-duplicate parts from one production batch dominate a dataset, use the batch as the group. Arbitrary per-image groups cannot prevent semantic leakage.

Masks must match the original image dimensions and use only 0, 1 or 255. Nonzero pixels denote defects. Defective labels require a non-empty mask; normal labels require an empty mask. Images resize with bilinear interpolation and masks with nearest-neighbor interpolation. If a defect disappears while resizing, loading fails so the lost target cannot silently become a normal example. Use a larger configured size or build a tile-level protocol for small defects.

## Audits

The audit rejects duplicate sample IDs, groups crossing splits and identical decoded grayscale image content across splits. Content hashes include decoded shape and pixels, so a change in image container metadata does not bypass the exact-duplicate check. It does not detect all near-duplicates, bursts, mislabeled defects or incorrectly chosen group IDs. Review those relationships before fitting.

Segmentation uses `train`, `val` and `test`. Quantization uses `train`, `cal` and `test`; labels of calibration images are ignored. Anomaly detection uses `train`, `cal` and `test`, with normal-only `train` and `cal`. The anomaly runner rejects positive masks in either fit or calibration data.

## Transfer to real images

1. Keep a copy of original images, annotation instructions and dataset permission/source information outside the repository.
2. Split by component or batch before cropping, augmentation or extraction of multiple views. Derived samples inherit their source group.
3. Review mask quality and the smallest relevant physical defect against image resolution.
4. Audit the CSV and inspect examples from every split. The audit validates structure, not industrial meaning.
5. Set model configuration on training/validation material. Keep calibration independent of test, and preserve a final untouched real test set.
6. Report camera, lens, illumination, bit depth, material, split composition and defect classes with the new results. Record false alarms per inspected part, recall per defect class, and failure examples at the selected threshold.

The synthetic demo contains scratches, pits and rectangular contrast changes over striped textures. It is deliberately narrow and only exercises the software and evaluation procedure.
