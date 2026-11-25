"""Regenerate the experiment table and figure from recorded metrics (no training)."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mean_sd(values):
    return f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}" if len(values) > 1 else f"{values[0]:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/demo")
    args = parser.parse_args()
    folder = Path(args.results)
    data = json.loads((folder / "metrics.json").read_text())
    runs = data["runs"]
    headers, rows, labels, means, sds = [], [], [], [], []
    if "summary" in data:
        headers = ["Variant", "Parameters", "Test Dice (mean ± SD)", "Test IoU", "Precision", "Recall", "CPU median ms"]
        for name in data["config"]["variants"]:
            rr = [r for r in runs if r["variant"] == name]
            values = [r["test"]["dice"] for r in rr]
            rows.append([name, rr[0]["parameters"], mean_sd(values), f"{np.mean([r['test']['iou'] for r in rr]):.4f}",
                         f"{np.mean([r['test']['precision'] for r in rr]):.4f}", f"{np.mean([r['test']['recall'] for r in rr]):.4f}",
                         f"{statistics.median(r['latency_cpu']['median_ms'] for r in rr):.3f}"])
            labels.append(name)
            means.append(np.mean(values))
            sds.append(np.std(values, ddof=1) if len(values) > 1 else 0)
        ylabel = "Held-out pixel Dice"
        notes = "The four cells form an attention × context-fusion ablation. Extra modules change parameter count and random-number consumption. Three seeds describe sensitivity; they do not establish statistical superiority. All seeds use the same held-out procedural test set. Selection used validation Dice only."
        paired = []
        for name in data["config"]["variants"]:
            if name == "cnn":
                continue
            delta = [r["test"]["dice"] - next(b["test"]["dice"] for b in runs if b["seed"] == r["seed"] and b["variant"] == "cnn") for r in runs if r["variant"] == name]
            paired.append(f"- {name}: paired Dice change versus CNN = {mean_sd(delta)}.")
        notes += "\n\n" + "\n".join(paired)
    elif "training_curves" in data:
        headers = ["Variant", "Bits", "Accuracy (mean ± SD)", "Recall", "Max fake/integer difference", "Packed NPZ bytes"]
        for name, bits in [("fp32", None), ("fp32_matched_finetune", None)] + [(method, bits) for bits in data["config"]["bits"] for method in ("ptq", "qat")]:
            rr = [r for r in runs if r["variant"] == name and r.get("bits") == bits]
            values = [r["metrics"]["accuracy"] for r in rr]
            rows.append([name, bits or "—", mean_sd(values), f"{np.mean([r['metrics']['recall'] for r in rr]):.4f}",
                         f"{max(r.get('integer_fake_max_abs_logit_error', 0) for r in rr):.3g}" if bits else "—",
                         round(np.mean([r["storage"]["serialized_npz_bytes"] for r in rr])) if bits else "—"])
            labels.append(name.replace("fp32_matched_finetune", "FP32 control") + (f" {bits}" if bits else ""))
            means.append(np.mean(values))
            sds.append(np.std(values, ddof=1) if len(values) > 1 else 0)
        ylabel = "Held-out image accuracy"
        notes = "PTQ and QAT start from the same FP32 checkpoint. QAT and the FP32 control receive the same additional updates and sample order. Activation observers are frozen before QAT. Calibration is independent of test data. INT4/6/8 refer to simulated numeric formats and actual packed weight storage. Timings include float preprocessing, unpacked int64 reference accumulation and float output rescaling; they are not native INT8 deployment measurements. NPZ sizes include compression and container overhead."
    else:
        headers = ["Method", "Normal fit images", "Memory budget", "Image AUROC (mean ± SD)", "Pixel AP", "Defect recall", "Normal false alarm rate"]
        for shots in data["config"]["shots"]:
            keys = [("autoencoder", None)] + [(name, budget) for budget in data["config"]["budgets"] for name in ("handcrafted_memory", "encoder_memory")]
            for name, budget in keys:
                rr = [r for r in runs if r["method"] == name and r["shots"] == shots and r["budget"] == budget]
                values = [r["metrics"]["image"]["auroc"] for r in rr]
                rows.append([name, shots, budget or "—", mean_sd(values), f"{np.mean([r['metrics']['pixel']['average_precision'] for r in rr]):.4f}",
                             f"{np.mean([r['metrics']['defect_recall'] for r in rr]):.4f}", f"{np.mean([r['metrics']['normal_false_alarm_rate'] for r in rr]):.4f}"])
                labels.append(name.replace("handcrafted_memory", "fixed memory").replace("encoder_memory", "encoder memory") + f" {shots}/{budget or 0}")
                means.append(np.mean(values))
                sds.append(np.std(values, ddof=1) if len(values) > 1 else 0)
        ylabel = "Held-out image AUROC"
        notes = "The shot budget counts fit images only. The 32 independent normal calibration images are additional. Thresholds use calibration scores only, with a strict greater-than alarm rule. Bootstrap intervals resample images, not pixels; per-seed intervals and Wilson intervals are in metrics.json. Encoder memory uses a normal-only trained small autoencoder, not an ImageNet network. This study does not reproduce PatchCore and uses no external benchmark."
    text = "# Recorded experiments\n\n"
    text += f"Data: **{data['data']}**. Seeds: `{data['config']['seeds']}`. Measurement recorded: `{data['environment']['recorded_utc']}`.\n\n"
    text += "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n"
    text += "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows) + "\n\n"
    text += "![Aggregate results](summary.png)\n\n" + notes + "\n\n"
    text += "## Reproduction record\n\nThe complete configuration, data audit, runtime versions, source SHA-256, raw per-run metrics and available per-image CSVs are retained beside this report. CPU batch size, warm-up and repeat counts are stored with each timing. A source digest identifies the package Python files used for the recorded run. Model checkpoints contain tensors and inference metadata; optimizer state is intentionally omitted.\n\n"
    text += "## Practical limits\n\nProcedural defects have a narrow appearance distribution and exact generated masks. High scores on this generator do not establish performance on real materials, cameras, lighting, unseen defect types or production batches. Test images were used for the reported comparisons; choose any future production model using a separate validation protocol and reserve another untouched real test set.\n"
    (folder / "EXPERIMENTS.md").write_text(text, encoding="utf-8")
    with (folder / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * .8), 4.4))
    ax.bar(np.arange(len(labels)), means, yerr=sds, capsize=4, color="#2563A6", alpha=.9)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    ax.set_title(f"Synthetic CPU study · mean and sample SD across {len(data['config']['seeds'])} seeds")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / "summary.png", dpi=160)
    plt.close(fig)
    print(folder / "EXPERIMENTS.md")


if __name__ == "__main__":
    main()
