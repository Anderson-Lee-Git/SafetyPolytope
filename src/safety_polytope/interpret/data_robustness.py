import os
from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_run_summaries(base_dir):
    return list(Path(base_dir).rglob("**/run_summary.json"))


def extract_metrics(summary_path, metrics_mode="standard"):
    with open(summary_path, "r") as f:
        d = json.load(f)
    res = {}
    subset_size = None
    # old: at root; new: sometimes nested under "config"
    config = d.get("config", d)
    subset_size = config.get("subset_size", None)
    if subset_size is not None:
        subset_size = float(subset_size)
    # Collect metrics per main.py convention
    res["subset_size"] = subset_size
    if metrics_mode == "test_model":
        tmr = d.get("test_model_result", {}) or {}
        res["acc"] = tmr.get("test_acc", None)
        res["fpr"] = tmr.get("test_fpr", None)
        res["fnr"] = tmr.get("test_fnr", None)
        res["f1"] = tmr.get("test_f1", None)
    else:
        res["acc"] = d.get("test_acc", None)
        res["fpr"] = d.get("test_fpr", None)
        res["fnr"] = d.get("test_fnr", None)
        res["f1"] = d.get("test_f1", None)
    res["category_distribution"] = None
    res["label_distribution"] = None
    dataset_summary = d.get("dataset_summary", None)
    if dataset_summary:
        cat_dist = dataset_summary.get("category_distribution", None)
        if cat_dist:
            res["category_distribution"] = cat_dist
        label_dist = dataset_summary.get("label_distribution", None)
        if label_dist:
            res["label_distribution"] = label_dist
    return res


def merge_nested_dicts(list_of_dicts):
    # Merge a list of dicts with same keys, summing values for each key
    merged = {}
    for d in list_of_dicts:
        for k, v in d.items():
            merged.setdefault(k, [])
            merged[k].append(v)
    # now, for each key, make sure all lists are the same length, pad with zeros if needed
    all_keys = list(merged)
    maxlen = max(len(v) for v in merged.values())
    for k in all_keys:
        n = len(merged[k])
        if n < maxlen:
            merged[k] += [0] * (maxlen - n)
    return merged


def plot_metrics(metrics, outdir, metrics_mode="standard"):
    subset_sizes = sorted(
        set(m["subset_size"] for m in metrics if m["subset_size"] is not None)
    )
    # Group metrics by subset size
    metrics_by_subset = {sz: [] for sz in subset_sizes}
    for m in metrics:
        if m["subset_size"] is not None:
            metrics_by_subset[m["subset_size"]].append(m)
    avgs = {k: [] for k in ["acc", "fpr", "fnr", "f1"]}
    stds = {k: [] for k in ["acc", "fpr", "fnr", "f1"]}
    for sz in subset_sizes:
        ms = metrics_by_subset[sz]
        for k in avgs:
            vals = [m[k] for m in ms if m[k] is not None]
            avgs[k].append(np.nanmean(vals) if vals else np.nan)
            stds[k].append(np.nanstd(vals, ddof=1) if vals else np.nan)
    plt.figure(figsize=(8, 5), dpi=150)
    for k, label, marker in zip(
        ["acc", "fpr", "fnr", "f1"],
        ["Accuracy", "FPR", "FNR", "F1"],
        ["o", "s", "^", "d"],
    ):
        plt.errorbar(
            subset_sizes, avgs[k], yerr=stds[k], label=label, marker=marker, capsize=3
        )
    plt.xlabel("Subset size")
    plt.ylabel("Metric value")
    mode_label = "Standard" if metrics_mode == "standard" else "Test-Model"
    plt.title(f"Average evaluation metrics vs subset size ({mode_label})")
    plt.xlim(min(subset_sizes), max(subset_sizes))
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    suffix = "standard" if metrics_mode == "standard" else "test_model"
    outpath = os.path.join(outdir, f"metrics_vs_subset_{suffix}.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved metrics plot to {outpath}")


def plot_category_label_distribution(metrics, outdir):
    # We want to show category and label counts for each subset size, averaged and error-barred,
    # Group per subset size, then over random seeds, average.
    subset_sizes = sorted(
        set(m["subset_size"] for m in metrics if m["subset_size"] is not None)
    )
    metrics_by_subset = {sz: [] for sz in subset_sizes}
    for m in metrics:
        if m["subset_size"] is not None:
            metrics_by_subset[m["subset_size"]].append(m)

    # Get all possible categories and labels
    all_categories = set()
    all_labels = set()
    for m in metrics:
        cat_dist = m.get("category_distribution", None)
        if cat_dist:
            all_categories.update(cat_dist.keys())
        label_dist = m.get("label_distribution", None)
        if label_dist:
            all_labels.update(label_dist.keys())
    all_categories = sorted(list(all_categories))
    all_labels = sorted(list(all_labels))

    # Category distribution
    cat_mean = {sz: [] for sz in subset_sizes}
    cat_std = {sz: [] for sz in subset_sizes}
    for sz in subset_sizes:
        per_seed_counts = []
        for m in metrics_by_subset[sz]:
            cat_dist = m.get("category_distribution", {})
            # Ensure all categories present
            counts = [cat_dist.get(cat, 0) for cat in all_categories]
            per_seed_counts.append(counts)
        arr = np.array(per_seed_counts, dtype=float)
        cat_mean[sz] = (
            arr.mean(axis=0) if len(arr) > 0 else np.zeros(len(all_categories))
        )
        cat_std[sz] = (
            arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros(len(all_categories))
        )
    # Plot as grouped barplot with error bars, one panel per subset size (or one panel w/ grouping)
    fig, ax = plt.subplots(figsize=(max(10, len(all_categories) * 0.7), 6))
    bar_width = 0.8 / len(subset_sizes)
    indices = np.arange(len(all_categories))
    colors = plt.cm.viridis(np.linspace(0, 1, len(subset_sizes)))
    for i, sz in enumerate(subset_sizes):
        offset = (i - len(subset_sizes) / 2) * bar_width + bar_width / 2
        ax.bar(
            indices + offset,
            cat_mean[sz],
            bar_width,
            yerr=cat_std[sz],
            label=f"{sz:.1f}",
            capsize=3,
            color=colors[i],
        )
    ax.set_xticks(indices)
    ax.set_xticklabels(all_categories, rotation=45, ha="right")
    ax.set_ylabel("Avg count per subset (± std)")
    ax.set_title("Category distribution per subset size")
    ax.legend(title="Subset size")
    plt.tight_layout()
    outpath = os.path.join(outdir, "category_distribution_vs_subset.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved category distribution plot to {outpath}")

    # Label distribution
    label_mean = {sz: [] for sz in subset_sizes}
    label_std = {sz: [] for sz in subset_sizes}
    for sz in subset_sizes:
        per_seed_counts = []
        for m in metrics_by_subset[sz]:
            label_dist = m.get("label_distribution", {})
            counts = [label_dist.get(label, 0) for label in all_labels]
            per_seed_counts.append(counts)
        arr = np.array(per_seed_counts, dtype=float)
        label_mean[sz] = arr.mean(axis=0) if len(arr) > 0 else np.zeros(len(all_labels))
        label_std[sz] = (
            arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros(len(all_labels))
        )
    fig, ax = plt.subplots(figsize=(max(7, len(all_labels) * 1.5), 4))
    bar_width = 0.8 / len(subset_sizes)
    indices = np.arange(len(all_labels))
    for i, sz in enumerate(subset_sizes):
        offset = (i - len(subset_sizes) / 2) * bar_width + bar_width / 2
        ax.bar(
            indices + offset,
            label_mean[sz],
            bar_width,
            yerr=label_std[sz],
            label=f"{sz:.1f}",
            capsize=3,
            color=colors[i],
        )
    ax.set_xticks(indices)
    ax.set_xticklabels(all_labels, rotation=0, ha="center")
    ax.set_ylabel("Avg label count (± std)")
    ax.set_title("Label distribution per subset size")
    ax.legend(title="Subset size")
    plt.tight_layout()
    outpath = os.path.join(outdir, "label_distribution_vs_subset.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved label distribution plot to {outpath}")


def main(base_dir, outdir="robustness_plots", metrics_mode="standard"):
    print(f"Finding metrics in {base_dir}...")
    run_summaries = []
    for bd in base_dir:
        run_summaries.extend(find_run_summaries(bd))
    print(f"Found {len(run_summaries)} run summary files.")
    metrics = [extract_metrics(path, metrics_mode) for path in run_summaries]
    plot_metrics(metrics, outdir, metrics_mode)
    plot_category_label_distribution(metrics, outdir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize safety-polytope experiment robustness over seeds."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        nargs="*",
        help="Base directory to search for run_summary.json files",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="robustness_plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--metrics-mode",
        type=str,
        choices=["standard", "test_model"],
        default="standard",
        help="Which metrics to plot: top-level standard test metrics or nested test_model_result",
    )
    args = parser.parse_args()
    main(args.base_dir, args.outdir, args.metrics_mode)
