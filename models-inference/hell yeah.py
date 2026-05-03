import json
import os
from pathlib import Path
import unicodedata
import re
import statistics

try:
    from rapidfuzz.distance import Levenshtein


    def edit_distance(a, b):
        return Levenshtein.distance(a, b)
except ImportError:
    def edit_distance(a, b):
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path("/home/artemiy/Documents/test_rec/calcs//ocr_quality")
REFERENCE_MODEL = "qwen3-vl-8b"
REFERENCE_JSON = "recognition_results_lmstudio.json"
OUT_DIR = BASE_DIR / "charts"
OUT_DIR.mkdir(exist_ok=True)

MODELS = {
    "gemma-3-4b": "recognition_results_lmstudio.json",
    "gemma-4-e2b": "recognition_results_lmstudio.json",
    "nanonets-ocr2-3b@q4_k_m": "recognition_results_lmstudio.json",
    "nanonets-ocr2-3b@q8_0": "recognition_results_lmstudio.json",
    "paddle": "recognition_results.json",
    "qwen3-vl-4b": "recognition_results_lmstudio.json",
}

SUBFOLDERS = ["06_12", "07_12", "09_11", "13_12", "15_11", "16_11", "29_11", "30_11"]

FOLDER_LABELS = {
    "06_12": "06 Dec\n(lens, ok)",
    "07_12": "07 Dec\n(lens, worse)",
    "09_11": "09 Nov\n(bad quality)",
    "13_12": "13 Dec\n(snow)",
    "15_11": "15 Nov\n(light, mixed)",
    "16_11": "16 Nov\n(light, mixed)",
    "29_11": "29 Nov\n(lens, meh)",
    "30_11": "30 Nov\n(full day, lens)",
}

PALETTE = [
    "#2E86AB",
    "#A23B72",
    "#F18F01",
    "#C73E1D",
    "#6A994E",
    "#3A7CA5",
    "#BC4B51",
    "#D4A76A",
]

BG = "#F5F5F5"
BG_PANEL = "#FFFFFF"
GRID_C = "#E0E0E0"
TEXT_C = "#333333"


def apply_light_style():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG_PANEL,
        "axes.edgecolor": GRID_C,
        "axes.labelcolor": TEXT_C,
        "axes.titlecolor": TEXT_C,
        "xtick.color": TEXT_C,
        "ytick.color": TEXT_C,
        "grid.color": GRID_C,
        "grid.linewidth": 0.5,
        "text.color": TEXT_C,
        "legend.facecolor": BG_PANEL,
        "legend.edgecolor": GRID_C,
        "font.family": "sans-serif",
        "font.size": 11,
    })


def is_valid_russian_plate(text: str) -> bool:
    if not text:
        return False
    pattern = r'^[АВЕКМНОРСТУХABEKMHOPCTYX]\d{3}[АВЕКМНОРСТУХABEKMHOPCTYX]{2}\d{2,3}$'
    return bool(re.match(pattern, text, re.IGNORECASE))


def is_too_long(text: str) -> bool:
    return len(text) > 12


def normalize(text: str) -> str:
    if not text or text.strip().upper() in ("N/A", "NONE", ""):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", "").replace("-", "").replace("_", "")
    return text.upper()


def cer(ref: str, hyp: str) -> float:
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else float("inf")
    return edit_distance(ref, hyp) / len(ref)


def exact_match(ref: str, hyp: str) -> bool:
    return ref == hyp


def load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stats(records):
    if not records:
        return {"n": 0, "exact_match_pct": 0.0, "mean_cer": 0.0,
                "n_empty_hyp": 0, "n_empty_ref": 0, "n_inf_cer": 0}
    total = len(records)
    exact = sum(1 for r in records if r["exact"])
    cers = [r["cer"] for r in records if r["cer"] != float("inf")]

    n_empty_hyp = sum(1 for r in records if r["hyp"] == "")

    return {
        "n": total,
        "exact_match_pct": round(exact / total * 100, 2) if total > 0 else 0.0,
        "mean_cer": round(sum(cers) / len(cers) if cers else 0.0, 4),
        "n_empty_hyp": n_empty_hyp,
        "n_empty_ref": sum(1 for r in records if r["ref"] == ""),
        "n_inf_cer": sum(1 for r in records if r["cer"] == float("inf")),
    }


def collect_speed_data(model_name, json_filename):
    all_times = []
    all_fps = []

    for folder in SUBFOLDERS:
        path = BASE_DIR / model_name / folder / json_filename
        records = load_json(path)
        for rec in records:
            ocr_time = rec.get("ocr_time_ms")
            ocr_fps = rec.get("ocr_fps")
            if ocr_time is not None and ocr_time > 0:
                all_times.append(ocr_time)
            if ocr_fps is not None and ocr_fps > 0:
                all_fps.append(ocr_fps)

    median_time = statistics.median(all_times) if all_times else 0
    median_fps = statistics.median(all_fps) if all_fps else 0

    return {"median_time_ms": median_time, "median_fps": median_fps}


def compare_model(model_name, json_filename, ref_global, allowed_filenames=None):
    all_records = []
    per_folder = {}

    total_original = 0
    total_filtered = 0

    for folder in SUBFOLDERS:
        path = BASE_DIR / model_name / folder / json_filename
        records = load_json(path)
        hyp_map = {r["filename"]: r["plate_text_raw"] for r in records}
        recs = []
        folder_original = 0
        folder_filtered = 0

        for filename, ref_raw in ref_global.items():
            if filename not in hyp_map:
                continue

            if allowed_filenames is not None and filename not in allowed_filenames:
                continue

            folder_original += 1
            ref_norm = normalize(ref_raw)

            if not is_valid_russian_plate(ref_norm):
                continue

            folder_filtered += 1
            hyp_raw = hyp_map[filename]
            hyp_norm = normalize(hyp_raw)

            if is_too_long(hyp_norm):
                hyp_norm = ""

            recs.append({
                "filename": filename,
                "ref": ref_norm,
                "hyp": hyp_norm,
                "cer": cer(ref_norm, hyp_norm),
                "exact": exact_match(ref_norm, hyp_norm),
            })

        total_original += folder_original
        total_filtered += folder_filtered

        folder_stats = stats(recs)
        folder_stats["original_count"] = folder_original
        folder_stats["filtered_count"] = folder_filtered
        folder_stats["filtered_out"] = folder_original - folder_filtered

        per_folder[folder] = folder_stats
        all_records.extend(recs)

    overall_stats = stats(all_records)
    overall_stats["original_count"] = total_original
    overall_stats["filtered_count"] = total_filtered
    overall_stats["filtered_out"] = total_original - total_filtered

    print(f"    {model_name}: отфильтровано {total_original - total_filtered} из {total_original} записей")

    return {"overall": overall_stats, "per_folder": per_folder,
            "all_records": all_records}

def build_reference_hit_set(ref_global, ref_model_name, ref_json_filename):
    """
    Возвращает множество filename-ов, где референсная модель 'попала':
    edit_distance(gt_norm, ref_hyp_norm) <= 2, и gt_norm — валидный номер.
    """
    hit_filenames = set()
    for folder in SUBFOLDERS:
        path = BASE_DIR / ref_model_name / folder / ref_json_filename
        records = load_json(path)
        for rec in records:
            filename = rec["filename"]
            if filename not in ref_global:
                continue
            gt_norm = normalize(ref_global[filename])
            if not is_valid_russian_plate(gt_norm):
                continue
            ref_hyp_norm = normalize(rec["plate_text_raw"])
            if is_too_long(ref_hyp_norm):
                ref_hyp_norm = ""
            dist = edit_distance(gt_norm, ref_hyp_norm)
            if dist <= 2:
                hit_filenames.add(filename)
    return hit_filenames


def short_name(m):
    return m.replace("nanonets-ocr2-3b@", "nanonets\n@")


def chart_overall_bars(results, sorted_models, speed_data):
    apply_light_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("OCR Models Overall Quality (reference: qwen3-vl-8b)",
                 fontsize=14, fontweight="bold", color=TEXT_C, y=1.02)

    labels = [short_name(m) for m in sorted_models]
    x = np.arange(len(sorted_models))
    colors = PALETTE[:len(sorted_models)]

    ax = axes[0]
    vals = [results[m]["overall"]["exact_match_pct"] for m in sorted_models]
    bars = ax.bar(x, vals, color=colors, width=0.7, zorder=3,
                  edgecolor="white", linewidth=1)
    ax.set_title("Exact Match % higher is better", fontsize=12, pad=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Exact Match %")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", zorder=0, alpha=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=10, color=TEXT_C, fontweight="bold")

    ax = axes[1]
    cer_vals = [results[m]["overall"]["mean_cer"] for m in sorted_models]
    order = np.argsort(cer_vals)[::-1]
    ax.bar(np.arange(len(sorted_models)),
           [cer_vals[i] for i in order],
           color=[colors[i] for i in order],
           width=0.7, zorder=3, edgecolor="white", linewidth=1)
    ax.set_title("Mean CER lower is better", fontsize=12, pad=10, fontweight="bold")
    ax.set_xticks(np.arange(len(sorted_models)))
    ax.set_xticklabels([short_name(sorted_models[i]) for i in order], fontsize=9)
    ax.set_ylabel("Character Error Rate")
    ax.grid(axis="y", zorder=0, alpha=0.5)
    for xi, i in enumerate(order):
        v = cer_vals[i]
        ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", va="bottom",
                fontsize=10, color=TEXT_C, fontweight="bold")

    fig.tight_layout()
    p = OUT_DIR / "01_overall_bars.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {p}")


def chart_heatmap(results, sorted_models, metric, title, fmt_fn, cmap):
    apply_light_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Per Day Heatmap {title}", fontsize=13,
                 fontweight="bold", color=TEXT_C, y=1.02)

    data = np.array([
        [results[m]["per_folder"][f][metric] for f in SUBFOLDERS]
        for m in sorted_models
    ])

    if cmap == "YlGn":
        cmap = "YlOrRd"

    im = ax.imshow(data, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(SUBFOLDERS)))
    ax.set_xticklabels([FOLDER_LABELS[f] for f in SUBFOLDERS], fontsize=9)
    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels([m.replace("nanonets-ocr2-3b@", "nanonets@")
                        for m in sorted_models], fontsize=10)

    for i in range(len(sorted_models)):
        for j in range(len(SUBFOLDERS)):
            rgba = im.cmap(im.norm(data[i, j]))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if brightness < 0.5 else "black"

            ax.text(j, i, fmt_fn(data[i, j]), ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_C)
    fig.tight_layout()
    p = OUT_DIR / f"02_heatmap_{metric}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {p}")


def chart_per_day_lines(results, sorted_models):
    apply_light_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Exact Match % by Day All Models", fontsize=13,
                 fontweight="bold", color=TEXT_C, y=1.02)

    x = np.arange(len(SUBFOLDERS))
    for i, m in enumerate(sorted_models):
        vals = [results[m]["per_folder"][f]["exact_match_pct"] for f in SUBFOLDERS]
        color = PALETTE[i % len(PALETTE)]
        label = m.replace("nanonets-ocr2-3b@", "nanonets@")
        ax.plot(x, vals, marker="o", linewidth=2.5, markersize=8,
                color=color, label=label, zorder=3)
        ax.annotate(f"{vals[-1]:.0f}%", xy=(x[-1], vals[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=9, color=color, va="center", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([FOLDER_LABELS[f] for f in SUBFOLDERS], fontsize=9)
    ax.set_ylabel("Exact Match %")
    ax.set_ylim(0, 110)
    ax.grid(axis="both", zorder=0, alpha=0.5)
    ax.legend(loc="lower left", fontsize=10, ncol=2, framealpha=0.9)
    fig.tight_layout()
    p = OUT_DIR / "03_per_day_lines.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {p}")


def chart_empty_hyp(results, sorted_models):
    apply_light_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Empty/Long Predictions per Model", fontsize=13,
                 fontweight="bold", color=TEXT_C, y=1.02)

    x = np.arange(len(sorted_models))
    n_total = [results[m]["overall"]["n"] for m in sorted_models]
    n_empty = [results[m]["overall"]["n_empty_hyp"] for m in sorted_models]
    n_ok = [t - e for t, e in zip(n_total, n_empty)]

    ax.bar(x, n_ok, color=PALETTE[3], label="Valid", width=0.7,
           zorder=3, edgecolor="white", linewidth=1)
    ax.bar(x, n_empty, bottom=n_ok, color=PALETTE[2], label="Empty / Long",
           width=0.7, zorder=3, edgecolor="white", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in sorted_models], fontsize=9)
    ax.set_ylabel("Count")
    ax.grid(axis="y", zorder=0, alpha=0.5)
    ax.legend(fontsize=10, framealpha=0.9)

    for xi, (e, t) in enumerate(zip(n_empty, n_total)):
        if e > 0:
            ax.text(xi, t + 2, f"{e / t * 100:.1f}%", ha="center",
                    fontsize=9, color=TEXT_C, fontweight="bold")

    fig.tight_layout()
    p = OUT_DIR / "04_empty_predictions.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {p}")


def chart_cer_boxplot(results, sorted_models):
    apply_light_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("CER Distribution per Model", fontsize=13,
                 fontweight="bold", color=TEXT_C, y=1.02)

    all_cers = []
    for m in sorted_models:
        cers_m = [r["cer"] for r in results[m]["all_records"]
                  if r["cer"] != float("inf") and r["ref"] != ""]
        all_cers.append(cers_m)

    bp = ax.boxplot(all_cers, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(color=TEXT_C, linewidth=1.2),
                    capprops=dict(color=TEXT_C, linewidth=1.2),
                    flierprops=dict(marker=".", color=GRID_C, alpha=0.5, markersize=3))

    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(sorted_models) + 1))
    ax.set_xticklabels([short_name(m) for m in sorted_models], fontsize=9)
    ax.set_ylabel("CER (lower is better)")
    ax.grid(axis="y", zorder=0, alpha=0.5)
    fig.tight_layout()
    p = OUT_DIR / "05_cer_boxplot.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {p}")


def chart_radar(results, sorted_models):
    apply_light_style()

    def scores(m):
        o = results[m]["overall"]
        return [
            o["exact_match_pct"] / 100,
            1 - min(o["mean_cer"], 1.0),
            1 - o["n_empty_hyp"] / o["n"] if o["n"] else 0,
        ]

    labels = ["ExactMatch%", "1 MeanCER", "Coverage\n(non-empty)"]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG_PANEL)
    fig.suptitle("Radar normalized metrics", fontsize=13,
                 fontweight="bold", color=TEXT_C, y=0.97)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12, color=TEXT_C, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=9, color=TEXT_C)
    ax.grid(color=GRID_C, linewidth=0.8, alpha=0.5)
    ax.spines["polar"].set_color(GRID_C)

    for i, m in enumerate(sorted_models):
        vals = scores(m) + scores(m)[:1]
        color = PALETTE[i % len(PALETTE)]
        label = m.replace("nanonets-ocr2-3b@", "nanonets@")
        ax.plot(angles, vals, color=color, linewidth=2.5, label=label)
        ax.fill(angles, vals, color=color, alpha=0.1)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15),
              fontsize=10, framealpha=0.9)
    p = OUT_DIR / "06_radar.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {p}")


def chart_speed_comparison(sorted_models, speed_data):
    apply_light_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Inference Speed Comparison (median, with reference)", fontsize=14,
                 fontweight="bold", color=TEXT_C, y=1.02)

    models_display = [short_name(m) for m in sorted_models]
    speeds = [speed_data[m]["median_time_ms"] for m in sorted_models]
    fps = [speed_data[m]["median_fps"] for m in sorted_models]

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(sorted_models))]
    x = np.arange(len(sorted_models))

    ax = axes[0]
    bars = ax.bar(x, speeds, color=colors, width=0.7, zorder=3,
                  edgecolor="white", linewidth=1)
    ax.set_title("Inference Time per Image lower is better",
                 fontsize=12, pad=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models_display, fontsize=9)
    ax.set_ylabel("Time (ms)")
    ax.grid(axis="y", zorder=0, alpha=0.5)

    for bar, v in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{v:.0f}ms", ha="center", va="bottom",
                fontsize=10, color=TEXT_C, fontweight="bold")

    ax = axes[1]
    bars = ax.bar(x, fps, color=colors, width=0.7, zorder=3,
                  edgecolor="white", linewidth=1)
    ax.set_title("Frames Per Second higher is better",
                 fontsize=12, pad=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models_display, fontsize=9)
    ax.set_ylabel("FPS")
    ax.grid(axis="y", zorder=0, alpha=0.5)

    for bar, v in zip(bars, fps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=10, color=TEXT_C, fontweight="bold")

    fig.tight_layout()
    p = OUT_DIR / "07_speed_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {p}")


def chart_quality_vs_speed(results, sorted_models, speed_data):
    apply_light_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("Quality vs Speed Trade-off (with reference)", fontsize=14,
                 fontweight="bold", color=TEXT_C, y=1.02)

    qualities = [results[m]["overall"]["exact_match_pct"] for m in sorted_models]
    speeds = [speed_data[m]["median_time_ms"] for m in sorted_models]
    fps = [speed_data[m]["median_fps"] for m in sorted_models]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(sorted_models))]

    sizes = [results[m]["overall"]["n"] / 10 for m in sorted_models]

    scatter = ax.scatter(speeds, qualities, c=colors, s=sizes,
                         alpha=0.7, edgecolors='white', linewidth=2, zorder=3)

    for i, m in enumerate(sorted_models):
        display_name = m.replace("nanonets-ocr2-3b@", "nanonets@")
        ax.annotate(display_name,
                    xy=(speeds[i], qualities[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    color=TEXT_C,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white',
                              edgecolor=colors[i],
                              alpha=0.8))

    ax.set_xlabel("Inference Time (ms) faster", fontsize=11, fontweight="bold")
    ax.set_ylabel("Exact Match % better", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, zorder=0)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    speed_ticks = ax.get_xticks()
    fps_ticks = [1000 / t if t > 0 else 0 for t in speed_ticks]
    ax2.set_xticks(speed_ticks)
    ax2.set_xticklabels([f"{f:.1f}" for f in fps_ticks])
    ax2.set_xlabel("FPS faster", fontsize=11, fontweight="bold")

    fig.tight_layout()
    p = OUT_DIR / "08_quality_vs_speed.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {p}")


def print_results(results, sorted_models, speed_data, ref_speed):
    print("\n" + "=" * 120)
    print("OVERALL (reference: qwen3-vl-8b) - ТОЛЬКО РОССИЙСКИЙ ФОРМАТ")
    print("=" * 120)
    print(
        f"{'Model':<35} {'Orig':>6} {'Filt':>6} {'N':>6} {'Exact%':>8} {'MeanCER':>9} {'Empty/Long':>12} {'Speed(ms)':>10} {'FPS':>8}")
    print("-" * 120)
    for m in sorted_models:
        if m == REFERENCE_MODEL:
            continue
        o = results[m]["overall"]
        speed = speed_data[m]["median_time_ms"]
        fps = speed_data[m]["median_fps"]
        print(f"{m:<35} {o['original_count']:>6} {o['filtered_out']:>6} "
              f"{o['n']:>6} {o['exact_match_pct']:>7.2f}% "
              f"{o['mean_cer']:>9.4f} {o['n_empty_hyp']:>12} "
              f"{speed:>10.0f} {fps:>8.1f}")

    print("-" * 120)
    print(f"{REFERENCE_MODEL + ' (reference)':<35} "
          f"{ref_speed['median_time_ms']:>47.0f} {ref_speed['median_fps']:>8.1f}")

    print("\n" + "=" * 120)
    print("PER FOLDER ExactMatch% (только валидные номера)")
    print("=" * 120)
    header = f"{'Model':<35}" + "".join(f"{f:>9}" for f in SUBFOLDERS)
    print(header)
    print("-" * 120)
    for m in sorted_models:
        if m == REFERENCE_MODEL:
            continue
        row = f"{m:<35}"
        for f in SUBFOLDERS:
            row += f"{results[m]['per_folder'][f]['exact_match_pct']:>8.1f}%"
        print(row)


def main():
    ref_path = BASE_DIR / REFERENCE_MODEL / REFERENCE_JSON
    print(f"Loading reference: {ref_path}")
    ref_records = load_json(ref_path)
    ref_global = {r["filename"]: r["plate_text_raw"] for r in ref_records}
    print(f"Reference records (всего): {len(ref_global)}")

    valid_refs = {k: v for k, v in ref_global.items()
                  if is_valid_russian_plate(normalize(v))}

    print("Building reference hit set (edit_distance ≤ 2)...")
    ref_hit_set = build_reference_hit_set(valid_refs, REFERENCE_MODEL, REFERENCE_JSON)
    print(f"Reference hits: {len(ref_hit_set)} / {len(valid_refs)} "
          f"({len(ref_hit_set) / len(valid_refs) * 100:.1f}% валидных номеров)")

    print(f"Reference records (российский формат): {len(valid_refs)}")
    print(f"Отфильтровано: {len(ref_global) - len(valid_refs)} записей\n")

    ref_speed = collect_speed_data(REFERENCE_MODEL, REFERENCE_JSON)
    print(f"Reference model ({REFERENCE_MODEL}) speed:")
    print(f"    Median time: {ref_speed['median_time_ms']:.0f} ms, "
          f"Median FPS: {ref_speed['median_fps']:.1f}\n")

    speed_data = {}
    for model_name, json_file in MODELS.items():
        print(f"Collecting speed data for: {model_name}...")
        speed_data[model_name] = collect_speed_data(model_name, json_file)
        print(f"    Median time: {speed_data[model_name]['median_time_ms']:.0f} ms, "
              f"Median FPS: {speed_data[model_name]['median_fps']:.1f}")

    speed_data[REFERENCE_MODEL] = ref_speed

    results = {}
    for model_name, json_file in MODELS.items():
        print(f"Processing: {model_name}...")
        results[model_name] = compare_model(model_name, json_file, valid_refs,
            allowed_filenames=ref_hit_set )

    sorted_models = sorted(
        MODELS.keys(),
        key=lambda m: results[m]["overall"]["exact_match_pct"],
        reverse=True,
    )

    all_models_speed = sorted_models + [REFERENCE_MODEL]

    print_results(results, sorted_models, speed_data, ref_speed)

    out_json = {}
    for m in sorted_models:
        out_json[m] = {
            "overall": results[m]["overall"],
            "per_folder": results[m]["per_folder"],
            "speed": speed_data[m]
        }
    out_json[REFERENCE_MODEL] = {
        "speed": ref_speed
    }
    out_path = BASE_DIR / "comparison_results_filtered.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"\nJSON saved: {out_path}")

    print(f"\nGenerating charts {OUT_DIR}/")

    chart_overall_bars(results, sorted_models, speed_data)
    chart_heatmap(results, sorted_models, "exact_match_pct",
                  "Exact Match %", lambda v: f"{v:.0f}%", "YlOrRd")
    chart_heatmap(results, sorted_models, "mean_cer",
                  "Mean CER", lambda v: f"{v:.3f}", "RdYlGn_r")
    chart_per_day_lines(results, sorted_models)
    chart_empty_hyp(results, sorted_models)
    chart_cer_boxplot(results, sorted_models)
    chart_radar(results, sorted_models)

    chart_speed_comparison(all_models_speed, speed_data)
    chart_quality_vs_speed(results, sorted_models, speed_data)

    print("\nDone! All charts saved.")


if __name__ == "__main__":
    main()