import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# =======================
# CONFIG
# =======================

ROOT_DIR = Path("/Volumes/Transcend/data_2/crops")
GT_MODEL = "yolov8l"
IOU_THRESHOLD = 0.5

# =======================
# HELPERS
# =======================

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / (boxAArea + boxBArea - interArea)


def load_model_results(model_dir: Path):
    records = []

    for date_dir in model_dir.iterdir():
        if not date_dir.is_dir():
            continue

        date = date_dir.name

        for json_path in date_dir.iterdir():
            if (
                not json_path.is_file()
                or json_path.name.startswith(".")
                or json_path.suffix.lower() != ".json"
            ):
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                continue

            for item in data:
                records.append({
                    "model": model_dir.name,
                    "date": date,
                    "frame": item["frame"],
                    "bbox": (
                        item["coordinates"]["x1"],
                        item["coordinates"]["y1"],
                        item["coordinates"]["x2"],
                        item["coordinates"]["y2"],
                    ),
                })

    return pd.DataFrame(records)


# =======================
# LOAD DATA
# =======================

dfs = []

print("Loading results...")
for model_dir in tqdm(sorted(ROOT_DIR.iterdir())):
    if model_dir.is_dir():
        df = load_model_results(model_dir)
        if not df.empty:
            dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
print(f"Total detections loaded: {len(all_df)}")

# =======================
# BUILD GT
# =======================

gt_df = all_df[all_df["model"] == GT_MODEL].copy()
print(f"GT objects (yolov8l): {len(gt_df)}")

# =======================
# GLOBAL RECALL (by model)
# =======================

global_stats = []

for model in sorted(all_df["model"].unique()):
    model_df = all_df[all_df["model"] == model]
    found = 0

    for frame, gt_frame_df in gt_df.groupby("frame"):
        model_frame_df = model_df[model_df["frame"] == frame]
        if model_frame_df.empty:
            continue

        model_boxes = model_frame_df["bbox"].tolist()

        for gt_box in gt_frame_df["bbox"]:
            if any(iou(gt_box, mb) >= IOU_THRESHOLD for mb in model_boxes):
                found += 1

    total = len(gt_df)

    global_stats.append({
        "model": model,
        "found_gt": found,
        "gt_total": total,
        "recall_%": found / total * 100
    })

global_df = pd.DataFrame(global_stats).sort_values("recall_%", ascending=False)

print("\n=== GLOBAL RECALL ===")
print(global_df)

# =======================
# PER-DATE RECALL
# =======================

date_stats = []

for date in sorted(all_df["date"].unique()):
    gt_date_df = gt_df[gt_df["date"] == date]
    if gt_date_df.empty:
        continue

    for model in sorted(all_df["model"].unique()):
        model_date_df = all_df[
            (all_df["model"] == model) &
            (all_df["date"] == date)
        ]

        found = 0

        for frame, gt_frame_df in gt_date_df.groupby("frame"):
            model_frame_df = model_date_df[
                model_date_df["frame"] == frame
            ]
            if model_frame_df.empty:
                continue

            model_boxes = model_frame_df["bbox"].tolist()

            for gt_box in gt_frame_df["bbox"]:
                if any(iou(gt_box, mb) >= IOU_THRESHOLD for mb in model_boxes):
                    found += 1

        total = len(gt_date_df)

        date_stats.append({
            "date": date,
            "model": model,
            "found_gt": found,
            "gt_total": total,
            "recall_%": found / total * 100
        })

date_df = pd.DataFrame(date_stats)

print("\n=== PER-DATE RECALL ===")
print(date_df)

# =======================
# PLOTS
# =======================

sns.set(style="whitegrid")

# Global recall
plt.figure(figsize=(12, 6))
sns.barplot(data=global_df, x="model", y="recall_%", palette="viridis")
plt.title("Global Recall vs yolov8l (IoU ≥ 0.5)")
plt.ylabel("Recall (%)")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Heatmap by date
pivot = date_df.pivot(index="model", columns="date", values="recall_%")

plt.figure(figsize=(14, 8))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
plt.title("Recall heatmap by date (IoU ≥ 0.5)")
plt.xlabel("Date")
plt.ylabel("Model")
plt.tight_layout()
plt.show()

# =======================
# SAVE
# =======================

global_df.to_csv("coverage_results_global.csv", index=False)
date_df.to_csv("coverage_results_by_date.csv", index=False)

print("\nSaved:")
print(" - coverage_results_global.csv")
print(" - coverage_results_by_date.csv")
