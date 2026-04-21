import os
import shutil
import random

SRC = "/media/artemiy/EXTERNAL_USB/FUCK/data_2/crops/yolov8l"
DST = "/media/artemiy/EXTERNAL_USB/FUCK/data_2/test/crops"
PERCENT = 0.15  # 15% из каждой подпапки, подкрути по вкусу

random.seed(42)
os.makedirs(DST, exist_ok=True)

total_copied = 0

for subdir in sorted(os.listdir(SRC)):
    subdir_path = os.path.join(SRC, subdir)
    if not os.path.isdir(subdir_path):
        continue

    files = [f for f in os.listdir(subdir_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    n = max(1, round(len(files) * PERCENT))
    sampled = random.sample(files, min(n, len(files)))

    dst_sub = os.path.join(DST, subdir)
    os.makedirs(dst_sub, exist_ok=True)

    for f in sampled:
        shutil.copy2(os.path.join(subdir_path, f), os.path.join(dst_sub, f))

    print(f"{subdir}: {len(files)} файлов → скопировано {len(sampled)}")
    total_copied += len(sampled)

print(f"\nИтого скопировано: {total_copied}")