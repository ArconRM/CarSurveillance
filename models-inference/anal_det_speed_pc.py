import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

# =======================
# НАСТРОЙКИ
# =======================

ROOT_DIR = Path("/home/artemiy/Documents/test_det/pc")
IGNORE_MODELS = ["yolov8l", "ocr", "gemma-3-4b", "qwen3-vl-4b"]  # Игнорируемые модели


# =======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =======================

def parse_model_device(folder_name):
    """
    Разбирает имя папки на model и device.
    Для PC: yolov8n_cpu -> model='yolov8n', device='cpu'
           yolov8n_cuda -> model='yolov8n', device='cuda'
    """
    parts = folder_name.rsplit('_', 1)

    if len(parts) == 2:
        known_devices = {'cpu', 'cuda', 'mps'}
        if parts[1].lower() in known_devices:
            return parts[0], parts[1].lower()
        else:
            return folder_name, 'cpu'  # если без суффикса, считаем cpu
    else:
        return folder_name, 'cpu'


def load_pc_results(root_dir: Path):
    """
    Загружает результаты только с PC.
    """
    records = []

    print(f"Сканирую директорию: {root_dir}")
    model_folders = [d for d in root_dir.iterdir() if d.is_dir()]
    print(f"Найдено папок с моделями: {len(model_folders)}")

    for model_folder in sorted(model_folders):
        folder_name = model_folder.name
        model_name, device = parse_model_device(folder_name)

        # Пропускаем игнорируемые модели
        if model_name in IGNORE_MODELS or folder_name in IGNORE_MODELS:
            print(f"  ⊘ {folder_name}: пропущено (в списке игнорирования)")
            continue

        # Ищем JSON файлы
        json_files = list(model_folder.glob("*.json"))

        if len(json_files) == 0:
            print(f"  ⚠ {folder_name}: нет JSON файлов")
            continue

        print(f"  ✓ {folder_name} -> модель={model_name}, устройство={device}, файлов={len(json_files)}")

        for json_path in json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    items = [data]
                elif isinstance(data, list):
                    items = data
                else:
                    continue

                for item in items:
                    # Скорость в мс -> переводим в секунды
                    speed_ms = None
                    speed_sec = None
                    if "speed" in item and isinstance(item["speed"], dict):
                        speed_ms = item["speed"].get("inference_ms", None)
                        if speed_ms is not None:
                            speed_sec = speed_ms / 1000.0

                    confidence = item.get("confidence", None)
                    class_id = item.get("class", None)

                    bbox = None
                    if "coordinates" in item:
                        coords = item["coordinates"]
                        bbox = (
                            coords["x1"], coords["y1"],
                            coords["x2"], coords["y2"]
                        )

                    frame = ""
                    if "frame" in item:
                        frame = Path(item["frame"]).name

                    records.append({
                        "model": model_name,
                        "device": device,
                        "frame": frame,
                        "bbox": bbox,
                        "confidence": confidence,
                        "class_id": class_id,
                        "inference_ms": speed_ms,
                        "inference_sec": speed_sec,
                    })

            except Exception as e:
                print(f"    Ошибка загрузки {json_path.name}: {e}")
                continue

    return pd.DataFrame(records)


# =======================
# ЗАГРУЗКА ДАННЫХ
# =======================

print("=" * 60)
print("ЗАГРУЗКА РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ СКОРОСТИ (PC)")
print("=" * 60)

df = load_pc_results(ROOT_DIR)

if df.empty:
    print("\nОШИБКА: Данные не загружены!")
    exit(1)

print(f"\n{'=' * 60}")
print(f"Всего записей загружено: {len(df)}")
print(f"Модели: {sorted(df['model'].unique())}")
print(f"Устройства: {sorted(df['device'].unique())}")

# Показываем, какие модели есть на обоих устройствах
model_devices = df.groupby("model")["device"].apply(set)
print("\nМодели с несколькими устройствами:")
for model, devices in model_devices.items():
    if len(devices) > 1:
        print(f"  {model}: {devices}")

# =======================
# АНАЛИЗ СКОРОСТИ
# =======================

print(f"\n{'=' * 60}")
print("АНАЛИЗ СКОРОСТИ (PC, CPU vs CUDA)")
print("=" * 60)

# Статистика по скорости в секундах
speed_stats = df.groupby(["model", "device"]).agg(
    количество=("inference_sec", "count"),
    среднее_сек=("inference_sec", "mean"),
    станд_откл_сек=("inference_sec", "std"),
    мин_сек=("inference_sec", "min"),
    макс_сек=("inference_sec", "max"),
    медиана_сек=("inference_sec", "median"),
    средняя_уверенность=("confidence", "mean"),
    x1_среднее=("bbox", lambda x: x.apply(lambda b: b[0] if b else None).mean()),
    y1_среднее=("bbox", lambda x: x.apply(lambda b: b[1] if b else None).mean()),
    x2_среднее=("bbox", lambda x: x.apply(lambda b: b[2] if b else None).mean()),
    y2_среднее=("bbox", lambda x: x.apply(lambda b: b[3] if b else None).mean()),
).reset_index()

# Сортируем по модели и устройству
speed_stats = speed_stats.sort_values(["model", "device"])

print("\nДетальная статистика скорости (в секундах):")
print(speed_stats[["model", "device", "количество", "среднее_сек", "станд_откл_сек",
                   "медиана_сек", "средняя_уверенность"]].to_string(index=False))

# =======================
# СРАВНЕНИЕ CPU vs CUDA
# =======================

print(f"\n{'=' * 60}")
print("СРАВНЕНИЕ CPU vs CUDA ДЛЯ ОДИНАКОВЫХ МОДЕЛЕЙ")
print("=" * 60)

# Находим модели, которые есть и на CPU и на CUDA
model_device_count = df.groupby("model")["device"].apply(lambda x: len(set(x)))
comparable_models = model_device_count[model_device_count > 1].index

for model in comparable_models:
    cpu_data = speed_stats[(speed_stats["model"] == model) & (speed_stats["device"] == "cpu")]
    cuda_data = speed_stats[(speed_stats["model"] == model) & (speed_stats["device"] == "cuda")]

    if not cpu_data.empty and not cuda_data.empty:
        cpu_time = cpu_data["среднее_сек"].iloc[0]
        cuda_time = cuda_data["среднее_сек"].iloc[0]
        speedup = cpu_time / cuda_time

        cpu_conf = cpu_data["средняя_уверенность"].iloc[0]
        cuda_conf = cuda_data["средняя_уверенность"].iloc[0]

        print(f"\n{model}:")
        print(f"  CPU:  {cpu_time:.3f}с (уверенность: {cpu_conf * 100:.1f}%)")
        print(f"  CUDA: {cuda_time:.3f}с (уверенность: {cuda_conf * 100:.1f}%)")
        print(f"  ⚡ CUDA быстрее в {speedup:.2f} раз(а)")

# =======================
# ВИЗУАЛИЗАЦИЯ
# =======================

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

# 1. Сравнение скорости CPU vs CUDA (группированный бар-чарт)
fig, ax = plt.subplots(figsize=(14, 8))

# Берем только сравнимые модели
comparable_stats = speed_stats[speed_stats["model"].isin(comparable_models)]

# Pivot для grouped bar chart
pivot_speed = comparable_stats.pivot(index="model", columns="device", values="среднее_сек")

# Сортируем по скорости CUDA (если есть) или CPU
if "cuda" in pivot_speed.columns and "cpu" in pivot_speed.columns:
    pivot_speed = pivot_speed.sort_values("cuda")
elif "cpu" in pivot_speed.columns:
    pivot_speed = pivot_speed.sort_values("cpu")

ax = pivot_speed.plot(kind="bar", ax=ax, color=["#3498db", "#e74c3c"], rot=45)

ax.set_ylabel("Среднее время инференса (сек)")
ax.set_title("Сравнение скорости CPU vs CUDA (PC)")
ax.legend(title="Устройство")
ax.grid(axis='y', alpha=0.3)

# Добавляем значения на бары
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=9, padding=3)

plt.tight_layout()
plt.show()

# 2. Горизонтальное сравнение (более читаемое)
fig, ax = plt.subplots(figsize=(12, 8))

# Создаем горизонтальные бары для каждой модели
models_sorted = pivot_speed.index.tolist()
y_positions = np.arange(len(models_sorted)) * 2  # Расстояние между группами
bar_height = 0.8

for i, model in enumerate(models_sorted):
    cpu_time = pivot_speed.loc[model, "cpu"] if "cpu" in pivot_speed.columns else 0
    cuda_time = pivot_speed.loc[model, "cuda"] if "cuda" in pivot_speed.columns else 0

    # CPU бар
    ax.barh(y_positions[i], cpu_time, bar_height, color="#3498db", alpha=0.8, label="CPU" if i == 0 else "")
    # CUDA бар
    ax.barh(y_positions[i] + bar_height, cuda_time, bar_height, color="#e74c3c", alpha=0.8,
            label="CUDA" if i == 0 else "")

    # Добавляем значения
    if cpu_time > 0:
        ax.text(cpu_time, y_positions[i], f" {cpu_time:.3f}с", va='center', fontsize=9)
    if cuda_time > 0:
        ax.text(cuda_time, y_positions[i] + bar_height, f" {cuda_time:.3f}с", va='center', fontsize=9)

ax.set_yticks(y_positions + bar_height / 2)
ax.set_yticklabels(models_sorted, fontsize=10)
ax.set_xlabel("Среднее время инференса (сек)")
ax.set_title("Сравнение скорости CPU vs CUDA по моделям")
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# 3. Ускорение CUDA относительно CPU
fig, ax = plt.subplots(figsize=(12, 6))

speedup_data = []
for model in comparable_models:
    cpu_time = pivot_speed.loc[model, "cpu"] if "cpu" in pivot_speed.columns else None
    cuda_time = pivot_speed.loc[model, "cuda"] if "cuda" in pivot_speed.columns else None

    if cpu_time and cuda_time and cuda_time > 0:
        speedup = cpu_time / cuda_time
        speedup_data.append({
            "model": model,
            "speedup": speedup,
            "cpu_time": cpu_time,
            "cuda_time": cuda_time
        })

if speedup_data:
    speedup_df = pd.DataFrame(speedup_data).sort_values("speedup")

    bars = ax.barh(speedup_df["model"], speedup_df["speedup"], color="#2ecc71", alpha=0.8)

    # Добавляем значения
    for bar, speedup in zip(bars, speedup_df["speedup"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {speedup:.1f}x", va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel("Ускорение (раз)")
    ax.set_title("Во сколько раз CUDA быстрее CPU")
    ax.grid(axis='x', alpha=0.3)

    # Добавляем линию 1x (нет ускорения)
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='1x (нет ускорения)')
    ax.legend()

    plt.tight_layout()
    plt.show()

# 4. ⚡ ТОЧЕЧНАЯ ДИАГРАММА: ТОЧНОСТЬ vs ПРОИЗВОДИТЕЛЬНОСТЬ ⚡
fig, ax = plt.subplots(figsize=(16, 12))

# Данные для scatter plot
scatter_data = comparable_stats.dropna(subset=["среднее_сек", "средняя_уверенность"])

# Разные маркеры и размеры для устройств
device_config = {
    "cpu": {"color": "#3498db", "marker": "o", "size": 200, "label": "CPU", "zorder": 3},
    "cuda": {"color": "#e74c3c", "marker": "s", "size": 250, "label": "CUDA", "zorder": 5}
}

for device, config in device_config.items():
    device_data = scatter_data[scatter_data["device"] == device]
    if not device_data.empty:
        ax.scatter(
            device_data["среднее_сек"],
            device_data["средняя_уверенность"] * 100,
            c=config["color"],
            marker=config["marker"],
            s=config["size"],
            label=config["label"],
            alpha=0.8,
            edgecolors='black',
            linewidth=2,
            zorder=config["zorder"]
        )

# Соединяем линиями одинаковые модели на разных устройствах
for model in comparable_models:
    model_data = scatter_data[scatter_data["model"] == model]
    if len(model_data) == 2:  # Есть и CPU и CUDA
        cpu_row = model_data[model_data["device"] == "cpu"]
        cuda_row = model_data[model_data["device"] == "cuda"]

        if not cpu_row.empty and not cuda_row.empty:
            ax.plot(
                [cpu_row["среднее_сек"].iloc[0], cuda_row["среднее_сек"].iloc[0]],
                [cpu_row["средняя_уверенность"].iloc[0] * 100, cuda_row["средняя_уверенность"].iloc[0] * 100],
                'gray', alpha=0.4, linewidth=2, linestyle='--', zorder=1
            )

# Подписи для КАЖДОЙ точки (и CPU и CUDA)
for _, row in scatter_data.iterrows():
    # Формируем подпись: модель + устройство
    label = f"{row['model']}\n[{row['device'].upper()}]"

    # Немного смещаем подписи для CPU вверх, для CUDA вниз чтобы не накладывались
    if row["device"] == "cpu":
        xytext = (15, 15)
    else:
        xytext = (15, -15)

    ax.annotate(
        label,
        (row["среднее_сек"], row["средняя_уверенность"] * 100),
        fontsize=9,
        fontweight='bold',
        xytext=xytext,
        textcoords='offset points',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            alpha=0.8,
            edgecolor='gray',
            linewidth=1
        ),
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle='arc3,rad=0.2',
            color='gray',
            alpha=0.6,
            linewidth=1
        )
    )

# Зоны эффективности с подписями
ax.axvline(x=0.1, color='#27ae60', linestyle=':', alpha=0.6, linewidth=3)
ax.axvline(x=0.5, color='#f39c12', linestyle=':', alpha=0.6, linewidth=3)
ax.axhline(y=90, color='#27ae60', linestyle='--', alpha=0.6, linewidth=3)
ax.axhline(y=70, color='#f39c12', linestyle='--', alpha=0.6, linewidth=3)

# Подписи зон с фоном
zone_style = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
ax.text(0.03, 96, '🎯 Идеальная зона\n(быстро + точно)', fontsize=11, color='#27ae60',
        fontweight='bold', bbox=zone_style)
ax.text(1.5, 96, '🐌 Медленно,\nно точно', fontsize=11, color='#f39c12',
        fontweight='bold', bbox=zone_style)
ax.text(0.03, 60, '⚡ Быстро,\nно неточно', fontsize=11, color='#f39c12',
        fontweight='bold', bbox=zone_style)
ax.text(1.5, 60, '👎 Медленно\nи неточно', fontsize=11, color='#e74c3c',
        fontweight='bold', bbox=zone_style)

# Настройка осей
ax.set_xlabel("Среднее время инференса (секунды)", fontsize=12, fontweight='bold')
ax.set_ylabel("Средняя уверенность (%)", fontsize=12, fontweight='bold')
ax.set_title("Точность vs Производительность: CPU vs CUDA (PC)", fontsize=14, fontweight='bold', pad=20)

# Добавляем сетку
ax.grid(True, alpha=0.2, linestyle='--')

# Легенда
legend = ax.legend(
    title="Устройство",
    loc='lower left',
    fontsize=11,
    title_fontsize=12,
    framealpha=0.9,
    edgecolor='black'
)

# Добавляем информацию о стрелках в легенду
from matplotlib.lines import Line2D

legend_elements = legend.get_patches()
legend_elements.append(Line2D([0], [0], color='gray', linestyle='--', alpha=0.4,
                              linewidth=2, label='CPU → CUDA (ускорение)'))
ax.legend(handles=legend_elements, title="Обозначения", loc='lower left',
          fontsize=11, title_fontsize=12, framealpha=0.9, edgecolor='black')

# Устанавливаем пределы с отступами для подписей
x_min = scatter_data["среднее_сек"].min()
x_max = scatter_data["среднее_сек"].max()
y_min = scatter_data["средняя_уверенность"].min() * 100
y_max = scatter_data["средняя_уверенность"].max() * 100

ax.set_xlim(x_min - 0.1, x_max * 1.3)  # Добавляем место справа
ax.set_ylim(y_min - 5, y_max + 5)

plt.tight_layout()
plt.show()

# =======================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =======================

output_dir = Path("/home/artemiy/Documents/test_det/pc")
speed_stats.to_csv(output_dir / "анализ_скорости_pc.csv", index=False)
df.to_csv(output_dir / "исходные_данные_pc.csv", index=False)

print(f"\n{'=' * 60}")
print("Результаты сохранены:")
print(f"  {output_dir / 'анализ_скорости_pc.csv'}")
print(f"  {output_dir / 'исходные_данные_pc.csv'}")

# Сводный отчет
print(f"\n{'=' * 60}")
print("СВОДНЫЙ ОТЧЕТ: PC (CPU vs CUDA)")
print("=" * 60)

print("\nТОП-5 САМЫХ БЫСТРЫХ КОНФИГУРАЦИЙ:")
top5 = speed_stats.nsmallest(5, "среднее_сек")[["model", "device", "среднее_сек", "средняя_уверенность"]]
for _, row in top5.iterrows():
    time_str = f"{row['среднее_сек']:.3f}с"
    conf_str = f" | Уверенность: {row['средняя_уверенность'] * 100:.1f}%"
    print(f"  {row['model']:20s} [{row['device']:4s}]: {time_str}{conf_str}")

print("\nСРАВНЕНИЕ CPU vs CUDA:")
for model in comparable_models:
    model_stats = speed_stats[speed_stats["model"] == model]
    if len(model_stats) == 2:
        cpu_row = model_stats[model_stats["device"] == "cpu"]
        cuda_row = model_stats[model_stats["device"] == "cuda"]
        if not cpu_row.empty and not cuda_row.empty:
            speedup = cpu_row["среднее_сек"].iloc[0] / cuda_row["среднее_сек"].iloc[0]
            print(f"  {model:20s}: CUDA быстрее в {speedup:.1f}x  "
                  f"(CPU: {cpu_row['среднее_сек'].iloc[0]:.3f}с, CUDA: {cuda_row['среднее_сек'].iloc[0]:.3f}с)")

print(f"\n{'=' * 60}")
print("ВЫВОДЫ:")
print("=" * 60)

# Находим лучший баланс скорость/точность
best_balance = speed_stats.nsmallest(3, "среднее_сек")
best_balance = best_balance[best_balance["средняя_уверенность"] > 0.7]

if not best_balance.empty:
    print("\nЛучший баланс скорость/точность:")
    for _, row in best_balance.iterrows():
        print(
            f"  {row['model']} [{row['device']}]: {row['среднее_сек']:.3f}с, уверенность {row['средняя_уверенность'] * 100:.1f}%")
else:
    print("\nНи одна модель не показала уверенность > 70%")