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

ROOT_DIR = Path("/home/artemiy/Documents/test_det")
IGNORE_MODELS = ["yolov8l"]  # Игнорируемые модели


# =======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =======================

def parse_model_name(folder_name):
    """
    Разбирает имя папки на model и device.
    """
    parts = folder_name.rsplit('_', 1)

    if len(parts) == 2:
        known_devices = {'cpu', 'cuda', 'mps'}
        if parts[1].lower() in known_devices:
            return parts[0], parts[1].lower()
        else:
            return folder_name, 'default'
    else:
        return folder_name, 'default'


def load_speed_results(root_dir: Path):
    """
    Загружает все результаты скорости и точности из структуры:
    machine/model_folder/*.json
    """
    records = []

    print(f"Сканирую директорию: {root_dir}")
    machines = [d for d in root_dir.iterdir() if d.is_dir()]
    print(f"Найдены машины: {[m.name for m in machines]}")

    for machine_dir in sorted(machines):
        machine = machine_dir.name
        print(f"\nОбрабатываю машину: {machine}")

        model_folders = [d for d in machine_dir.iterdir() if d.is_dir()]
        print(f"  Папок с моделями: {len(model_folders)}")

        for model_folder in sorted(model_folders):
            model_name, device = parse_model_name(model_folder.name)

            # Пропускаем игнорируемые модели
            if model_name in IGNORE_MODELS:
                print(f"    ⊘ {model_folder.name}: пропущено (в списке игнорирования)")
                continue

            # Ищем JSON файлы
            json_files = list(model_folder.glob("*.json"))

            if len(json_files) == 0:
                print(f"    ⚠ {model_folder.name}: нет JSON файлов")
                continue

            print(f"    ✓ {model_folder.name} -> модель={model_name}, устройство={device}, файлов={len(json_files)}")

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
                                speed_sec = speed_ms / 1000.0  # мс -> секунды

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
                            "machine": machine,
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
                    print(f"      Ошибка загрузки {json_path.name}: {e}")
                    continue

    return pd.DataFrame(records)


# =======================
# ЗАГРУЗКА ДАННЫХ
# =======================

print("=" * 60)
print("ЗАГРУЗКА РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ СКОРОСТИ")
print("=" * 60)

df = load_speed_results(ROOT_DIR)

if df.empty:
    print("\nОШИБКА: Данные не загружены!")
    exit(1)

print(f"\n{'=' * 60}")
print(f"Всего записей загружено: {len(df)}")
print(f"Машины: {sorted(df['machine'].unique())}")
print(f"Модели: {sorted(df['model'].unique())}")
print(f"Устройства: {sorted(df['device'].unique())}")

# =======================
# АНАЛИЗ СКОРОСТИ (в секундах)
# =======================

print(f"\n{'=' * 60}")
print("АНАЛИЗ СКОРОСТИ (в секундах)")
print("=" * 60)

# Статистика по скорости в секундах
speed_stats = df.groupby(["machine", "model", "device"]).agg(
    количество=("inference_sec", "count"),
    среднее_сек=("inference_sec", "mean"),
    станд_откл_сек=("inference_sec", "std"),
    мин_сек=("inference_sec", "min"),
    макс_сек=("inference_sec", "max"),
    q25_сек=("inference_sec", lambda x: x.quantile(0.25)),
    медиана_сек=("inference_sec", lambda x: x.quantile(0.50)),
    q75_сек=("inference_sec", lambda x: x.quantile(0.75)),
    средняя_уверенность=("confidence", "mean"),
).reset_index()

# Сортируем по скорости
speed_stats = speed_stats.sort_values(["machine", "среднее_сек"])

print("\nДетальная статистика скорости (в секундах):")
print(speed_stats.to_string(index=False))

# Топ-5 самых быстрых конфигураций
print("\n🏆 ТОП-5 САМЫХ БЫСТРЫХ:")
top5 = speed_stats.nsmallest(5, "среднее_сек")[["machine", "model", "device", "среднее_сек", "станд_откл_сек"]]
print(top5.to_string(index=False))

# Топ-5 самых медленных
print("\n🐌 ТОП-5 САМЫХ МЕДЛЕННЫХ:")
bottom5 = speed_stats.nlargest(5, "среднее_сек")[["machine", "model", "device", "среднее_сек", "станд_откл_сек"]]
print(bottom5.to_string(index=False))

# =======================
# СРАВНЕНИЕ УСТРОЙСТВ ДЛЯ ОДИНАКОВЫХ МОДЕЛЕЙ
# =======================

print(f"\n{'=' * 60}")
print("СРАВНЕНИЕ УСТРОЙСТВ (ОДИНАКОВЫЕ МОДЕЛИ, РАЗНЫЕ УСТРОЙСТВА)")
print("=" * 60)

model_devices = df.groupby(["machine", "model"])["device"].apply(set).reset_index()
model_devices["количество_устройств"] = model_devices["device"].apply(len)
multi_device = model_devices[model_devices["количество_устройств"] > 1]

for _, row in multi_device.iterrows():
    machine = row["machine"]
    model = row["model"]

    subset = speed_stats[(speed_stats["machine"] == machine) & (speed_stats["model"] == model)]

    print(f"\n{machine} / {model}:")
    print(subset[["device", "среднее_сек", "станд_откл_сек", "количество"]].to_string(index=False))

    if len(subset) >= 2:
        fastest = subset.loc[subset["среднее_сек"].idxmin()]
        slowest = subset.loc[subset["среднее_сек"].idxmax()]
        speedup = slowest["среднее_сек"] / fastest["среднее_сек"]
        print(f"  ⚡ {fastest['device']} быстрее {slowest['device']} в {speedup:.2f} раз(а)")

# =======================
# ВИЗУАЛИЗАЦИЯ
# =======================

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

# 1. Общий обзор скорости по всем машинам и моделям (в секундах)
fig, ax = plt.subplots(figsize=(16, 8))

speed_stats["метка"] = speed_stats.apply(
    lambda x: f"{x['model']}\n({x['device']})", axis=1
)

bars = ax.bar(
    range(len(speed_stats)),
    speed_stats["среднее_сек"],
    yerr=speed_stats["станд_откл_сек"],
    capsize=3,
    alpha=0.8
)

colors = plt.cm.Set3(np.linspace(0, 1, len(speed_stats["machine"].unique())))
machine_colors = dict(zip(sorted(speed_stats["machine"].unique()), colors))

for i, (bar, machine) in enumerate(zip(bars, speed_stats["machine"])):
    bar.set_color(machine_colors[machine])

ax.set_xticks(range(len(speed_stats)))
ax.set_xticklabels(speed_stats["метка"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Время инференса (сек)")
ax.set_title("Сравнение скорости инференса моделей (со станд. откл.)")

legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=machine_colors[m], label=m)
                   for m in sorted(machine_colors.keys())]
ax.legend(handles=legend_elements, title="Машина", bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()

# 2. Детальные графики по каждой машине
for machine in sorted(df["machine"].unique()):
    machine_data = speed_stats[speed_stats["machine"] == machine].copy()

    if machine_data.empty:
        continue

    machine_data = machine_data.sort_values("среднее_сек", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = machine_data.apply(lambda x: f"{x['model']}_{x['device']}", axis=1)

    bars = ax.barh(range(len(machine_data)), machine_data["среднее_сек"])

    devices = machine_data["device"].unique()
    device_colors = plt.cm.Set2(np.linspace(0, 1, len(devices)))
    device_color_map = dict(zip(devices, device_colors))

    for i, (bar, device) in enumerate(zip(bars, machine_data["device"])):
        bar.set_color(device_color_map[device])

    ax.set_yticks(range(len(machine_data)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Время инференса (сек)")
    ax.set_title(f"Сравнение скорости - {machine}")

    # Добавляем значения на бары в секундах
    for i, (v, std) in enumerate(zip(machine_data["среднее_сек"], machine_data["станд_откл_сек"])):
        if not pd.isna(v):
            if v < 1.0:
                ax.text(v, i, f" {v * 1000:.0f}мс", va='center', fontsize=8)
            else:
                ax.text(v, i, f" {v:.2f}с", va='center', fontsize=8)

    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=device_color_map[d], label=d)
                       for d in devices]
    ax.legend(handles=legend_elements, title="Устройство", loc='lower right')

    plt.tight_layout()
    plt.show()

# 3. Сравнение CPU vs GPU/MPS для каждой машины
for machine in sorted(df["machine"].unique()):
    machine_df = df[df["machine"] == machine]

    devices = machine_df["device"].unique()

    if len(devices) <= 1:
        continue

    model_device_count = machine_df.groupby("model")["device"].apply(lambda x: len(set(x)))
    multi_device_models = model_device_count[model_device_count > 1].index

    if len(multi_device_models) == 0:
        continue

    fig, ax = plt.subplots(figsize=(12, 6))

    plot_data = []
    for model in multi_device_models:
        for device in devices:
            subset = machine_df[(machine_df["model"] == model) & (machine_df["device"] == device)]
            if not subset.empty:
                avg_speed = subset["inference_sec"].mean()
                plot_data.append({
                    "Модель": model,
                    "Устройство": device,
                    "Среднее время (сек)": avg_speed
                })

    plot_df = pd.DataFrame(plot_data)

    pivot = plot_df.pivot(index="Модель", columns="Устройство", values="Среднее время (сек)")

    pivot.plot(kind="bar", ax=ax, rot=45)
    ax.set_ylabel("Среднее время инференса (сек)")
    ax.set_title(f"Сравнение CPU vs GPU/MPS - {machine}")
    ax.legend(title="Устройство")

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)

    plt.tight_layout()
    plt.show()

# 4. ⚡ ТОЧЕЧНАЯ ДИАГРАММА: ТОЧНОСТЬ vs ПРОИЗВОДИТЕЛЬНОСТЬ ⚡
print(f"\n{'=' * 60}")
print("АНАЛИЗ ТОЧНОСТЬ vs ПРОИЗВОДИТЕЛЬНОСТЬ")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

# 4.1 Общая точечная диаграмма всех конфигураций
ax = axes[0]

scatter_data = speed_stats.dropna(subset=["среднее_сек", "средняя_уверенность"])

# Разные маркеры для разных машин
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
machine_markers = {m: markers[i % len(markers)] for i, m in enumerate(sorted(scatter_data["machine"].unique()))}

for machine in sorted(scatter_data["machine"].unique()):
    machine_data = scatter_data[scatter_data["machine"] == machine]
    ax.scatter(
        machine_data["среднее_сек"],
        machine_data["средняя_уверенность"] * 100,  # в проценты
        c=[machine_colors[machine]] * len(machine_data),
        marker=machine_markers[machine],
        s=100,
        label=machine,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

# Добавляем подписи для каждой точки
for _, row in scatter_data.iterrows():
    ax.annotate(
        f"{row['model']}\n{row['device']}",
        (row["среднее_сек"], row["средняя_уверенность"] * 100),
        fontsize=7,
        alpha=0.7,
        xytext=(5, 5),
        textcoords='offset points'
    )

ax.set_xlabel("Среднее время инференса (сек)")
ax.set_ylabel("Средняя уверенность (%)")
ax.set_title("Точность vs Производительность (все конфигурации)")
ax.legend(title="Машина", bbox_to_anchor=(1.05, 1))
ax.grid(True, alpha=0.3)

# 4.2 По машинам отдельно
for idx, machine in enumerate(sorted(df["machine"].unique())):
    ax = axes[idx + 1]

    machine_scatter = scatter_data[scatter_data["machine"] == machine]

    if machine_scatter.empty:
        continue

    # Разные цвета для разных устройств
    devices = machine_scatter["device"].unique()
    dev_colors = plt.cm.Set1(np.linspace(0, 1, len(devices)))
    device_colors = {d: dev_colors[i] for i, d in enumerate(devices)}

    for device in sorted(devices):
        device_data = machine_scatter[machine_scatter["device"] == device]
        ax.scatter(
            device_data["среднее_сек"],
            device_data["средняя_уверенность"] * 100,
            c=[device_colors[device]] * len(device_data),
            s=120,
            label=device,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    # Подписи
    for _, row in machine_scatter.iterrows():
        ax.annotate(
            row["model"],
            (row["среднее_сек"], row["средняя_уверенность"] * 100),
            fontsize=8,
            alpha=0.8,
            xytext=(5, 5),
            textcoords='offset points',
            fontweight='bold'
        )

    ax.set_xlabel("Среднее время инференса (сек)")
    ax.set_ylabel("Средняя уверенность (%)")
    ax.set_title(f"Точность vs Производительность - {machine}")
    ax.legend(title="Устройство")
    ax.grid(True, alpha=0.3)

    # Добавляем линии-ориентиры
    # Быстрая зона (< 0.5 сек)
    ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.3, label='Быстро (< 0.5с)')
    # Высокая точность (> 80%)
    ax.axhline(y=80, color='blue', linestyle='--', alpha=0.3, label='Высокая точность (> 80%)')

plt.suptitle("Анализ компромисса между точностью и скоростью", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# 5. Расширенная точечная диаграмма с размерами точек
fig, ax = plt.subplots(figsize=(16, 10))

# Размер точек пропорционален количеству детекций
scatter_data = speed_stats.dropna(subset=["среднее_сек", "средняя_уверенность"])

scatter = ax.scatter(
    scatter_data["среднее_сек"],
    scatter_data["средняя_уверенность"] * 100,
    c=[machine_colors[m] for m in scatter_data["machine"]],
    s=scatter_data["количество"] / 10,  # размер пропорционален количеству
    alpha=0.6,
    edgecolors='black',
    linewidth=0.5
)

# Подписи с доп. информацией
for _, row in scatter_data.iterrows():
    info_text = f"{row['model']} [{row['device']}]"
    ax.annotate(
        info_text,
        (row["среднее_сек"], row["средняя_уверенность"] * 100),
        fontsize=7,
        alpha=0.8,
        xytext=(5, 5),
        textcoords='offset points'
    )

# Зоны эффективности
ax.axvline(x=0.1, color='green', linestyle=':', alpha=0.5, label='Очень быстро (<0.1с)')
ax.axvline(x=0.5, color='orange', linestyle=':', alpha=0.5, label='Быстро (<0.5с)')
ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.5, label='Медленно (>1с)')

ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Отличная точность (>90%)')
ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Хорошая точность (>70%)')

ax.set_xlabel("Среднее время инференса (сек)")
ax.set_ylabel("Средняя уверенность (%)")
ax.set_title("Детальный анализ: Точность vs Скорость (размер точки = количество детекций)")

# Две легенды
legend1 = ax.legend(title="Машина", bbox_to_anchor=(1.15, 1))
legend2 = ax.legend(title="Ориентиры", bbox_to_anchor=(1.15, 0.5))
ax.add_artist(legend1)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Boxplot распределения скорости
fig, ax = plt.subplots(figsize=(16, 8))

plot_df = df.copy()
plot_df["метка"] = plot_df.apply(lambda x: f"{x['machine']}\n{x['model']}\n{x['device']}", axis=1)

medians = plot_df.groupby("метка")["inference_sec"].median().sort_values()
plot_df["метка"] = pd.Categorical(plot_df["метка"], categories=medians.index, ordered=True)

sns.boxplot(data=plot_df, x="метка", y="inference_sec", ax=ax, palette="Set3")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Время инференса (сек)")
ax.set_title("Распределение скорости по конфигурациям")

# Добавляем горизонтальные линии для ориентиров
ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5)
ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# =======================
# СРАВНЕНИЕ МАШИН ДЛЯ ОДИНАКОВЫХ КОНФИГУРАЦИЙ
# =======================

print(f"\n{'=' * 60}")
print("СРАВНЕНИЕ МАШИН ДЛЯ ОДИНАКОВЫХ КОНФИГУРАЦИЙ")
print("=" * 60)

config_machines = df.groupby(["model", "device"])["machine"].apply(set).reset_index()
config_machines["количество_машин"] = config_machines["machine"].apply(len)
multi_machine_configs = config_machines[config_machines["количество_машин"] > 1]

for _, row in multi_machine_configs.iterrows():
    model = row["model"]
    device = row["device"]

    subset = speed_stats[(speed_stats["model"] == model) & (speed_stats["device"] == device)]

    if len(subset) > 1:
        fastest = subset.loc[subset["среднее_сек"].idxmin()]
        slowest = subset.loc[subset["среднее_сек"].idxmax()]

        print(f"\n{model} [{device}]:")
        print(subset[["machine", "среднее_сек", "станд_откл_сек"]].to_string(index=False))
        speedup = slowest["среднее_сек"] / fastest["среднее_сек"]
        print(f"  ⚡ {fastest['machine']} быстрее {slowest['machine']} в {speedup:.2f} раз(а)")

# =======================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =======================

output_dir = Path("/home/artemiy/Documents/test_det")
speed_stats.to_csv(output_dir / "анализ_скорости.csv", index=False)
df.to_csv(output_dir / "исходные_данные_скорости.csv", index=False)

print(f"\n{'=' * 60}")
print("Результаты сохранены:")
print(f"  {output_dir / 'анализ_скорости.csv'}")
print(f"  {output_dir / 'исходные_данные_скорости.csv'}")

# Сводный отчет в секундах
print(f"\n{'=' * 60}")
print("СВОДНЫЙ ОТЧЕТ")
print("=" * 60)

for machine in sorted(df["machine"].unique()):
    print(f"\n📊 {machine.upper()}")
    machine_stats = speed_stats[speed_stats["machine"] == machine].sort_values("среднее_сек")

    for _, row in machine_stats.iterrows():
        time_str = f"{row['среднее_сек']:.3f}с" if row['среднее_сек'] < 1.0 else f"{row['среднее_сек']:.2f}с"
        conf_str = f"{row['средняя_уверенность'] * 100:.1f}%" if not pd.isna(row['средняя_уверенность']) else "N/A"
        print(f"  {row['model']:20s} [{row['device']:6s}]: {time_str:>12s} (±{row['станд_откл_сек']:.3f}с)  "
              f"Уверенность: {conf_str}")

print(f"\n{'=' * 60}")
print("ИТОГО: САМЫЕ БЫСТРЫЕ КОНФИГУРАЦИИ")
print("=" * 60)

for machine in sorted(df["machine"].unique()):
    machine_stats = speed_stats[speed_stats["machine"] == machine].nsmallest(3, "среднее_сек")
    print(f"\n{machine}:")
    for i, (_, row) in enumerate(machine_stats.iterrows(), 1):
        time_str = f"{row['среднее_сек']:.3f}с" if row['среднее_сек'] < 1.0 else f"{row['среднее_сек']:.2f}с"
        conf_str = f" | Точность: {row['средняя_уверенность'] * 100:.1f}%" if not pd.isna(
            row['средняя_уверенность']) else ""
        print(f"  {i}. {row['model']} [{row['device']}]: {time_str}{conf_str}")