import os
import time
import json
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from highspy import Highs
from tqdm import tqdm

# === НАСТРОЙКИ ===
DATA_DIR = "benchmarks"
RESULT_FILE = "netlib_gridsearch_results.xlsx"
CONFIG_FILE = "netlib_gridsearch_best_config.json"
SEED = 42
random.seed(SEED)

# === РАСШИРЕННОЕ ПРОСТРАНСТВО ПАРАМЕТРОВ ДЛЯ GRID SEARCH ===

# Общие параметры (для всех солверов)
general_params_grid = {
    "presolve": ["off", "choose", "on"],
    "parallel": ["off", "choose", "on"],
}

# Параметры симплекс-метода
simplex_params_grid = {
    "simplex_strategy": [0, 1, 2, 3, 4],
    "simplex_scale_strategy": [0, 1, 2, 3, 4],
    "simplex_dual_edge_weight_strategy": [-1, 0, 1, 2],
    "simplex_primal_edge_weight_strategy": [-1, 0, 1, 2],
}

# Параметры IPM
ipm_params_grid = {
    "run_crossover": ["off", "choose", "on"],
}


# === ГЕНЕРАЦИЯ ВСЕХ КОМБИНАЦИЙ ===
def generate_all_configs():
    """Генерирует все возможные комбинации параметров"""
    configs = []

    # 1. Simplex конфигурации
    simplex_keys = list(simplex_params_grid.keys())
    simplex_values = [simplex_params_grid[k] for k in simplex_keys]

    general_keys = list(general_params_grid.keys())
    general_values = [general_params_grid[k] for k in general_keys]

    for simplex_combo in itertools.product(*simplex_values):
        for general_combo in itertools.product(*general_values):
            cfg = {"solver": "simplex"}
            for i, key in enumerate(simplex_keys):
                cfg[key] = simplex_combo[i]
            for i, key in enumerate(general_keys):
                cfg[key] = general_combo[i]
            configs.append(cfg)

    # 2. IPM конфигурации
    ipm_keys = list(ipm_params_grid.keys())
    ipm_values = [ipm_params_grid[k] for k in ipm_keys]

    for ipm_combo in itertools.product(*ipm_values):
        for general_combo in itertools.product(*general_values):
            cfg = {"solver": "ipm"}
            for i, key in enumerate(ipm_keys):
                cfg[key] = ipm_combo[i]
            for i, key in enumerate(general_keys):
                cfg[key] = general_combo[i]
            configs.append(cfg)

    # 3. IPX конфигурации
    for ipm_combo in itertools.product(*ipm_values):
        for general_combo in itertools.product(*general_values):
            cfg = {"solver": "ipx"}
            for i, key in enumerate(ipm_keys):
                cfg[key] = ipm_combo[i]
            for i, key in enumerate(general_keys):
                cfg[key] = general_combo[i]
            configs.append(cfg)

    # 4. HiPO конфигурации (только общие параметры)
    for general_combo in itertools.product(*general_values):
        cfg = {"solver": "hipo"}
        for i, key in enumerate(general_keys):
            cfg[key] = general_combo[i]
        configs.append(cfg)

    # 5. PDLP конфигурации (только общие параметры)
    for general_combo in itertools.product(*general_values):
        cfg = {"solver": "pdlp"}
        for i, key in enumerate(general_keys):
            cfg[key] = general_combo[i]
        configs.append(cfg)

    return configs


# === ЗАГРУЗКА И РАЗДЕЛЕНИЕ ЗАДАЧ ===
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".mps")])
random.shuffle(files)

n_train = int(len(files) * 0.7)
train_files = set(files[:n_train])
test_files = set(files[n_train:])

all_configs = generate_all_configs()

print("=" * 70)
print("ПОЛНЫЙ GRID SEARCH ПАРАМЕТРОВ HIGHS НА NETLIB")
print("=" * 70)
print(f"Всего задач: {len(files)}")
print(f"Обучающая выборка: {len(train_files)} задач (70%)")
print(f"Тестовая выборка: {len(test_files)} задач (30%)")
print(f"\nВСЕГО КОНФИГУРАЦИЙ ДЛЯ ПЕРЕБОРА: {len(all_configs)}")

# Подсчет по солверам
solver_counts = {}
for cfg in all_configs:
    solver = cfg["solver"]
    solver_counts[solver] = solver_counts.get(solver, 0) + 1

print("\nРаспределение по солверам:")
for solver, count in sorted(solver_counts.items()):
    print(f"  {solver:10s}: {count:5d} конфигураций")

# Расчет времени
estimated_time_per_config = 2  # секунд на конфигурацию (примерно)
total_time_estimate = (
    len(all_configs) * len(train_files) * estimated_time_per_config / 3600
)
print(f"\nПримерное время выполнения: {total_time_estimate:.1f} часов")
print("=" * 70)

input("\nНажмите Enter для начала grid search или Ctrl+C для отмены...")


# === ФУНКЦИЯ РЕШЕНИЯ ЗАДАЧИ ===
def solve_problem(path, params):
    """Решает задачу с заданными параметрами"""
    h = Highs()
    h.setOptionValue("output_flag", False)

    for k, v in params.items():
        try:
            h.setOptionValue(k, v)
        except Exception:
            pass

    try:
        h.readModel(path)
        t0 = time.monotonic()
        h.run()
        elapsed = time.monotonic() - t0

        return {
            "status": str(h.getModelStatus()),
            "time_sec": elapsed,
            "success": True,
        }
    except Exception as e:
        return {
            "status": f"error: {e.__class__.__name__}",
            "time_sec": None,
            "success": False,
        }


# === ФУНКЦИЯ ОЦЕНКИ КОНФИГУРАЦИИ ===
def evaluate_config(cfg, file_subset):
    """Оценивает конфигурацию на заданном наборе задач"""
    total_time = 0
    count = 0
    failed = 0

    for fname in file_subset:
        path = os.path.join(DATA_DIR, fname)
        result = solve_problem(path, cfg)

        if result["success"] and result["time_sec"] is not None:
            total_time += result["time_sec"]
            count += 1
        else:
            failed += 1

    avg_time = total_time / max(count, 1)
    return avg_time, count, failed, total_time


# === GRID SEARCH ===
print("\n" + "=" * 70)
print("ЗАПУСК GRID SEARCH")
print("=" * 70)

results = []
best_config = None
best_time = float("inf")

for idx, cfg in enumerate(tqdm(all_configs, desc="Grid Search")):
    avg_time, solved, failed, total_time = evaluate_config(cfg, train_files)

    result = {
        "config_id": idx,
        "config": json.dumps(cfg, sort_keys=True),
        "solver": cfg["solver"],
        "avg_time": avg_time,
        "total_time": total_time,
        "solved": solved,
        "failed": failed,
        "success_rate": solved / len(train_files),
    }

    # Добавляем отдельные параметры для удобства фильтрации
    for key, val in cfg.items():
        result[f"param_{key}"] = val

    results.append(result)

    if avg_time < best_time and failed == 0:
        best_time = avg_time
        best_config = cfg.copy()
        print(f"\n>>> НОВЫЙ ЛУЧШИЙ [{idx}]: {json.dumps(cfg)} => {avg_time:.3f}s")

print("\n" + "=" * 70)
print("GRID SEARCH ЗАВЕРШЁН")
print("=" * 70)
print(f"Лучшая конфигурация: {best_config}")
print(f"Среднее время: {best_time:.3f}s")

# === СОЗДАНИЕ DATAFRAME ===
df_grid = pd.DataFrame(results)
df_grid = df_grid.sort_values("avg_time")

# === ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ ЛУЧШЕЙ КОНФИГУРАЦИИ ===
print("\n" + "=" * 70)
print("ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ НА ВСЕХ ЗАДАЧАХ")
print("=" * 70)

# Дефолтные параметры для сравнения
default_params = {
    "solver": "simplex",
    "simplex_strategy": 1,
    "simplex_scale_strategy": 2,
    "simplex_dual_edge_weight_strategy": -1,
    "simplex_primal_edge_weight_strategy": -1,
    "presolve": "choose",
    "parallel": "choose",
}

final_results = []

for fname in tqdm(sorted(files), desc="Final test"):
    path = os.path.join(DATA_DIR, fname)
    is_test = fname in test_files

    # Решение с дефолтными параметрами
    res_default = solve_problem(path, default_params)

    # Решение с лучшими параметрами
    res_best = solve_problem(path, best_config)

    # Информация о задаче
    h_info = Highs()
    h_info.setOptionValue("output_flag", False)
    h_info.readModel(path)

    speedup = None
    if res_default["time_sec"] and res_best["time_sec"]:
        speedup = res_default["time_sec"] / res_best["time_sec"]

    final_results.append(
        {
            "problem": fname,
            "dataset": "TEST" if is_test else "TRAIN",
            "rows": h_info.getNumRow(),
            "cols": h_info.getNumCol(),
            "nonzeros": h_info.getNumNz(),
            "status_default": res_default["status"],
            "time_default": res_default["time_sec"],
            "status_best": res_best["status"],
            "time_best": res_best["time_sec"],
            "speedup": speedup,
        }
    )

df_final = pd.DataFrame(final_results)
df_final["density"] = df_final.apply(
    lambda r: (
        r["nonzeros"] / (r["rows"] * r["cols"]) if r["rows"] and r["cols"] else None
    ),
    axis=1,
)

# === СТАТИСТИКА ===
print("\n" + "=" * 70)
print("ИТОГОВАЯ СТАТИСТИКА")
print("=" * 70)

for dataset in ["TRAIN", "TEST"]:
    subset = df_final[df_final["dataset"] == dataset]
    avg_default = subset["time_default"].mean()
    avg_best = subset["time_best"].mean()
    avg_speedup = subset["speedup"].mean()

    print(f"\n{dataset} выборка ({len(subset)} задач):")
    print(f"  Среднее время (дефолт): {avg_default:.3f}s")
    print(f"  Среднее время (лучшие): {avg_best:.3f}s")
    print(f"  Среднее ускорение: {avg_speedup:.2f}x")

total_avg_default = df_final["time_default"].mean()
total_avg_best = df_final["time_best"].mean()
total_avg_speedup = df_final["speedup"].mean()

print(f"\nВСЕГО ({len(df_final)} задач):")
print(f"  Среднее время (дефолт): {total_avg_default:.3f}s")
print(f"  Среднее время (лучшие): {total_avg_best:.3f}s")
print(f"  Среднее ускорение: {total_avg_speedup:.2f}x")

# === СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
with pd.ExcelWriter(RESULT_FILE, engine="openpyxl") as writer:
    # Итоговые результаты
    df_final_sorted = df_final.sort_values(["dataset", "rows", "cols"])
    df_final_sorted.to_excel(writer, sheet_name="Final_Results", index=False)

    # Все конфигурации grid search
    df_grid.to_excel(writer, sheet_name="Grid_Search", index=False)

    # Top-20 лучших конфигураций
    df_top20 = df_grid.head(20)
    df_top20.to_excel(writer, sheet_name="Top_20_Configs", index=False)

    # Статистика по солверам
    solver_stats = (
        df_grid.groupby("solver")
        .agg(
            {
                "avg_time": ["mean", "min", "max"],
                "solved": "mean",
                "failed": "mean",
            }
        )
        .round(3)
    )
    solver_stats.to_excel(writer, sheet_name="Solver_Stats")

    # Сводная статистика
    summary = []
    for dataset in ["TRAIN", "TEST", "ALL"]:
        if dataset == "ALL":
            subset = df_final
        else:
            subset = df_final[df_final["dataset"] == dataset]

        summary.append(
            {
                "dataset": dataset,
                "n_problems": len(subset),
                "avg_time_default": subset["time_default"].mean(),
                "avg_time_best": subset["time_best"].mean(),
                "avg_speedup": subset["speedup"].mean(),
                "max_speedup": subset["speedup"].max(),
                "min_speedup": subset["speedup"].min(),
            }
        )

    df_summary = pd.DataFrame(summary)
    df_summary.to_excel(writer, sheet_name="Summary", index=False)

# === СОХРАНЕНИЕ КОНФИГУРАЦИИ ===
with open(CONFIG_FILE, "w") as f:
    json.dump(best_config, f, indent=2)

# === ВИЗУАЛИЗАЦИЯ ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# График 1: Распределение времён по солверам
ax1 = axes[0, 0]
solver_data = []
solver_labels = []
for solver in df_grid["solver"].unique():
    solver_subset = df_grid[df_grid["solver"] == solver]["avg_time"]
    solver_data.append(solver_subset)
    solver_labels.append(f"{solver}\n(n={len(solver_subset)})")

ax1.boxplot(solver_data, labels=solver_labels)
ax1.set_ylabel("Avg time (sec)")
ax1.set_title("Time Distribution by Solver")
ax1.grid(axis="y", alpha=0.3)

# График 2: Лучшие 50 конфигураций
ax2 = axes[0, 1]
top50 = df_grid.head(50)
colors = [
    {
        "simplex": "blue",
        "ipm": "red",
        "ipx": "green",
        "hipo": "orange",
        "pdlp": "purple",
    }[s]
    for s in top50["solver"]
]
ax2.bar(range(len(top50)), top50["avg_time"], color=colors, alpha=0.7)
ax2.set_xlabel("Config rank")
ax2.set_ylabel("Avg time (sec)")
ax2.set_title("Top 50 Configurations")
ax2.grid(axis="y", alpha=0.3)

# График 3: Speedup distribution
ax3 = axes[1, 0]
speedups_train = df_final[df_final["dataset"] == "TRAIN"]["speedup"].dropna()
speedups_test = df_final[df_final["dataset"] == "TEST"]["speedup"].dropna()

ax3.hist(
    speedups_train,
    bins=30,
    alpha=0.6,
    label=f"Train (mean={speedups_train.mean():.2f}x)",
    color="blue",
)
ax3.hist(
    speedups_test,
    bins=30,
    alpha=0.6,
    label=f"Test (mean={speedups_test.mean():.2f}x)",
    color="red",
)
ax3.axvline(1.0, color="black", linestyle="--", linewidth=2, label="No speedup")
ax3.set_xlabel("Speedup (x)")
ax3.set_ylabel("Count")
ax3.set_title("Speedup Distribution (Best vs Default)")
ax3.legend()
ax3.grid(alpha=0.3)

# График 4: Параметры лучшей конфигурации
ax4 = axes[1, 1]
ax4.axis("off")
best_config_text = "ЛУЧШАЯ КОНФИГУРАЦИЯ:\n\n"
best_config_text += f"Средее время: {best_time:.3f}s\n"
best_config_text += f"Решено задач: {df_grid.iloc[0]['solved']}/{len(train_files)}\n\n"
best_config_text += "Параметры:\n"
for key, val in sorted(best_config.items()):
    best_config_text += f"  {key}: {val}\n"

ax4.text(
    0.1,
    0.5,
    best_config_text,
    fontsize=11,
    family="monospace",
    verticalalignment="center",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.savefig("netlib_gridsearch_analysis.png", dpi=150)

print("\n" + "=" * 70)
print("ГОТОВО!")
print("=" * 70)
print(f"Результаты: {RESULT_FILE}")
print(f"Конфигурация: {CONFIG_FILE}")
print(f"График: netlib_gridsearch_analysis.png")
print(f"\nПроверено конфигураций: {len(all_configs)}")
print(f"Лучшая конфигурация: {best_config}")
