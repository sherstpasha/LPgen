import os
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from highspy import Highs

# === НАСТРОЙКИ ===
DATA_DIR = "benchmarks"
RESULT_FILE = "netlib_optimization_results.xlsx"
CONFIG_FILE = "netlib_best_config.json"
SEED = 42
random.seed(SEED)

# === РАСШИРЕННОЕ ПРОСТРАНСТВО ПАРАМЕТРОВ ===

# Общие параметры (для всех солверов)
general_params = {
    "presolve": ["off", "choose", "on"],  # 3 варианта
    "parallel": ["off", "choose", "on"],  # 3 варианта
}

# Параметры симплекс-метода
simplex_params = {
    "simplex_strategy": [0, 1, 2, 3, 4],  # 5 вариантов
    "simplex_scale_strategy": [0, 1, 2, 3, 4],  # 5 вариантов
    "simplex_dual_edge_weight_strategy": [-1, 0, 1, 2],  # 4 варианта
    "simplex_primal_edge_weight_strategy": [-1, 0, 1, 2],  # 4 варианта
}

# Параметры IPM
ipm_params = {
    "run_crossover": ["off", "choose", "on"],  # 3 варианта
}

# Базовые варианты солверов
solver_variants = [
    {"solver": "simplex"},
    {"solver": "ipm"},
    {"solver": "ipx"},
    {"solver": "hipo"},
    {"solver": "pdlp"},
]

# === ДЕФОЛТНЫЕ ПАРАМЕТРЫ ===
default_params = {
    "solver": "simplex",
    "simplex_strategy": 1,
    "simplex_scale_strategy": 2,
    "simplex_dual_edge_weight_strategy": -1,
    "simplex_primal_edge_weight_strategy": -1,
    "presolve": "choose",
    "parallel": "choose",
    "run_crossover": "on",
}

# === СТАТИСТИКА КОМБИНАЦИЙ ===
# Simplex: 5 * 5 * 4 * 4 = 400 комбинаций параметров
# IPM/IPX/HiPO/PDLP: 3 варианта run_crossover (для IPM/IPX)
# Общие: 3 * 3 = 9 комбинаций (presolve * parallel)
# ИТОГО для simplex: 400 * 9 = 3,600 комбинаций
# ИТОГО для других: 5 * 9 * 3 ≈ 135 комбинаций
# ОБЩАЯ ОЦЕНКА: ~3,735 комбинаций для полного перебора

best_params = default_params.copy()

# === ЗАГРУЗКА И РАЗДЕЛЕНИЕ ЗАДАЧ ===
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".mps")])
random.shuffle(files)

n_train = int(len(files) * 0.7)
train_files = set(files[:n_train])
test_files = set(files[n_train:])

print("=" * 70)
print("ОПТИМИЗАЦИЯ ПАРАМЕТРОВ HIGHS НА NETLIB")
print("=" * 70)
print(f"Всего задач: {len(files)}")
print(f"Обучающая выборка: {len(train_files)} задач (70%)")
print(f"Тестовая выборка: {len(test_files)} задач (30%)")
print(f"Обучающие задачи: {sorted(train_files)[:5]}... (первые 5)")
print(f"Тестовые задачи: {sorted(test_files)[:5]}... (первые 5)")
print("=" * 70)


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
            "objective": h.getObjectiveValue(),
            "time_sec": elapsed,
            "success": True,
        }
    except Exception as e:
        return {
            "status": f"error: {e.__class__.__name__}",
            "objective": None,
            "time_sec": None,
            "success": False,
        }


# === ФУНКЦИЯ ОЦЕНКИ КОНФИГУРАЦИИ ===
def evaluate_config(cfg, file_subset, note=""):
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
    print(
        f"  {note:50s} | avg={avg_time:7.3f}s | solved={count}/{len(file_subset)} | failed={failed}"
    )

    return avg_time, count, failed


# === ИСТОРИЯ РЕЗУЛЬТАТОВ ===
tuning_history = []

# === BASELINE ===
print("\n" + "=" * 70)
print("ЭТАП 1: BASELINE (дефолтные параметры)")
print("=" * 70)
base_time, base_solved, base_failed = evaluate_config(
    default_params, train_files, "Дефолтные параметры"
)
tuning_history.append(
    {
        "config": json.dumps(default_params),
        "note": "baseline",
        "avg_time": base_time,
        "solved": base_solved,
        "failed": base_failed,
    }
)

# === ЭТАП 2: ПОДБОР СОЛВЕРА ===
print("\n" + "=" * 70)
print("ЭТАП 2: ПОДБОР СОЛВЕРА")
print("=" * 70)

best_solver_cfg = best_params.copy()
best_solver_time = base_time

for s in solver_variants:
    test_cfg = default_params.copy()
    test_cfg.update(s)

    note = f"solver={test_cfg['solver']}"
    t, solved, failed = evaluate_config(test_cfg, train_files, note)

    tuning_history.append(
        {
            "config": json.dumps(test_cfg),
            "note": note,
            "avg_time": t,
            "solved": solved,
            "failed": failed,
        }
    )

    if t < best_solver_time:
        best_solver_time = t
        best_solver_cfg = test_cfg.copy()
        print(f"  >>> НОВЫЙ ЛУЧШИЙ: {note} ({t:.3f}s)")

best_params = best_solver_cfg.copy()
base_time = best_solver_time

print(f"\nЛучший солвер: {best_params}")
print(f"Среднее время: {base_time:.3f}s")

# === ЭТАП 3: ПОДБОР ПАРАМЕТРОВ SIMPLEX (если выбран) ===
if best_params.get("solver") == "simplex":
    print("\n" + "=" * 70)
    print("ЭТАП 3: ПОДБОР ПАРАМЕТРОВ SIMPLEX")
    print("=" * 70)

    for param_name, param_values in simplex_params.items():
        print(f"\nПодбор параметра: {param_name}")
        print("-" * 70)

        original = best_params.get(param_name, param_values[0])
        best_local = base_time
        best_choice = original

        for val in param_values:
            test_cfg = best_params.copy()
            test_cfg[param_name] = val

            note = f"{param_name}={val}"
            t, solved, failed = evaluate_config(test_cfg, train_files, note)

            tuning_history.append(
                {
                    "config": json.dumps(test_cfg),
                    "note": note,
                    "avg_time": t,
                    "solved": solved,
                    "failed": failed,
                }
            )

            if t < best_local:
                best_local = t
                best_choice = val
                print(f"  >>> УЛУЧШЕНИЕ: {param_name}={val} ({t:.3f}s)")

        if best_choice != original:
            best_params[param_name] = best_choice
            base_time = best_local
            print(f"  Выбран: {param_name}={best_choice}")
        else:
            print(f"  Оставлен: {param_name}={original}")

# === ЭТАП 4: ПОДБОР ПАРАМЕТРОВ IPM (если выбран ipm или ipx) ===
elif best_params.get("solver") in ["ipm", "ipx"]:
    print("\n" + "=" * 70)
    print("ЭТАП 3: ПОДБОР ПАРАМЕТРОВ IPM")
    print("=" * 70)

    for param_name, param_values in ipm_params.items():
        print(f"\nПодбор параметра: {param_name}")
        print("-" * 70)

        original = best_params.get(param_name, param_values[0])
        best_local = base_time
        best_choice = original

        for val in param_values:
            test_cfg = best_params.copy()
            test_cfg[param_name] = val

            note = f"{param_name}={val}"
            t, solved, failed = evaluate_config(test_cfg, train_files, note)

            tuning_history.append(
                {
                    "config": json.dumps(test_cfg),
                    "note": note,
                    "avg_time": t,
                    "solved": solved,
                    "failed": failed,
                }
            )

            if t < best_local:
                best_local = t
                best_choice = val
                print(f"  >>> УЛУЧШЕНИЕ: {param_name}={val} ({t:.3f}s)")

        if best_choice != original:
            best_params[param_name] = best_choice
            base_time = best_local
            print(f"  Выбран: {param_name}={best_choice}")
        else:
            print(f"  Оставлен: {param_name}={original}")

# === ЭТАП 5: ПОДБОР ОБЩИХ ПАРАМЕТРОВ ===
print("\n" + "=" * 70)
print("ЭТАП 4: ПОДБОР ОБЩИХ ПАРАМЕТРОВ")
print("=" * 70)

for param_name, param_values in general_params.items():
    print(f"\nПодбор параметра: {param_name}")
    print("-" * 70)

    original = best_params.get(param_name, param_values[0])
    best_local = base_time
    best_choice = original

    for val in param_values:
        test_cfg = best_params.copy()
        test_cfg[param_name] = val

        note = f"{param_name}={val}"
        t, solved, failed = evaluate_config(test_cfg, train_files, note)

        tuning_history.append(
            {
                "config": json.dumps(test_cfg),
                "note": note,
                "avg_time": t,
                "solved": solved,
                "failed": failed,
            }
        )

        if t < best_local:
            best_local = t
            best_choice = val
            print(f"  >>> УЛУЧШЕНИЕ: {param_name}={val} ({t:.3f}s)")

    if best_choice != original:
        best_params[param_name] = best_choice
        base_time = best_local
        print(f"  Выбран: {param_name}={best_choice}")
    else:
        print(f"  Оставлен: {param_name}={original}")

print("\n" + "=" * 70)
print("ПОДБОР ПАРАМЕТРОВ ЗАВЕРШЁН")
print("=" * 70)
print("Лучшие параметры:", best_params)
print(f"Среднее время на обучающей выборке: {base_time:.3f}s")

# === ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ ===
print("\n" + "=" * 70)
print("ЭТАП 4: ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ НА ВСЕХ ЗАДАЧАХ")
print("=" * 70)

final_results = []

for fname in sorted(files):
    path = os.path.join(DATA_DIR, fname)
    is_test = fname in test_files

    # Решение с дефолтными параметрами
    res_default = solve_problem(path, default_params)

    # Решение с подобранными параметрами
    res_tuned = solve_problem(path, best_params)

    # Информация о задаче
    h_info = Highs()
    h_info.setOptionValue("output_flag", False)
    h_info.readModel(path)

    speedup = None
    if res_default["time_sec"] and res_tuned["time_sec"]:
        speedup = res_default["time_sec"] / res_tuned["time_sec"]

    final_results.append(
        {
            "problem": fname,
            "dataset": "TEST" if is_test else "TRAIN",
            "rows": h_info.getNumRow(),
            "cols": h_info.getNumCol(),
            "nonzeros": h_info.getNumNz(),
            "status_default": res_default["status"],
            "time_default": res_default["time_sec"],
            "status_tuned": res_tuned["status"],
            "time_tuned": res_tuned["time_sec"],
            "objective_default": res_default["objective"],
            "objective_tuned": res_tuned["objective"],
            "speedup": speedup,
        }
    )

    status_symbol = "✓" if res_tuned["success"] else "✗"
    speedup_str = f"{speedup:.2f}x" if speedup else "—"
    print(
        f"  {status_symbol} {fname:20s} | {'TEST' if is_test else 'TRAIN':5s} | "
        f"default={res_default['time_sec'] or 0:6.3f}s | "
        f"tuned={res_tuned['time_sec'] or 0:6.3f}s | speedup={speedup_str:6s}"
    )

# === СОЗДАНИЕ DATAFRAME ===
df = pd.DataFrame(final_results)
df["density"] = df.apply(
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
    subset = df[df["dataset"] == dataset]
    avg_default = subset["time_default"].mean()
    avg_tuned = subset["time_tuned"].mean()
    avg_speedup = subset["speedup"].mean()

    print(f"\n{dataset} выборка ({len(subset)} задач):")
    print(f"  Среднее время (дефолт): {avg_default:.3f}s")
    print(f"  Среднее время (подобранные): {avg_tuned:.3f}s")
    print(f"  Среднее ускорение: {avg_speedup:.2f}x")

total_avg_default = df["time_default"].mean()
total_avg_tuned = df["time_tuned"].mean()
total_avg_speedup = df["speedup"].mean()

print(f"\nВСЕГО ({len(df)} задач):")
print(f"  Среднее время (дефолт): {total_avg_default:.3f}s")
print(f"  Среднее время (подобранные): {total_avg_tuned:.3f}s")
print(f"  Среднее ускорение: {total_avg_speedup:.2f}x")

# === СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
with pd.ExcelWriter(RESULT_FILE, engine="openpyxl") as writer:
    # Основная таблица
    df_sorted = df.sort_values(["dataset", "rows", "cols"])
    df_sorted.to_excel(writer, sheet_name="Results", index=False)

    # История подбора параметров
    df_history = pd.DataFrame(tuning_history)
    df_history.to_excel(writer, sheet_name="Tuning_History", index=False)

    # Сводная статистика
    summary = []
    for dataset in ["TRAIN", "TEST", "ALL"]:
        if dataset == "ALL":
            subset = df
        else:
            subset = df[df["dataset"] == dataset]

        summary.append(
            {
                "dataset": dataset,
                "n_problems": len(subset),
                "avg_time_default": subset["time_default"].mean(),
                "avg_time_tuned": subset["time_tuned"].mean(),
                "avg_speedup": subset["speedup"].mean(),
                "max_speedup": subset["speedup"].max(),
                "min_speedup": subset["speedup"].min(),
            }
        )

    df_summary = pd.DataFrame(summary)
    df_summary.to_excel(writer, sheet_name="Summary", index=False)

# === СОХРАНЕНИЕ КОНФИГУРАЦИИ ===
with open(CONFIG_FILE, "w") as f:
    json.dump(best_params, f, indent=2)

# === ВИЗУАЛИЗАЦИЯ ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Сравнение времён
ax1 = axes[0]
train_df = df[df["dataset"] == "TRAIN"].sort_values("time_default")
test_df = df[df["dataset"] == "TEST"].sort_values("time_default")

x_train = range(len(train_df))
x_test = range(len(train_df), len(train_df) + len(test_df))

ax1.bar(
    x_train, train_df["time_default"], alpha=0.5, label="Train (default)", color="blue"
)
ax1.bar(
    x_train, train_df["time_tuned"], alpha=0.5, label="Train (tuned)", color="lightblue"
)
ax1.bar(x_test, test_df["time_default"], alpha=0.5, label="Test (default)", color="red")
ax1.bar(
    x_test, test_df["time_tuned"], alpha=0.5, label="Test (tuned)", color="lightcoral"
)

ax1.set_xlabel("Problem index")
ax1.set_ylabel("Time (sec)")
ax1.set_title("Default vs Tuned Parameters")
ax1.legend()
ax1.axvline(len(train_df) - 0.5, color="black", linestyle="--", linewidth=1)

# График 2: Распределение ускорений
ax2 = axes[1]
speedups_train = df[df["dataset"] == "TRAIN"]["speedup"].dropna()
speedups_test = df[df["dataset"] == "TEST"]["speedup"].dropna()

ax2.hist(
    speedups_train,
    bins=20,
    alpha=0.6,
    label=f"Train (mean={speedups_train.mean():.2f}x)",
    color="blue",
)
ax2.hist(
    speedups_test,
    bins=20,
    alpha=0.6,
    label=f"Test (mean={speedups_test.mean():.2f}x)",
    color="red",
)
ax2.axvline(1.0, color="black", linestyle="--", linewidth=1, label="No speedup")
ax2.set_xlabel("Speedup (x)")
ax2.set_ylabel("Count")
ax2.set_title("Speedup Distribution")
ax2.legend()

plt.tight_layout()
plt.savefig("netlib_optimization_comparison.png", dpi=150)
print(f"\nГрафик сохранён: netlib_optimization_comparison.png")

print("\n" + "=" * 70)
print("ГОТОВО!")
print("=" * 70)
print(f"Результаты: {RESULT_FILE}")
print(f"Конфигурация: {CONFIG_FILE}")
print(f"График: netlib_optimization_comparison.png")
