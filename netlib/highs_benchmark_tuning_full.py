import os
import time
import json
import random
import pandas as pd
from highspy import Highs

# === Настройки ===
DATA_DIR = "benchmarks"
RESULT_FILE = "highs_benchmark_results.xlsx"
SEED = 42
random.seed(SEED)

param_space = {
    "presolve": ["on", "off"],
    "parallel": ["on", "off"],
    "scaling": ["on", "off"],
}

solver_variants = [
    {"solver": "ipm"},
    {"solver": "simplex", "simplex_strategy": 0},
    {"solver": "simplex", "simplex_strategy": 1},
    {"solver": "simplex", "simplex_strategy": 2},
    {"solver": "simplex", "simplex_strategy": 3},
]

best_params = {
    "solver": "simplex",
    "simplex_strategy": 1,
    "presolve": "on",
    "parallel": "off",
    "scaling": "on",
}

# === Получаем список задач ===
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".mps")])
random.shuffle(files)

n_train = int(len(files) * 0.7)
train_files = files[:n_train]
test_files = files[n_train:]

print(f"📘 Всего задач: {len(files)}")
print(f"🧩 Обучение: {len(train_files)}, Тест: {len(test_files)}")


# === Универсальная функция решения ===
def solve_problem(path, params):
    h = Highs()
    h.setOptionValue("output_flag", False)
    for k, v in params.items():
        try:
            h.setOptionValue(k, v)
        except Exception:
            pass

    try:
        h.readModel(path)
        t0 = time.time()
        h.run()
        elapsed = time.time() - t0
        return {
            "status": str(h.getModelStatus()),
            "objective": h.getObjectiveValue(),
            "rows": h.getNumRow(),
            "cols": h.getNumCol(),
            "nonzeros": h.getNumNz(),
            "time_sec": round(elapsed, 3),
        }
    except Exception as e:
        return {
            "status": f"error: {e.__class__.__name__}",
            "objective": None,
            "rows": None,
            "cols": None,
            "nonzeros": None,
            "time_sec": None,
        }


# === Функция оценки параметров на множестве задач ===
def evaluate_config(cfg, subset, note=""):
    total_time = 0
    count = 0
    for fname in subset:
        path = os.path.join(DATA_DIR, fname)
        result = solve_problem(path, cfg)
        if result["time_sec"] is not None:
            total_time += result["time_sec"]
            count += 1
    avg_time = total_time / max(count, 1)
    print(f"  {note} ⏱ avg={avg_time:.3f}s")
    return avg_time


# === BASELINE ===
print("\n=== BASELINE ===")
base_time = evaluate_config(best_params, train_files, "baseline")

# === SOLVER TUNING ===
best_solver_cfg = best_params.copy()
best_solver_time = base_time
print("\n=== SOLVER TUNING ===")
for s in solver_variants:
    test_cfg = best_params.copy()
    test_cfg.update(s)
    t = evaluate_config(test_cfg, train_files, f"{test_cfg}")
    if t < best_solver_time:
        best_solver_time = t
        best_solver_cfg = test_cfg.copy()

best_params = best_solver_cfg.copy()
base_time = best_solver_time

# === PARAMETER TUNING ===
print("\n=== PARAMETER TUNING ===")
for p, values in param_space.items():
    best_choice = best_params[p]
    best_local = base_time
    for val in values:
        test_cfg = best_params.copy()
        test_cfg[p] = val
        t = evaluate_config(test_cfg, train_files, f"{p}={val}")
        if t < best_local:
            best_local = t
            best_choice = val
    best_params[p] = best_choice
    base_time = best_local

print("\n🏁 Подбор параметров завершён")
print("Лучшие параметры:", best_params)
print(f"Среднее время на обучении: {base_time:.3f}s")

# === Решаем ВСЕ задачи с baseline и с лучшими параметрами ===
print("\n=== Финальное тестирование всех задач ===")
records = []

for fname in files:
    path = os.path.join(DATA_DIR, fname)
    is_test = fname in test_files

    # решение с параметрами по умолчанию
    res_default = solve_problem(
        path,
        {
            "solver": "simplex",
            "simplex_strategy": 1,
            "presolve": "on",
            "parallel": "off",
            "scaling": "on",
        },
    )
    res_tuned = solve_problem(path, best_params)

    records.append(
        {
            "problem": fname,
            "is_test": is_test,
            "rows": res_default["rows"],
            "cols": res_default["cols"],
            "nonzeros": res_default["nonzeros"],
            "status_default": res_default["status"],
            "time_default": res_default["time_sec"],
            "status_tuned": res_tuned["status"],
            "time_tuned": res_tuned["time_sec"],
            "objective_default": res_default["objective"],
            "objective_tuned": res_tuned["objective"],
            "speedup": (
                (res_default["time_sec"] / res_tuned["time_sec"])
                if res_default["time_sec"] and res_tuned["time_sec"]
                else None
            ),
        }
    )

# === Итоговая таблица ===
df = pd.DataFrame(records)
df["density"] = df.apply(
    lambda r: (
        r["nonzeros"] / (r["rows"] * r["cols"]) if r["rows"] and r["cols"] else None
    ),
    axis=1,
)
df = df.sort_values(["rows", "cols"])
df.to_excel(RESULT_FILE, index=False)

# === Сохраняем конфигурацию ===
with open("best_config.json", "w") as f:
    json.dump(best_params, f, indent=2)

print(f"\n✅ Результаты сохранены в {RESULT_FILE}")
print("📝 Лучшая конфигурация сохранена в best_config.json")
