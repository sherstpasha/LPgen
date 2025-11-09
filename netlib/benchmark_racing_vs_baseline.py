import os
import time
import numpy as np
import pandas as pd
from highspy import Highs
from rh import RacingHighs


# === Настройки ===
DATA_DIR = "benchmarks"
RESULT_FILE = "compare_racing_all.xlsx"
SHIFT = 10  # для SGM

# Собираем список задач
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".mps")])
print(f"🔍 Найдено {len(files)} задач в {DATA_DIR}\n")

results = []


def shifted_geometric_mean(values, shift=10):
    """Shifted geometric mean как в LP/MIP бенчмарках"""
    values = np.array([v for v in values if v is not None and np.isfinite(v)])
    if len(values) == 0:
        return None
    return np.exp(np.mean(np.log(np.maximum(1, values + shift)))) - shift


# === Основной цикл ===
for fname in files:
    path = os.path.join(DATA_DIR, fname)
    print(f"\n⚙️ Задача: {fname}")

    # --- baseline ---
    h = Highs()
    h.readModel(path)
    h.setOptionValue("output_flag", False)
    start = time.time()
    try:
        h.run()
        baseline_time = time.time() - start
        baseline_obj = h.getObjectiveValue()
        status = str(h.getModelStatus())
    except Exception as e:
        baseline_time = None
        baseline_obj = None
        status = f"error: {e}"

    # --- racing ---
    try:
        solver = RacingHighs(
            model_path=path,
            warmup_fraction=0.05,  # 5% итераций — “гонка”
            configs=[
                {"solver": "simplex", "presolve": "on"},
                {"solver": "ipm", "presolve": "on"},
            ],
        )
        res = solver.run()
        racing_time = res["total_time"]
        racing_obj = res["final_obj"]
        racing_solver = res["best_cfg"]["solver"]
    except Exception as e:
        racing_time = None
        racing_obj = None
        racing_solver = f"error: {e}"

    # --- результат ---
    results.append(
        {
            "problem": fname,
            "status": status,
            "objective_baseline": baseline_obj,
            "objective_racing": racing_obj,
            "solver_racing": racing_solver,
            "time_baseline": baseline_time,
            "time_racing": racing_time,
            "speedup": (
                baseline_time / racing_time
                if (baseline_time and racing_time and racing_time > 0)
                else None
            ),
        }
    )

    # Быстрая сводка
    if baseline_time and racing_time:
        print(
            f"⏱ baseline={baseline_time:.2f}s, racing={racing_time:.2f}s "
            f"(x{baseline_time / racing_time:.2f} быстрее)"
        )
    else:
        print(
            f"⚠️ Ошибка в задаче {fname}: baseline={baseline_time}, racing={racing_time}"
        )

# === Сохранение результатов ===
df = pd.DataFrame(results)
df = df.sort_values(by="speedup", ascending=False)
df.to_excel(RESULT_FILE, index=False)

# === Геометрическое среднее ===
baseline_times = df["time_baseline"].dropna().values
racing_times = df["time_racing"].dropna().values

sgm_baseline = shifted_geometric_mean(baseline_times, shift=SHIFT)
sgm_racing = shifted_geometric_mean(racing_times, shift=SHIFT)
sgm_speedup = sgm_baseline / sgm_racing if sgm_racing else None

print("\n✅ Все результаты сохранены в", RESULT_FILE)
print(f"Среднее ускорение (обычное): {df['speedup'].dropna().mean():.2f}×")
if sgm_speedup:
    print(f"Среднее ускорение (SGM): {sgm_speedup:.2f}×")
