import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
import highspy
from utils import lpgen

# ======================
# CONFIG
# ======================
SIZES = [2000]
REPEATS = 1
density = 0.01
seed = 42

param_space = {
    "solver": ["simplex", "ipm"],
    "presolve": ["on", "off"],
    "simplex_strategy": [0, 1, 2, 3],
    "parallel": ["on", "off"],
    "scaling": ["on", "off"],
}

best_params = {
    "solver": "simplex",
    "presolve": "on",
    "simplex_strategy": 1,
    "parallel": "off",
    "scaling": "on"
}

results = []

# ======================
# RUN SOLVER
# ======================
def solve_with_params(n, params):
    A, b, c, (x_lower, x_upper), *_ = lpgen(
        n=n, m=n, density=density, seed=seed
    )

    A = A.tocsc()
    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = n

    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)

    lp.col_cost_ = c.astype(np.float64)
    lp.col_lower_ = np.where(np.isfinite(x_lower), x_lower, -highspy.kHighsInf)
    lp.col_upper_ = np.where(np.isfinite(x_upper), x_upper, highspy.kHighsInf)

    lp.row_lower_ = np.full(n, -highspy.kHighsInf, dtype=np.float64)
    lp.row_upper_ = b.astype(np.float64)

    model = highspy.Highs()
    model.setOptionValue("output_flag", False)
    for k, v in params.items():
        model.setOptionValue(k, v)

    model.passModel(lp)

    t0 = time.monotonic()
    try:
        model.run()
        return time.monotonic() - t0, str(model.getModelStatus())
    except:
        return None, "error"


# ======================
# TEST CONFIG ON ALL SIZES
# ======================
def evaluate_config(cfg, note=""):
    rec = {"config": json.dumps(cfg), "note": note}
    ok = True

    for n in SIZES:
        t, st = solve_with_params(n, cfg)
        rec[f"time_{n}"] = t
        rec[f"status_{n}"] = st

        if st not in ("1", "kOptimal"):
            ok = False

    rec["all_optimal"] = ok
    results.append(rec)

    return rec["time_2000"]


# ======================
# BASELINE
# ======================
print("=== BASELINE ===")
base_time = evaluate_config(best_params, "baseline")
print("baseline =>", base_time)


# ======================
# COORDINATE TUNING
# ======================
print("\n=== COORDINATE TUNING ===")

for p in param_space:

    # пропуск simplex_strategy если solver – ipm
    if p == "simplex_strategy" and best_params["solver"] == "ipm":
        print(f"\n--- Skipping {p} (solver=ipm) ---")
        continue

    print(f"\n--- Optimizing {p} ---")
    original_value = best_params[p]
    best_local_time = base_time
    best_local_value = original_value

    for val in param_space[p]:

        # если тестируем simplex_strategy, убедимся что solver=simplex
        if p == "simplex_strategy" and best_params["solver"] != "simplex":
            continue

        test_cfg = best_params.copy()
        test_cfg[p] = val

        t = evaluate_config(test_cfg, f"test {p}={val}")
        print(f"  {p}={val} → {t:.3f} c")

        if t < best_local_time:
            best_local_time = t
            best_local_value = val

    if best_local_value != original_value:
        print(f"{p}: {original_value} → {best_local_value}, выигрыш {base_time:.3f} → {best_local_time:.3f}")
        best_params[p] = best_local_value
        base_time = best_local_time
    else:
        print(f"Улучшений для {p} нет")


# ======================
# SAVE RESULTS
# ======================
print("\n=== FINISHED ===")
print("Лучшие параметры:", best_params)
print("Финальное время на n=2000:", base_time)

df = pd.DataFrame(results)
df.to_csv("results_table.csv", index=False)
df.to_excel("results_table.xlsx", index=False)
print("Сохранено: results_table.csv, results_table.xlsx")

with open("best_config.json", "w") as f:
    json.dump(best_params, f, indent=2)

# график
df_sorted = df.sort_values("time_2000")
plt.figure(figsize=(9,4))
plt.bar(range(len(df_sorted)), df_sorted["time_2000"])
plt.ylabel("Время (сек)")
plt.title("Точки поиска (меньше — лучше)")
plt.tight_layout()
plt.savefig("speed_comparison_2000.png")
print("График: speed_comparison_2000.png")
