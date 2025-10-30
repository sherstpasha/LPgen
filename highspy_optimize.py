import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
import highspy
from utils import lpgen

SIZES = [2000]
REPEATS = 1
density = 0.01
seed = 42

param_space = {
    "presolve": ["on", "off"],
    "parallel": ["on", "off"],
    "scaling": ["on", "off"],
}

best_params = {
    "solver": "simplex",
    "simplex_strategy": 1,
    "presolve": "on",
    "parallel": "off",
    "scaling": "on",
}

solver_variants = [
    {"solver": "ipm"},
    {"solver": "simplex", "simplex_strategy": 0},
    {"solver": "simplex", "simplex_strategy": 1},
    {"solver": "simplex", "simplex_strategy": 2},
    {"solver": "simplex", "simplex_strategy": 3},
]

results = []

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

print("=== BASELINE ===")
base_time = evaluate_config(best_params, "baseline")
print("baseline =", base_time)

print("\n=== SOLVER TUNING ===")
best_solver_cfg = best_params.copy()
best_solver_time = base_time

for s in solver_variants:
    test_cfg = best_params.copy()
    for k, v in s.items():
        test_cfg[k] = v
    t = evaluate_config(test_cfg, f"solver={test_cfg['solver']} strategy={test_cfg.get('simplex_strategy','-')}")
    print(test_cfg, " => ", t)
    if t < best_solver_time:
        best_solver_time = t
        best_solver_cfg = test_cfg.copy()

best_params = best_solver_cfg.copy()
base_time = best_solver_time
print("best solver =", best_params, "time=", base_time)

print("\n=== PARAMETER TUNING ===")
for p in param_space:
    original = best_params[p]
    best_local = base_time
    best_choice = original
    for val in param_space[p]:
        test_cfg = best_params.copy()
        test_cfg[p] = val
        t = evaluate_config(test_cfg, f"test {p}={val}")
        print(test_cfg, " => ", t)
        if t < best_local:
            best_local = t
            best_choice = val
    if best_choice != original:
        best_params[p] = best_choice
        base_time = best_local
    else:
        pass

print("\n=== FINISHED ===")
print("best parameters:", best_params)
print("final time:", base_time)

df = pd.DataFrame(results)
df.to_csv("results_table.csv", index=False)
df.to_excel("results_table.xlsx", index=False)

with open("best_config.json", "w") as f:
    json.dump(best_params, f, indent=2)

df_sorted = df.sort_values("time_2000")
plt.figure(figsize=(9,4))
plt.bar(range(len(df_sorted)), df_sorted["time_2000"])
plt.ylabel("time (sec)")
plt.title("solver search")
plt.tight_layout()
plt.savefig("speed_comparison_2000.png")
