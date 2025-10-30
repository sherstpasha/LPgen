import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
import cvxpy as cvx
import highspy
from utils import lp_cvxp

SIZES = [500, 1000, 1500, 2000, 2500, 3000]

best_params = {
    "solver": "ipm",
    "presolve": "off",
    "simplex_strategy": 1,
    "parallel": "off",
    "scaling": "on"
}

density = 0.01
seed = 42
REPEATS = 1

results = []

def solve_highs(A, b, c, x_lower, x_upper, params):
    if sparse.issparse(A):
        A = A.tocsc()
    else:
        A = sparse.csc_matrix(A)

    c = np.array(c).reshape(-1).astype(np.float64)
    b = np.array(b).reshape(-1).astype(np.float64)

    n = A.shape[1]
    m = A.shape[0]

    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = m

    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)

    lp.col_cost_ = c
    lp.col_lower_ = np.full(n, x_lower, dtype=np.float64)
    lp.col_upper_ = np.full(n, x_upper, dtype=np.float64)

    lp.row_lower_ = np.full(m, -highspy.kHighsInf, dtype=np.float64)
    lp.row_upper_ = b

    model = highspy.Highs()
    model.setOptionValue("output_flag", False)

    for k, v in params.items():
        model.setOptionValue(k, v)

    model.passModel(lp)

    t0 = time.monotonic()
    try:
        model.run()
        return time.monotonic() - t0, str(model.getModelStatus())
    except Exception as e:
        return None, f"ERROR_{type(e).__name__}"

print("=== Запуск тестирования HiGHS на lp_cvxp (easy=True) ===")

for n in SIZES:
    prob, A, b, c, x = lp_cvxp(
        n=n, m=n,
        density=density,
        seed=seed,
        easy=True
    )

    best_t = None
    best_status = None

    for _ in range(REPEATS):
        t, st = solve_highs(A, b, c, -1000, 1000, best_params)
        if best_t is None or (t is not None and t < best_t):
            best_t, best_status = t, st

    print(f"{n}x{n}: {best_t:.3f} c, статус = {best_status}")
    results.append({"n": n, "time": best_t, "status": best_status})

df = pd.DataFrame(results)
df.to_csv("highs_cvxp_easy.csv", index=False)
df.to_excel("highs_cvxp_easy.xlsx", index=False)
print("Результаты сохранены.")

plt.figure(figsize=(9,4))
plt.plot(df["n"], df["time"], marker="o")
plt.title("HiGHS с лучшими параметрами (easy=True)")
plt.xlabel("Размер задачи n=m")
plt.ylabel("Время, сек")
plt.grid()
plt.tight_layout()
plt.savefig("highs_cvxp_easy_plot.png")
print("График сохранён.")
