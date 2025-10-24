import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import cvxpy as cvx

from utils import lp_cvxp  # убедись, что lp_cvxp в том же каталоге или в PYTHONPATH


def visualize_lp2d_cvxp(
    n=2,
    m=3,
    density=1.0,
    a_min=-5.0,
    a_max=5.0,
    x_bounds=(-3.0, 3.0),
    seed=None,
    easy=True,
    grid_points=400,
    solver=cvx.HIGHS,
    title=None,
):
    """
    Визуализация 2D задачи ЛП, сгенерированной через lp_cvxp (easy или случайную).
    """

    # === Генерация задачи ===
    prob, A, b, c, x = lp_cvxp(
        n=n,
        m=m,
        density=density,
        a_min=a_min,
        a_max=a_max,
        x_bounds=x_bounds,
        seed=seed,
        easy=easy,
    )

    # Если sparse -> csr
    if sp.issparse(A):
        A = A.tocsr()
        a1 = A[:, 0].toarray().ravel()
        a2 = A[:, 1].toarray().ravel()
    else:
        A = np.asarray(A)
        a1 = A[:, 0].ravel()
        a2 = A[:, 1].ravel()

    b = np.asarray(b).ravel()
    c = np.asarray(c).ravel()
    x_lower, x_upper = np.asarray(x_bounds)

    # === Решение ===
    print("Решение задачи через CVXPY...")
    prob.solve(solver=solver)
    x_sol = x.value.ravel() if x.value is not None else None
    print("x_sol =", x_sol)
    print("Оптимум =", prob.value)

    # === Сетка ===
    x1 = np.linspace(x_lower, x_upper, grid_points)
    x2 = np.linspace(x_lower, x_upper, grid_points)
    X1, X2 = np.meshgrid(x1, x2)
    LHS = a1[:, None, None] * X1[None, :, :] + a2[:, None, None] * X2[None, :, :]
    feas_mask = np.all(LHS <= b[:, None, None] + 1e-9, axis=0)

    # === Линии ограничений ===
    plt.figure(figsize=(7.5, 7))
    plt.pcolormesh(X1, X2, feas_mask, shading="auto", alpha=0.25)

    for i in range(len(b)):
        if abs(a2[i]) > 1e-12:
            y_line = (b[i] - a1[i] * x1) / a2[i]
            valid = (y_line >= x_lower - 1e-9) & (y_line <= x_upper + 1e-9)
            if np.any(valid):
                plt.plot(
                    x1[valid],
                    y_line[valid],
                    linewidth=1.0,
                    alpha=0.6,
                    label=f"огр. {i+1}",
                )
        else:
            if abs(a1[i]) > 1e-12:
                x_vert = b[i] / a1[i]
                if x_lower - 1e-9 <= x_vert <= x_upper + 1e-9:
                    plt.plot(
                        [x_vert, x_vert],
                        [x_lower, x_upper],
                        linewidth=1.0,
                        alpha=0.6,
                        label=f"огр. {i+1}",
                    )

    # === Изолинии цели ===
    Z = c[0] * X1 + c[1] * X2
    zmin, zmax = np.min(Z), np.max(Z)
    levels = np.linspace(zmin, zmax, 9)
    cs = plt.contour(X1, X2, Z, levels=levels, linestyles="dashed", linewidths=0.9)
    plt.clabel(cs, inline=True, fontsize=8, fmt="cᵀx=%.1f")

    # === Решение ===
    if x_sol is not None:
        plt.scatter(
            [x_sol[0]], [x_sol[1]], s=60, c="red", marker="x", label="x (решение)"
        )

    plt.xlim(x_lower, x_upper)
    plt.ylim(x_lower, x_upper)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend(loc="best")
    if title:
        plt.title(title)
    else:
        plt.title("2D LP через lp_cvxp (easy = {})".format(easy))
    plt.tight_layout()
    plt.show()

    return x_sol


# === Пример использования ===
if __name__ == "__main__":
    visualize_lp2d_cvxp(
        n=2,
        m=3,
        density=1.0,
        a_min=-3.0,
        a_max=5.0,
        x_bounds=(-2.0, 2.0),
        seed=42,
        easy=True,  # <- тот самый "встроенный" случай
        title="LP-cvxp (easy=True): встроенное решение",
    )
