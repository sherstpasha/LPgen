import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import cvxpy as cvx

# ---- здесь предполагается, что lpgen уже импортирована из utils ----
# from utils import lpgen

def visualize_lp2d(A, b, c, x_lower, x_upper, x_star=None,
                   solve=False, solver=cvx.HIGHS, grid_points=400, title=None):
    """
    Визуализация 2D ЛП:
      - заштрихованная область допустимых значений
      - линии ограничений Ai• x = b_i
      - линии уровня c^T x
      - точки x_star (если дана) и x_sol (если solve=True)

    A: scipy.sparse (m x 2) или ndarray
    b: (m,)
    c: (2,) или (1,2)
    x_lower, x_upper: (2,)
    """
    # приведение типов/форм
    if sp.issparse(A):
        A = A.tocsr()
        a1 = A[:, 0].toarray().ravel()
        a2 = A[:, 1].toarray().ravel()
    else:
        A = np.asarray(A)
        a1 = A[:, 0].ravel()
        a2 = A[:, 1].ravel()

    b = np.asarray(b).ravel()
    c = np.asarray(c).ravel()  # (2,)

    x_lower = np.asarray(x_lower).ravel()
    x_upper = np.asarray(x_upper).ravel()

    # сетка по прямоугольнику [x_lower, x_upper]
    x1 = np.linspace(x_lower[0], x_upper[0], grid_points)
    x2 = np.linspace(x_lower[1], x_upper[1], grid_points)
    X1, X2 = np.meshgrid(x1, x2)

    # проверка ограничений Ax <= b
    # создаем 3D массив: для каждого ограничения i проверяем a1_i*X1 + a2_i*X2 <= b_i
    LHS = a1[:, None, None] * X1[None, :, :] + a2[:, None, None] * X2[None, :, :]
    feas_mask = np.all(LHS <= b[:, None, None] + 1e-9, axis=0)

    # уровни целевой функции c^T x = const
    Z = c[0] * X1 + c[1] * X2
    # уровни вокруг значения в x_star, если оно известно
    if x_star is not None:
        v0 = float(c @ np.asarray(x_star).ravel())
        levels = np.linspace(v0 - 0.5 * max(abs(v0), 1.0), v0 + 0.5 * max(abs(v0), 1.0), 7)
    else:
        # иначе берем уровни по всему прямоугольнику
        zmin, zmax = np.min(Z), np.max(Z)
        levels = np.linspace(zmin, zmax, 9)

    # рисунок
    plt.figure(figsize=(7.5, 7))
    # область допустимых значений
    plt.pcolormesh(X1, X2, feas_mask, shading='auto', alpha=0.20)

    # линии ограничений Ai1*x1 + Ai2*x2 = b_i (в пределах прямоугольника)
    for i in range(len(b)):
        if abs(a2[i]) > 1e-12:
            y_line = (b[i] - a1[i] * x1) / a2[i]
            # обрезаем показ по пределам
            valid = (y_line >= x_lower[1] - 1e-9) & (y_line <= x_upper[1] + 1e-9)
            if np.any(valid):
                plt.plot(x1[valid], y_line[valid], linewidth=1.0, alpha=0.6)
        else:
            # вертикальная прямая: a1*x1 = b -> x1 = b/a1 (если a1 != 0)
            if abs(a1[i]) > 1e-12:
                x_vert = b[i] / a1[i]
                if x_lower[0] - 1e-9 <= x_vert <= x_upper[0] + 1e-9:
                    plt.plot([x_vert, x_vert], [x_lower[1], x_upper[1]], linewidth=1.0, alpha=0.6)

    # контуры целевой функции
    cs = plt.contour(X1, X2, Z, levels=levels, linewidths=0.9, linestyles='dashed')
    plt.clabel(cs, inline=True, fontsize=8, fmt="cᵀx=%.1f")

    # отметим x_star (если дан)
    if x_star is not None:
        xs = np.asarray(x_star).ravel()
        plt.scatter([xs[0]], [xs[1]], s=60, marker='o', label='x* (из lpgen)')

    # при необходимости решим задачу и отметим найденное решение
    x_sol = None
    if solve:
        x_var = cvx.Variable(shape=(2, 1))
        # Во избежание проблем с формой b/x_bounds приводим к столбцам
        b_vec = b.reshape(-1, 1)
        xl = x_lower.reshape(-1, 1)
        xu = x_upper.reshape(-1, 1)
        # A может быть sparse; CVXPY это поддерживает
        obj = cvx.Minimize(c.reshape(1, 2) @ x_var)
        cons = [A @ x_var <= b_vec, x_var >= xl, x_var <= xu]
        prob = cvx.Problem(obj, cons)
        prob.solve(solver=solver)
        if x_var.value is not None:
            x_sol = x_var.value.ravel()
            plt.scatter([x_sol[0]], [x_sol[1]], s=60, marker='x', label='x (решатель)')

    plt.xlim(x_lower[0], x_upper[0])
    plt.ylim(x_lower[1], x_upper[1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    if title:
        plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    return x_sol

if __name__ == "__main__":
    # Пример: генерируем 2D-задачу через lpgen и визуализируем
    from utils import lpgen  # убедись, что lpgen доступна

    # Параметры генерации: 2 переменные, 5 ограничений
    A, b, c, (x_lower, x_upper), x_star, lambda_star = lpgen(
        n=2,
        m=5,
        delta=30.0,
        a_min=-20.0,
        a_max=20.0,
        lam_min=1.0,
        lam_max=9.0,
        density=0.6,   # для 2D можно брать повыше, чтобы прямые были разнообразнее
        seed=42,
        save_to=None,
    )

    # Визуализация и решение (HiGHS); можно поставить solve=False, чтобы только рисовать
    visualize_lp2d(A, b, c, x_lower, x_upper, x_star=x_star,
                   solve=True, solver=cvx.HIGHS,
                   grid_points=500,
                   title="2D LP: область допустимых значений, ограничения и изолинии цели")
