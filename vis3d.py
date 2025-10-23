import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import cvxpy as cvx
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def visualize_lp3d_polytope(A, b, c, x_lower, x_upper, x_star=None,
                            solve=False, solver=cvx.HIGHS,
                            n_points=100000, title=None):
    """
    Визуализация 3D ЛП как многогранника допустимой области:
      - строит выпуклую оболочку (polytope)
      - включает x_star в облако для точного включения
      - проверяет реальную допустимость A x_star <= b
    """
    if sp.issparse(A):
        A = A.toarray()

    A = np.asarray(A)
    b = np.asarray(b).ravel()
    c = np.asarray(c).ravel()
    x_lower = np.asarray(x_lower).ravel()
    x_upper = np.asarray(x_upper).ravel()

    # --- 1. Сэмплируем пространство ---
    pts = np.random.uniform(x_lower, x_upper, size=(n_points, 3))
    feas_mask = np.all((A @ pts.T).T <= b + 1e-9, axis=1)
    pts_feas = pts[feas_mask]

    if x_star is not None:
        pts_feas = np.vstack([pts_feas, x_star])  # добавляем x_star в множество
    if len(pts_feas) < 4:
        print("⚠️ Недостаточно допустимых точек для построения многогранника.")
        return None

    # --- 2. Построение выпуклой оболочки ---
    hull = ConvexHull(pts_feas)

    # --- 3. Решение задачи (если нужно) ---
    x_sol = None
    if solve:
        x_var = cvx.Variable(shape=(3, 1))
        obj = cvx.Minimize(c.reshape(1, 3) @ x_var)
        cons = [A @ x_var <= b.reshape(-1, 1),
                x_var >= x_lower.reshape(-1, 1),
                x_var <= x_upper.reshape(-1, 1)]
        prob = cvx.Problem(obj, cons)
        prob.solve(solver=solver)
        if x_var.value is not None:
            x_sol = x_var.value.ravel()

    # --- 4. Проверка допустимости x_star ---
    inside_flag = None
    if x_star is not None:
        inside_flag = bool(np.all(A @ x_star <= b + 1e-9))
        print(f"x* допустима: {inside_flag}")
        if not inside_flag:
            max_violation = np.max(A @ x_star - b)
            print(f"🔺 Нарушение ограничения: {max_violation:.3e}")

    # --- 5. Визуализация ---
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # многогранник
    faces = [pts_feas[simplex] for simplex in hull.simplices]
    poly = Poly3DCollection(faces, alpha=0.25, facecolor='tab:blue',
                            edgecolor='k', linewidths=0.3)
    ax.add_collection3d(poly)

    # направление -c
    norm_c = c / (np.linalg.norm(c) + 1e-9)
    center = np.mean(pts_feas, axis=0)
    ax.quiver(center[0], center[1], center[2],
              -norm_c[0], -norm_c[1], -norm_c[2],
              length=np.linalg.norm(x_upper - x_lower) / 3,
              color='red', linewidth=2, label='направление -c')

    # x_star
    if x_star is not None:
        xs = np.asarray(x_star).ravel()
        color = 'green' if inside_flag else 'crimson'
        label = 'x* (из lpgen, допустима)' if inside_flag else 'x* (из lpgen, вне области)'
        ax.scatter(xs[0], xs[1], xs[2],
                   s=80, color=color, marker='o', label=label)

    # найденное решение
    if x_sol is not None:
        ax.scatter(x_sol[0], x_sol[1], x_sol[2],
                   s=90, color='orange', marker='x', label='x (решатель)')

    # оформление
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('x₃')
    ax.set_xlim(x_lower[0], x_upper[0])
    ax.set_ylim(x_lower[1], x_upper[1])
    ax.set_zlim(x_lower[2], x_upper[2])
    ax.set_box_aspect([1, 1, 1])
    if title:
        ax.set_title(title)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    return x_sol


if __name__ == "__main__":
    from utils import lpgen

    A, b, c, (x_lower, x_upper), x_star, lambda_star = lpgen(
        n=3,
        m=7,
        delta=10.0,
        a_min=-10.0,
        a_max=10.0,
        lam_min=1.0,
        lam_max=9.0,
        density=0.8,
        save_to=None,
    )

    visualize_lp3d_polytope(A, b, c, x_lower, x_upper, x_star=x_star,
                            solve=True, solver=cvx.HIGHS,
                            n_points=120000,
                            title="3D LP: итоговый многогранник и проверка допустимости x*")
