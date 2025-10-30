import numpy as np
import scipy.sparse as sp
import cvxpy as cvx


def lpgen(
    n=7500,
    m=7500,
    delta=30.0,
    a_min=-20.0,
    a_max=20.0,
    lam_min=1.0,
    lam_max=9.0,
    density=0.01,
    seed=None,
    save_to=None,
):
    """
    Генерация крупной разреженной тестовой задачи ЛП.
    Возвращает или сохраняет (A, b, c, x_lower, x_upper, x_star, λ_star).
    """

    if seed is not None:
        np.random.seed(seed)

    # === Шаг 0 ===
    x_star = np.ones(n, dtype=np.float64)
    x_lower = x_star - delta
    x_upper = x_star + delta

    # === Шаг 1 ===
    # Разреженная матрица A (CSR формат)
    A = sp.random(m, n, density=density, format="csr", dtype=np.float64)
    A.data = np.random.uniform(a_min, a_max, size=A.nnz)

    # === Шаг 2 ===
    b_hat = A @ x_star

    # === Шаг 3 ===
    # b_max[i] = max{Ai·x : x_lower <= x <= x_upper}
    # Для CSR формата вычисляем построчно
    b_max = np.empty(m, dtype=np.float64)
    for i in range(m):
        row_data = A.data[A.indptr[i]:A.indptr[i + 1]]
        row_indices = A.indices[A.indptr[i]:A.indptr[i + 1]]
        x_low_i = x_lower[row_indices]
        x_up_i = x_upper[row_indices]
        # выбираем верхнюю или нижнюю границу в зависимости от знака
        b_max[i] = np.sum(np.where(row_data < 0, row_data * x_low_i, row_data * x_up_i))

    # === Шаг 4 ===
    b = np.empty(m, dtype=np.float64)
    b[:n] = b_hat[:n]
    if m > n:
        b[n:] = np.random.uniform(b_hat[n:], b_max[n:])

    # === Шаг 5 ===
    lambda_hat = np.random.uniform(lam_min, lam_max, size=n)

    # === Шаг 6 ===
    # c = - Σ λ̂_i * A_i•  (только для первых n строк)
    A_part = A[:n, :]
    c = - (lambda_hat @ A_part.toarray())

    # === Оптимальные двойственные переменные ===
    lambda_star = np.concatenate([lambda_hat, np.zeros(m - n)])

    # === Сохранение ===
    if save_to:
        sp.save_npz(f"{save_to}_A.npz", A)
        np.savez(
            f"{save_to}_data.npz",
            b=b,
            c=c,
            x_lower=x_lower,
            x_upper=x_upper,
            x_star=x_star,
            lambda_star=lambda_star,
        )
        print(f"Задача сохранена в файлы: {save_to}_A.npz и {save_to}_data.npz")
        return None

    return A, b, c, (x_lower, x_upper), x_star, lambda_star


def lp_cvxp(
    n=7500,
    m=7500,
    density=0.01,
    a_min=-30.0,
    a_max=60.0,
    x_bounds=(-1000.0, 1000.0),
    seed=None,
    save_to=None,
    easy=False,
):
    """
    Генерация задачи линейного программирования (ЛП) через CVXPY.
    
    Формат:
        minimize c^T x
        subject to A x <= b,  x_lower <= x <= x_upper

    Аргументы:
        easy=True согласованный случай (x0 = 1, b = A@x0, c = -A.sum(axis=0))
        easy=False случайная реальная задача

    Возвращает:
        prob, A, b, c, x
    """

    if seed is not None:
        np.random.seed(seed)

    x_lower, x_upper = x_bounds

    # === 1. Генерация матрицы A ===
    A = sp.random(m, n, density=density, format="csr", dtype=np.float64)
    A.data = np.random.uniform(a_min, a_max, size=A.nnz)

    # === 2. Варианты генерации ===
    if easy:
        # Согласованный случай
        x0 = np.ones((n, 1))
        b = A @ x0
        c = -A.sum(axis=0)
        print("Генерация: согласованный случай (easy=True)")
    else:
        # Случайная задача
        b = np.random.uniform(0, 100, size=m)
        c = np.random.uniform(-1, 1, size=(1, n))
        print("Генерация: случайная задача (easy=False)")

    # === 3. Формирование задачи CVXPY ===
    x = cvx.Variable(shape=(n, 1), name="x")
    obj = cvx.Minimize(c @ x)
    constraints = [A @ x <= b, x >= x_lower, x <= x_upper]
    prob = cvx.Problem(obj, constraints)

    # === 4. Сохранение ===
    if save_to:
        sp.save_npz(f"{save_to}_A.npz", A)
        np.savez(
            f"{save_to}_data.npz",
            b=b,
            c=c,
            x_lower=x_lower,
            x_upper=x_upper,
        )
        print(f"Данные сохранены: {save_to}_A.npz, {save_to}_data.npz")

    return prob, A, b, c, x
