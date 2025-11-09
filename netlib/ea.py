from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
import numpy as np
from highspy import Highs


def read_mps(filename, bound_limit=1e6):
    """Считывает задачу из MPS через HiGHS и возвращает (A, b, c, bounds)
    Все бесконечные значения заменяются на ±bound_limit."""
    h = Highs()
    h.readModel(filename)
    lp = h.getLp()

    # Матрица ограничений
    start = lp.a_matrix_.start_
    index = lp.a_matrix_.index_
    value = lp.a_matrix_.value_

    A = np.zeros((lp.num_row_, lp.num_col_))
    for j in range(lp.num_col_):
        for k in range(start[j], start[j + 1]):
            A[index[k], j] = value[k]

    # Коэффициенты
    c = np.nan_to_num(np.array(lp.col_cost_), nan=0.0)

    # Ограничения
    row_lower = np.nan_to_num(np.array(lp.row_lower_), nan=-bound_limit)
    row_upper = np.nan_to_num(np.array(lp.row_upper_), nan=bound_limit)

    # Границы переменных
    col_lower = np.nan_to_num(np.array(lp.col_lower_), nan=-bound_limit)
    col_upper = np.nan_to_num(np.array(lp.col_upper_), nan=bound_limit)

    # Заменяем бесконечности
    col_lower = np.clip(col_lower, -bound_limit, bound_limit)
    col_upper = np.clip(col_upper, -bound_limit, bound_limit)
    row_lower = np.clip(row_lower, -bound_limit, bound_limit)
    row_upper = np.clip(row_upper, -bound_limit, bound_limit)

    # Для простоты: берём b как верхние границы
    b = np.where(np.isfinite(row_upper), row_upper, row_lower)
    bounds = np.vstack((col_lower, col_upper)).T

    return A, b, c, bounds


# Загружаем задачу
A, b, c, bounds = read_mps("benchmarks/afiro.mps")


def f(x):
    # штраф за нарушение ограничений
    penalty = np.sum(np.maximum(0, A @ x - b) ** 2)
    return c @ x + 1e5 * penalty


problem = FunctionalProblem(n_var=len(c), objs=f, xl=bounds[:, 0], xu=bounds[:, 1])

res = minimize(problem, GA(pop_size=1000), termination=("n_gen", 2000))
print("Лучшее значение:", res.F[0])
