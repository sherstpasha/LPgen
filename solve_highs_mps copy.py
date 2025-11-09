# --- импорт библиотек ---
from highspy import Highs
import numpy as np
from scipy.sparse import csr_matrix
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize


# === 1️⃣ Загружаем задачу из MPS ===
mps_path = "16_n14.mps"  # замените на ваш файл

solver = Highs()
solver.readModel(mps_path)
lp = solver.getLp()

# размеры
m, n = lp.num_row_, lp.num_col_

# Целевая функция
c = np.array(lp.col_cost_)

# Разреженная матрица A (CSR формат)
start = np.array(lp.a_matrix_.start_)
index = np.array(lp.a_matrix_.index_)
value = np.array(lp.a_matrix_.value_)
A = csr_matrix((value, index, start), shape=(m, n))

# Правая часть ограничений
b = np.array(lp.row_upper_)

# Границы переменных
lb = np.array(lp.col_lower_)
ub = np.array(lp.col_upper_)

print(f"✅ Загружена модель: {m} ограничений, {n} переменных")


# === 2️⃣ Определяем задачу для pymoo ===
class LPProblem(Problem):
    def __init__(self, A, b, c, lb, ub):
        super().__init__(n_var=len(c), n_obj=1, n_constr=len(b), xl=lb, xu=ub)
        self.A = A
        self.b = b
        self.c = c

    def _evaluate(self, X, out, *args, **kwargs):
        # Целевая функция f = c^T x
        F = np.dot(X, self.c)
        # Ограничения: A x <= b → G = A x - b <= 0
        G = self.A.dot(X) - self.b
        out["F"] = F
        out["G"] = G


problem = LPProblem(A, b, c, lb, ub)

print(f"⚙️ Задача создана: {problem.n_var} переменных, {problem.n_constr} ограничений")


# === 3️⃣ Запускаем эволюционный алгоритм ===
algorithm = GA(pop_size=40)  # можно увеличить для точности

res = minimize(
    problem, algorithm, ("n_gen", 100), seed=1, verbose=True  # число поколений
)

# === 4️⃣ Результаты ===
print("\n🎯 Эволюционное решение задачи из MPS:")
print("Лучшее значение целевой функции:", res.F)
print("Лучшее найденное решение (первые 10 переменных):", res.X[:10])
