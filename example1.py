import time
import numpy as np
from scipy.stats import uniform
import scipy.sparse as sp
import cvxpy as cvx


# === Класс для измерения времени ===
class timex:
    def __enter__(self):
        self.t = time.monotonic()
        return self
    def __exit__(self, typ, value, traceback):
        print('actual run time: {:.4f} s'.format(time.monotonic() - self.t))


# === Параметры задачи ===
i = 1000
j = 1000
density = 0.01

# === Сжатое хранение строк (Compressed Sparse Row) ===
A = sp.random(i, j, density=density, format='csr',
              data_rvs=lambda s: uniform.rvs(loc=-30., scale=60., size=s))

# === Исходное приближение ===
x0 = np.ones(shape=(j, 1), dtype='d')

# === Правая часть ===
b = A @ x0

# === Целевая функция ===
c = -A.sum(axis=0)

# === Переменная ===
x = cvx.Variable(shape=(j, 1))

# === Целевая функция и ограничения ===
obj = cvx.Minimize(c @ x)
constraints = [A @ x <= b, x >= -1000, x <= 1000]
prob = cvx.Problem(obj, constraints)

# === Решение с замером времени ===
with timex():
    prob.solve(solver=cvx.CLARABEL)

# === Вывод результатов ===
print(f'used solver {prob.solver_stats.solver_name}')
print(prob.status)
print(f'obj0 = {(c @ x0)[0,0]}')
print(f'obj* = {(c @ x.value)[0,0]}')
