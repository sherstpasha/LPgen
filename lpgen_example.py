from utils import lpgen
import cvxpy as cvx
import time


# === Генерация тестовой задачи ===
A, b, c, (x_lower, x_upper), x_star, lambda_star = lpgen(
    n=500,
    m=500,
    density=0.1,
)

# === Формирование задачи CVXPY ===
x = cvx.Variable(shape=(A.shape[1], 1), name="x")
obj = cvx.Minimize(c @ x)
constraints = [A @ x <= b, x >= x_lower.reshape(-1, 1), x <= x_upper.reshape(-1, 1)]
prob = cvx.Problem(obj, constraints)

# === Решение задачи ===
t0 = time.monotonic()
prob.solve(solver=cvx.HIGHS)
t1 = time.monotonic()

# === Результаты ===
print(f"Решение завершено за {t1 - t0:.3f} с")
print(f"Статус: {prob.status}")
print(f"Оптимум: {prob.value:.6f}")
if x.value is not None:
    print(f"Первые 5 переменных: {x.value[:5].ravel()}")
else:
    print("Переменные не вычислены (решение не найдено).")
