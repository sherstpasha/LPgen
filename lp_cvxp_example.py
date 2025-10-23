from utils import lp_cvxp
import cvxpy as cvx
import time


prob, A, b, c, x = lp_cvxp(n=500, m=500, density=0.1, easy=True)

t0 = time.monotonic()
prob.solve(solver=cvx.HIGHS)
t1 = time.monotonic()

print(f"Решение завершено за {t1 - t0:.3f} с")
print(f"Статус: {prob.status}")
print(f"Оптимум: {prob.value:.6f}")
print(f"Первые 5 переменных: {x.value[:5].ravel()}")