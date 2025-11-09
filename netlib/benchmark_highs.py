import os
import time
import pandas as pd
from highspy import Highs

# Папка с задачами и имя выходного Excel
DATA_DIR = "benchmarks"
RESULT_FILE = "highs_results.xlsx"

# Собираем список MPS-файлов
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".mps")])
results = []

for fname in files:
    path = os.path.join(DATA_DIR, fname)
    print(f"Решаем {fname} ...")

    h = Highs()
    start = time.time()
    try:
        # Загружаем и решаем модель
        h.readModel(path)
        h.run()
        elapsed = time.time() - start

        # Получаем статистику через API
        status = h.getModelStatus()
        objective = h.getObjectiveValue()
        nrows = h.getNumRow()
        ncols = h.getNumCol()
        nnz = h.getNumNz()

        results.append(
            {
                "problem": fname,
                "status": status,
                "objective": objective,
                "rows": nrows,
                "cols": ncols,
                "nonzeros": nnz,
                "time_sec": round(elapsed, 3),
            }
        )
        print(f"{fname}: статус={status}, obj={objective:.4f}, время={elapsed:.2f} сек")

    except Exception as e:
        elapsed = time.time() - start
        results.append(
            {
                "problem": fname,
                "status": f"error: {e.__class__.__name__}",
                "objective": None,
                "rows": None,
                "cols": None,
                "nonzeros": None,
                "time_sec": round(elapsed, 3),
            }
        )
        print(f"Ошибка при решении {fname}: {e}")

# Создаём таблицу результатов
df = pd.DataFrame(results)

# Добавляем относительную плотность матрицы (ненулевые / общее количество элементов)
df["density"] = df.apply(
    lambda r: (
        r["nonzeros"] / (r["rows"] * r["cols"]) if r["rows"] and r["cols"] else None
    ),
    axis=1,
)

# Сортируем по количеству строк и столбцов
df = df.sort_values(by=["rows", "cols"], ascending=[True, True])

# Сохраняем в Excel
df.to_excel(RESULT_FILE, index=False)

print(f"\nГотово! Результаты сохранены в {RESULT_FILE}")
