import time
import pandas as pd
from highspy import Highs
from rh import RacingHighs

MODEL = r"16_n14.mps"


# === Базовый запуск HiGHS ===
def solve_baseline(model_path):
    h = Highs()
    h.readModel(model_path)
    h.setOptionValue("output_flag", False)
    start = time.time()
    h.run()
    elapsed = time.time() - start
    obj = h.getObjectiveValue()
    return obj, elapsed


print("🔹 Запускаем базовый HiGHS...")
obj_base, t_base = solve_baseline(MODEL)
print(f"Базовый HiGHS: objective={obj_base:.6f}, time={t_base:.3f}s\n")

# === RacingHighs ===
print("🏎 Запускаем RacingHighs (гонка)...")
solver = RacingHighs(
    model_path=MODEL,
    warmup_fraction=0.05,  # первые 5% итераций — “гонка”
    configs=[
        {"solver": "simplex", "presolve": "on"},
        {"solver": "ipm", "presolve": "on"},
    ],
)

result = solver.run()

# === Сравнение результатов ===
df = pd.DataFrame(
    [
        {
            "method": "baseline",
            "objective": obj_base,
            "time_sec": t_base,
            "comment": "обычный HiGHS",
        },
        {
            "method": "racing",
            "objective": result["final_obj"],
            "time_sec": result["total_time"],
            "comment": f"гонка ({result['best_cfg']['solver']})",
        },
    ]
)

print("\n📊 Сравнение:")
print(df.to_string(index=False))

# Сохраняем таблицу в Excel
df.to_excel("compare_racing_vs_baseline.xlsx", index=False)
print("\n✅ Результаты сохранены в compare_racing_vs_baseline.xlsx")
