"""
Решение задачи линейного программирования из MPS файла с помощью HiGHS
"""

import time
import highspy

# === ПАРАМЕТРЫ ===
mps_file = "16_n14.mps"

print(f"Загрузка задачи из файла: {mps_file}")
print("=" * 60)

# Создаём экземпляр HiGHS
h = highspy.Highs()

# Настройки (опционально)
h.setOptionValue("output_flag", True)  # Показывать вывод
h.setOptionValue("log_to_console", True)

# Читаем MPS файл
print(f"\n📂 Чтение MPS файла...\n")
status = h.readModel(mps_file)

if status != highspy.HighsStatus.kOk:
    print(f"❌ Ошибка при чтении файла: {status}")
    exit(1)

print(f"✅ Файл успешно загружен")

# Получаем информацию о задаче
lp = h.getLp()
print(f"\n📊 Информация о задаче:")
print(f"   Переменных: {lp.num_col_}")
print(f"   Ограничений: {lp.num_row_}")
print(f"   Ненулевых элементов: {len(lp.a_matrix_.value_)}")

# Решаем задачу
print(f"\n🚀 Запуск решателя HiGHS...\n")
t0 = time.time()
status = h.run()
elapsed = time.time() - t0

print("\n" + "=" * 60)
print(f"⏱️  Время решения: {elapsed:.3f} секунд")
print("=" * 60)

# Получаем информацию о решении
info = h.getInfo()
model_status = h.getModelStatus()

print(f"\n📋 Статус модели: {model_status}")
print(f"   Итераций симплекс-метода: {info.simplex_iteration_count}")
print(f"   Итераций IPM: {info.ipm_iteration_count}")

if model_status == highspy.HighsModelStatus.kOptimal:
    solution = h.getSolution()
    print(f"\n✅ Оптимальное решение найдено!")
    print(f"   Целевая функция: {info.objective_function_value:.10f}")

    # Выводим первые несколько значений переменных
    print(f"\n📊 Первые 10 значений переменных:")
    for i in range(min(10, lp.num_col_)):
        print(f"   x[{i}] = {solution.col_value[i]:.6f}")

    if lp.num_col_ > 10:
        print(f"   ... (всего {lp.num_col_} переменных)")

    # Проверка прималов и двойственных переменных
    print(f"\n📊 Дополнительная информация:")
    print(
        f"   Количество базисных переменных: {sum(1 for v in solution.col_value if abs(v) > 1e-10)}"
    )

elif model_status == highspy.HighsModelStatus.kInfeasible:
    print(f"\n❌ Задача несовместна (infeasible)")
elif model_status == highspy.HighsModelStatus.kUnbounded:
    print(f"\n❌ Задача неограничена (unbounded)")
else:
    print(f"\n⚠️  Неожиданный статус: {model_status}")

print("\n" + "=" * 60)
print("✅ Готово!")
