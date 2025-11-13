# Оптимизация параметров HiGHS для NETLIB

## Быстрый старт

```bash
# Умный подбор (рекомендуется, 2-4 часа)
python netlib_optimize.py

# Полный перебор (долго, 1-5 дней)
python netlib_gridsearch.py

# Просмотр сводки параметров
python PARAMETERS_SUMMARY.txt
```

## Что здесь?

- **netlib_optimize.py** - умный последовательный подбор параметров (greedy search)
- **netlib_gridsearch.py** - полный перебор всех 3,672 комбинаций (grid search)
- **PARAMETERS_INFO.md** - детальная документация по параметрам
- **PARAMETERS_SUMMARY.txt** - краткая сводка

## Параметры оптимизации

### Всего комбинаций: **3,672**

| Solver  | Комбинаций |
|---------|------------|
| Simplex | 3,600      |
| IPM     | 27         |
| IPX     | 27         |
| HiPO    | 9          |
| PDLP    | 9          |

### 7 параметров:

1. **presolve** (3 варианта) - предобработка
2. **parallel** (3 варианта) - параллелизм
3. **simplex_strategy** (5 вариантов) - стратегия симплекса
4. **simplex_scale_strategy** (5 вариантов) - масштабирование
5. **simplex_dual_edge_weight_strategy** (4 варианта) - веса двойственных рёбер
6. **simplex_primal_edge_weight_strategy** (4 варианта) - веса прямых рёбер
7. **run_crossover** (3 варианта) - кроссовер IPM

## Результаты

Оба скрипта создают:
- Excel с результатами (3 листа: Results, Tuning_History, Summary)
- JSON с лучшей конфигурацией
- Графики сравнения
- Разделение на TRAIN (70%) / TEST (30%)

## Какой выбрать?

**Умный подбор** (`netlib_optimize.py`):
- ✅ Быстро (2-4 часа)
- ✅ Хорошие результаты
- ⚠️ Не гарантирует оптимум

**Полный перебор** (`netlib_gridsearch.py`):
- ✅ Находит лучшую комбинацию
- ✅ Полная статистика по всем вариантам
- ⚠️ Очень долго (1-5 дней)

## Требования

```bash
pip install highspy pandas openpyxl matplotlib tqdm
```
