import time
import numpy as np
from highspy import Highs, HighsSolution


class RacingHighs:
    """
    Обертка вокруг HiGHS, реализующая "гонку прогревов" (racing)
    между несколькими стратегиями, например IPM vs SIMPLEX.
    """

    def __init__(
        self,
        model_path,
        warmup_fraction=0.1,
        max_iter=None,
        configs=None,
        verbose=False,
    ):
        self.model_path = model_path
        self.warmup_fraction = warmup_fraction
        self.max_iter = max_iter  # если None — будет считано из HiGHS
        self.verbose = verbose

        self.configs = configs or [
            {"solver": "simplex", "presolve": "on"},
            {"solver": "ipm", "presolve": "on"},
        ]

        self.best_cfg = None
        self.best_obj = None
        self.best_sol = None
        self.best_time = None

    def _get_default_iter_limits(self):
        """Считывает стандартные лимиты из HiGHS"""
        h = Highs()
        h.readModel(self.model_path)
        ipm_limit = h.getOptionValue("ipm_iteration_limit")
        simplex_limit = h.getOptionValue("simplex_iteration_limit")

        # Приводим к int, если строка
        try:
            ipm_limit = int(ipm_limit)
        except Exception:
            ipm_limit = 10000  # запасной дефолт

        try:
            simplex_limit = int(simplex_limit)
        except Exception:
            simplex_limit = 10000

        return ipm_limit, simplex_limit

    def _run_partial(self, cfg, limit):
        """Запуск HiGHS с ограничением итераций"""
        h = Highs()
        h.readModel(self.model_path)
        h.setOptionValue("output_flag", self.verbose)

        for k, v in cfg.items():
            h.setOptionValue(k, v)

        if cfg.get("solver") == "ipm":
            h.setOptionValue("ipm_iteration_limit", limit)
        else:
            h.setOptionValue("simplex_iteration_limit", limit)

        start = time.time()
        h.run()
        elapsed = time.time() - start

        try:
            obj = h.getObjectiveValue()
            sol = h.getSolution()
            col_values = np.array(sol.col_value, dtype=float)
        except Exception:
            obj, col_values = None, None

        return obj, elapsed, col_values

    def run(self):
        """Главный метод: выполняет гонку и дооптимизацию лучшей стратегии"""
        # получаем реальные лимиты HiGHS
        ipm_def, simplex_def = self._get_default_iter_limits()
        base_iter = self.max_iter or max(ipm_def, simplex_def)
        partial_limit = max(5, int(base_iter * self.warmup_fraction))

        if self.verbose:
            print(f"⚙️ warmup_fraction={self.warmup_fraction}, limit={partial_limit}")

        candidates = []
        for cfg in self.configs:
            obj, t, sol = self._run_partial(cfg, partial_limit)
            print(f"   > {cfg['solver']}: obj={obj}, time={t:.3f}s")
            if obj is not None and np.isfinite(obj):
                candidates.append((obj, t, cfg, sol))

        if not candidates:
            raise RuntimeError("Ни одна конфигурация не завершилась корректно!")

        # выбираем по наименьшему objective
        best_obj, best_time, best_cfg, best_sol = min(candidates, key=lambda x: x[0])

        self.best_cfg = best_cfg
        self.best_obj = best_obj
        self.best_sol = best_sol
        self.best_time = best_time

        print(f"🏁 Победитель гонки: {best_cfg['solver']} (obj={best_obj:.6f})")

        # === Финальное решение ===
        remaining_iters = max(1, base_iter - partial_limit)
        h_final = Highs()
        h_final.readModel(self.model_path)

        # применяем только нужные опции, остальные — дефолтные
        for k, v in best_cfg.items():
            h_final.setOptionValue(k, v)

        # передаём решение, если оно есть
        if best_sol is not None:
            sol_struct = HighsSolution()
            sol_struct.col_value = best_sol.tolist()
            h_final.setSolution(sol_struct)

        if best_cfg["solver"] == "ipm":
            h_final.setOptionValue("ipm_iteration_limit", remaining_iters)
        else:
            h_final.setOptionValue("simplex_iteration_limit", remaining_iters)

        start = time.time()
        h_final.run()
        elapsed = time.time() - start

        final_obj = h_final.getObjectiveValue()
        total_time = best_time + elapsed

        print(
            f"✅ Финал: {best_cfg['solver']}, obj={final_obj:.6f}, total_time={total_time:.3f}s"
        )

        return {
            "best_cfg": best_cfg,
            "warmup_obj": best_obj,
            "final_obj": final_obj,
            "total_time": total_time,
        }
