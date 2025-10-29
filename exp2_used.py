import os
import json
import time
import webbrowser
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import optuna
import clarabel as cb

from utils import lpgen

try:
    import optuna_dashboard  # type: ignore
except Exception:
    optuna_dashboard = None  # type: ignore


# Defaults / config
DEFAULT_N_TRIALS = int(os.getenv("OPTUNA_TRIALS", 100))
DEFAULT_DENSITY = float(os.getenv("LP_DENSITY", 0.01))
DEFAULT_SIZES = tuple(map(int, os.getenv("LP_SIZES", "").split(","))) if os.getenv("LP_SIZES") else tuple(map(int, np.linspace(150, 2000, 15)))
DEFAULT_GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", 42))
DEFAULT_PARALLEL_JOBS = int(os.getenv("OPTUNA_JOBS", max(1, (os.cpu_count() or 2) - 1)))
DEFAULT_STORAGE_URL = os.getenv("OPTUNA_STORAGE", "sqlite:///clarabel_optuna.db")
DEFAULT_DASHBOARD_HOST = os.getenv("OPTUNA_DASHBOARD_HOST", "127.0.0.1")
DEFAULT_DASHBOARD_PORT = int(os.getenv("OPTUNA_DASHBOARD_PORT", 8090))
DEFAULT_OPEN_DASHBOARD = os.getenv("OPEN_DASHBOARD", "1") != "0"
DEFAULT_TASKS_PER_TRIAL = int(os.getenv("OPTUNA_TASKS_PER_TRIAL", 20))


@dataclass
class ProblemData:
    n: int
    P: sp.csc_matrix
    q: np.ndarray
    A_ineq: sp.csc_matrix
    b_ineq: np.ndarray
    cones: List[Any]


def _build_inequalities(A: sp.spmatrix, b: np.ndarray, x_lower: np.ndarray, x_upper: np.ndarray) -> Tuple[sp.csc_matrix, np.ndarray]:
    m, n = A.shape
    I = sp.eye(n, format="csc")
    A_ineq = sp.vstack([A.tocsc(), I, -I], format="csc")
    b_ineq = np.concatenate([b.reshape(-1), x_upper.reshape(-1), (-x_lower).reshape(-1)])
    return A_ineq, b_ineq


def build_problems(sizes: List[int], density: float, base_seed: int) -> List[ProblemData]:
    problems: List[ProblemData] = []
    for i, n in enumerate(sizes):
        seed = base_seed + i
        A, b, c, (x_lower, x_upper), _, _ = lpgen(
            n=n,
            m=n,
            delta=30.0,
            a_min=-20.0,
            a_max=20.0,
            lam_min=1.0,
            lam_max=9.0,
            density=density,
            seed=seed,
            save_to=None,
        )
        A_ineq, b_ineq = _build_inequalities(A, b, x_lower, x_upper)
        P = sp.csc_matrix((n, n))
        q = c.reshape(-1)
        cones: List[Any] = [cb.NonnegativeConeT(A_ineq.shape[0])]
        problems.append(ProblemData(n=n, P=P, q=q, A_ineq=A_ineq, b_ineq=b_ineq, cones=cones))
    return problems


def solve_one(problem: ProblemData, settings: Optional[Any] = None) -> Tuple[str, Optional[float]]:
    if settings is None:
        settings = cb.DefaultSettings()
    try:
        solver = cb.DefaultSolver(problem.P, problem.q, problem.A_ineq, problem.b_ineq, problem.cones, settings)
    except Exception:
        return "build_error", None
    t0 = time.monotonic()
    try:
        result = solver.solve()
        elapsed = time.monotonic() - t0
        status = getattr(result, "status", getattr(solver, "status", "unknown"))
        return str(status), elapsed
    except Exception:
        return "error", None


def build_settings(trial: optuna.Trial) -> Any:
    s = cb.DefaultSettings()

    def setp(name: str, value: Any):
        try:
            if hasattr(s, name):
                setattr(s, name, value)
        except Exception:
            pass

    # Equilibration
    if hasattr(s, "equilibrate_enable"):
        eq = trial.suggest_categorical("equilibrate_enable", [False, True])
        setp("equilibrate_enable", eq)
        if eq:
            if hasattr(s, "equilibrate_max_iter"):
                setp("equilibrate_max_iter", trial.suggest_int("equilibrate_max_iter", 3, 25))
            if hasattr(s, "equilibrate_min_scaling"):
                setp("equilibrate_min_scaling", trial.suggest_float("equilibrate_min_scaling", 1e-8, 1e-2, log=True))
            if hasattr(s, "equilibrate_max_scaling"):
                setp("equilibrate_max_scaling", trial.suggest_float("equilibrate_max_scaling", 1e3, 1e7, log=True))

    # Presolve
    if hasattr(s, "presolve_enable"):
        setp("presolve_enable", trial.suggest_categorical("presolve_enable", [False, True]))

    # Static regularization
    if hasattr(s, "static_regularization_enable"):
        sreg = trial.suggest_categorical("static_reg_enable", [False, True])
        setp("static_regularization_enable", sreg)
        if sreg and hasattr(s, "static_regularization_eps"):
            setp("static_regularization_eps", trial.suggest_float("static_reg_eps", 1e-12, 1e-3, log=True))
        if sreg and hasattr(s, "static_regularization_proportional"):
            use_prop = trial.suggest_categorical("use_static_reg_prop", [False, True])
            setp("static_regularization_proportional", 0.0 if not use_prop else trial.suggest_float("static_reg_prop", 1e-12, 1e-6, log=True))

    # Dynamic regularization
    if hasattr(s, "dynamic_regularization_enable"):
        dreg = trial.suggest_categorical("dyn_reg_enable", [False, True])
        setp("dynamic_regularization_enable", dreg)
        if dreg and hasattr(s, "dynamic_regularization_eps"):
            setp("dynamic_regularization_eps", trial.suggest_float("dyn_reg_eps", 1e-15, 1e-9, log=True))

    # Line search / steps
    if hasattr(s, "max_step_fraction"):
        setp("max_step_fraction", trial.suggest_float("max_step_fraction", 0.90, 0.995))
    if hasattr(s, "linesearch_backtrack_step"):
        setp("linesearch_backtrack_step", trial.suggest_float("linesearch_backtrack_step", 0.60, 0.95))
    if hasattr(s, "min_switch_step_length"):
        setp("min_switch_step_length", trial.suggest_float("min_switch_step_length", 1e-3, 2e-1, log=True))
    if hasattr(s, "min_terminate_step_length"):
        setp("min_terminate_step_length", trial.suggest_float("min_terminate_step_length", 1e-6, 1e-3, log=True))

    # Iterative refinement
    if hasattr(s, "iterative_refinement_enable"):
        ir = trial.suggest_categorical("ir_enable", [False, True])
        setp("iterative_refinement_enable", ir)
        if ir:
            if hasattr(s, "iterative_refinement_max_iter"):
                setp("iterative_refinement_max_iter", trial.suggest_int("ir_max_iter", 2, 12))
            if hasattr(s, "iterative_refinement_reltol"):
                setp("iterative_refinement_reltol", trial.suggest_float("ir_reltol", 1e-15, 1e-10, log=True))
            if hasattr(s, "iterative_refinement_abstol"):
                setp("iterative_refinement_abstol", trial.suggest_float("ir_abstol", 1e-15, 1e-10, log=True))

    # Direct solve method (restrict to qdldl for compatibility)
    if hasattr(s, "direct_solve_method"):
        setp("direct_solve_method", trial.suggest_categorical("direct_solve_method", ["qdldl"]))

    return s


def params_to_settings(params: Dict) -> Any:
    s = cb.DefaultSettings()

    def setp(name: str, value: Any):
        try:
            if hasattr(s, name):
                setattr(s, name, value)
        except Exception:
            pass

    if "equilibrate_enable" in params:
        setp("equilibrate_enable", params["equilibrate_enable"])
    if hasattr(s, "equilibrate_enable") and getattr(s, "equilibrate_enable", False):
        if "equilibrate_max_iter" in params:
            setp("equilibrate_max_iter", params["equilibrate_max_iter"])
        if "equilibrate_min_scaling" in params:
            setp("equilibrate_min_scaling", params["equilibrate_min_scaling"])
        if "equilibrate_max_scaling" in params:
            setp("equilibrate_max_scaling", params["equilibrate_max_scaling"])

    if "presolve_enable" in params:
        setp("presolve_enable", params["presolve_enable"])

    if "static_reg_enable" in params:
        setp("static_regularization_enable", params["static_reg_enable"])
    if hasattr(s, "static_regularization_enable") and getattr(s, "static_regularization_enable", False):
        if "static_reg_eps" in params and hasattr(s, "static_regularization_eps"):
            setp("static_regularization_eps", params["static_reg_eps"])
        use_prop = params.get("use_static_reg_prop", False)
        if hasattr(s, "static_regularization_proportional"):
            setp("static_regularization_proportional", 0.0 if not use_prop else params.get("static_reg_prop", 0.0))

    if "dyn_reg_enable" in params:
        setp("dynamic_regularization_enable", params["dyn_reg_enable"])
    if hasattr(s, "dynamic_regularization_enable") and getattr(s, "dynamic_regularization_enable", False):
        if "dyn_reg_eps" in params and hasattr(s, "dynamic_regularization_eps"):
            setp("dynamic_regularization_eps", params["dyn_reg_eps"])

    if "max_step_fraction" in params:
        setp("max_step_fraction", params["max_step_fraction"])
    if "linesearch_backtrack_step" in params:
        setp("linesearch_backtrack_step", params["linesearch_backtrack_step"])
    if "min_switch_step_length" in params:
        setp("min_switch_step_length", params["min_switch_step_length"])
    if "min_terminate_step_length" in params:
        setp("min_terminate_step_length", params["min_terminate_step_length"])

    if "ir_enable" in params:
        setp("iterative_refinement_enable", params["ir_enable"])
    if hasattr(s, "iterative_refinement_enable") and getattr(s, "iterative_refinement_enable", False):
        if "ir_max_iter" in params:
            setp("iterative_refinement_max_iter", params["ir_max_iter"])
        if "ir_reltol" in params:
            setp("iterative_refinement_reltol", params["ir_reltol"])
        if "ir_abstol" in params:
            setp("iterative_refinement_abstol", params["ir_abstol"])

    if "direct_solve_method" in params:
        setp("direct_solve_method", params["direct_solve_method"])

    return s


def build_problems_for_trial(trial: optuna.Trial, sizes: List[int], density: float, base_seed: int, tasks_per_trial: int) -> List[ProblemData]:
    rng = np.random.RandomState(base_seed + int(trial.number) * 100003)
    chosen_sizes = list(rng.choice(sizes, size=tasks_per_trial, replace=(tasks_per_trial > len(sizes))))
    problems: List[ProblemData] = []
    for i, n in enumerate(chosen_sizes):
        seed = base_seed + int(trial.number) * 100003 + i
        A, b, c, (x_lower, x_upper), _, _ = lpgen(
            n=int(n),
            m=int(n),
            delta=30.0,
            a_min=-20.0,
            a_max=20.0,
            lam_min=1.0,
            lam_max=9.0,
            density=density,
            seed=int(seed),
            save_to=None,
        )
        A_ineq, b_ineq = _build_inequalities(A, b, x_lower, x_upper)
        P = sp.csc_matrix((int(n), int(n)))
        q = c.reshape(-1)
        cones: List[Any] = [cb.NonnegativeConeT(A_ineq.shape[0])]
        problems.append(ProblemData(n=int(n), P=P, q=q, A_ineq=A_ineq, b_ineq=b_ineq, cones=cones))
    return problems


def objective_factory(sizes: List[int], density: float, base_seed: int, tasks_per_trial: int):
    def objective(trial: optuna.Trial) -> float:
        use_defaults = trial.suggest_categorical("use_defaults", [False, True])
        settings = cb.DefaultSettings() if use_defaults else build_settings(trial)

        problems = build_problems_for_trial(trial, sizes, density, base_seed, tasks_per_trial)

        total_time = 0.0
        for p in problems:
            status, elapsed = solve_one(p, settings)
            ok = str(status).lower().startswith("solved") and elapsed is not None
            if not ok:
                return 1e12  # fail trial if any problem not solved
            total_time += float(elapsed)
        return total_time

    return objective


def start_optuna_dashboard_thread(storage_url: str, host: str, port: int) -> Optional[threading.Thread]:
    if optuna_dashboard is None:
        print("[dashboard] optuna-dashboard не установлен. pip install optuna-dashboard")
        return None
    def _run():
        try:
            optuna_dashboard.run_server(storage_url, host=host, port=port)
        except Exception as e:
            print(f"[dashboard] ошибка запуска: {e}")
    th = threading.Thread(target=_run, name="optuna-dashboard", daemon=True)
    th.start()
    time.sleep(1.0)
    try:
        webbrowser.open(f"http://{host}:{port}/")
    except Exception:
        pass
    return th


def ensure_storage_ready(storage_url: str, study_name: str = "clarabel_speed") -> None:
    try:
        optuna.create_study(direction="minimize", study_name=study_name, storage=storage_url, load_if_exists=True)
    except Exception:
        pass


def tune_clarabel(
    n_trials: int = DEFAULT_N_TRIALS,
    density: float = DEFAULT_DENSITY,
    sizes: Tuple[int, ...] = DEFAULT_SIZES,
    base_seed: int = DEFAULT_GLOBAL_SEED,
    n_jobs: int = DEFAULT_PARALLEL_JOBS,
    storage_url: str = DEFAULT_STORAGE_URL,
):
    study = optuna.create_study(
        direction="minimize",
        study_name="clarabel_speed",
        storage=storage_url,
        load_if_exists=True,
    )
    study.enqueue_trial({"use_defaults": True})
    obj = objective_factory(list(sizes), density, base_seed, DEFAULT_TASKS_PER_TRIAL)
    study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)
    return study


def evaluate_compare(
    tuned_params: Dict,
    density: float = DEFAULT_DENSITY,
    sizes: Tuple[int, ...] = DEFAULT_SIZES,
    eval_seed: int = 20240501,
):
    problems = build_problems(list(sizes), density, eval_seed)
    rows = []
    default_settings = cb.DefaultSettings()
    tuned_settings = default_settings if tuned_params.get("use_defaults") else params_to_settings({k: v for k, v in tuned_params.items() if k != "use_defaults"})
    for p in problems:
        st_status, st_time = solve_one(p, default_settings)
        tu_status, tu_time = solve_one(p, tuned_settings)
        rows.append({
            "n": p.n,
            "CLARABEL_standard_status": st_status,
            "CLARABEL_standard_time_sec": None if st_time is None else round(float(st_time), 6),
            "CLARABEL_optimized_status": tu_status,
            "CLARABEL_optimized_time_sec": None if tu_time is None else round(float(tu_time), 6),
        })
    df = pd.DataFrame(rows).sort_values("n").reset_index(drop=True)
    return df


def main():
    n_trials = DEFAULT_N_TRIALS
    density = DEFAULT_DENSITY
    sizes = DEFAULT_SIZES
    n_jobs = DEFAULT_PARALLEL_JOBS

    ensure_storage_ready(DEFAULT_STORAGE_URL, "clarabel_speed")
    if DEFAULT_OPEN_DASHBOARD:
        start_optuna_dashboard_thread(DEFAULT_STORAGE_URL, DEFAULT_DASHBOARD_HOST, DEFAULT_DASHBOARD_PORT)

    study = tune_clarabel(n_trials=n_trials, density=density, sizes=sizes, n_jobs=n_jobs, storage_url=DEFAULT_STORAGE_URL)
    best_params = dict(study.best_trial.params)

    print("Best params:", json.dumps(best_params, ensure_ascii=False))
    compare_df = evaluate_compare(best_params, density=density, sizes=sizes, eval_seed=DEFAULT_GLOBAL_SEED + 777)

    out_csv = "clarabel_optuna_compare.csv"
    compare_df.to_csv(out_csv, index=False)
    print(f"Saved compare CSV: {out_csv}")
    print(compare_df)


if __name__ == "__main__":
    main()
