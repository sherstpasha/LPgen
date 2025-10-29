import time
from typing import Tuple

import numpy as np
import scipy.sparse as sp

import clarabel as cb

from utils import lpgen


def build_lp_as_inequalities(
    A: sp.spmatrix,
    b: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
) -> Tuple[sp.csc_matrix, np.ndarray]:
    """
    Converts constraints
        A x <= b,  x_lower <= x <= x_upper
    into Clarabel standard inequality form A_ineq x + s = b_ineq with s in R_+.
    Returns (A_ineq, b_ineq).
    """
    m, n = A.shape
    I = sp.eye(n, format="csc")
    blocks = [A.tocsc(), I, -I]
    A_ineq = sp.vstack(blocks, format="csc")
    b_ineq = np.concatenate([b.reshape(-1), x_upper.reshape(-1), (-x_lower).reshape(-1)])
    return A_ineq, b_ineq


def solve_with_clarabel(n: int = 200, density: float = 0.01, seed: int = 42):
    # Generate LP instance
    A, b, c, (x_lower, x_upper), *_ = lpgen(
        n=n, m=n, density=density, seed=seed,
        delta=30.0, a_min=-20.0, a_max=20.0, lam_min=1.0, lam_max=9.0,
    )

    # Build inequality form
    A_ineq, b_ineq = build_lp_as_inequalities(A, b, x_lower, x_upper)

    # Clarabel problem: minimize q^T x subject to A_ineq x + s = b_ineq, s in R_+
    P = sp.csc_matrix((n, n))  # LP -> zero Hessian
    q = c.reshape(-1)
    cones = [cb.NonnegativeConeT(A_ineq.shape[0])]

    settings = cb.DefaultSettings()
    settings.max_iter = 1000
    settings.time_limit = 10.0

    solver = cb.DefaultSolver(P, q, A_ineq, b_ineq, cones, settings)

    t0 = time.monotonic()
    result = solver.solve()
    elapsed = time.monotonic() - t0

    status = getattr(result, "status", getattr(solver, "status", "unknown"))
    obj = getattr(result, "obj_val", None)
    print(f"Clarabel status: {status}, time: {elapsed:.3f}s, obj: {obj}")


if __name__ == "__main__":
    solve_with_clarabel()

