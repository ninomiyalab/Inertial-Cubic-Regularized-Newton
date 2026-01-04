import numpy as np
from scipy.linalg import lu_factor, lu_solve

def newton_method(f, jac, hess, w_init, alpha=0.1, max_iter=100, tol=1e-6):
    """Standard Newton Method"""
    w = np.array(w_init, dtype=float)
    history = [f(w)]
    for ite in range(max_iter):
        try:
            lu, piv = lu_factor(hess(w))
            delta_x = lu_solve((lu, piv), -jac(w))
            w += alpha * delta_x
            history.append(f(w))
            if np.linalg.norm(delta_x) < tol: break
        except np.linalg.LinAlgError: break
    return {"x": w, "history": history, "iterations": ite + 1}
