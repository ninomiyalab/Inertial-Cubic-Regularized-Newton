import numpy as np
from scipy.linalg import lu_factor, lu_solve

def diagonal_newton(f, jac, hess, w_init, alpha=1.0, eps=1.0, max_iter=100, tol=1e-6):
    """Diagonal (Proximal) Newton Method"""
    w = np.array(w_init, dtype=float)
    identity = np.identity(len(w))
    history = [f(w)]
    for ite in range(max_iter):
        try:
            lu, piv = lu_factor(hess(w) + eps * identity)
            delta_x = lu_solve((lu, piv), -jac(w))
            w += alpha * delta_x
            history.append(f(w))
            if np.linalg.norm(delta_x) < tol: break
        except np.linalg.LinAlgError: break
    return {"x": w, "history": history, "iterations": ite + 1}
