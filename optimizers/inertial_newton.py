import numpy as np
from scipy.linalg import lu_factor, lu_solve

def inertial_newton(f, jac, hess, w_init, mu=0.95, alpha=1.0, max_iter=100, tol=1e-6):
    """Inertial Newton Method (with Momentum)"""
    w = np.array(w_init, dtype=float)
    mom = np.zeros_like(w)
    history = [f(w)]
    for ite in range(max_iter):
        pre_w = w.copy()
        w = w + mu * mom
        try:
            lu, piv = lu_factor(hess(w))
            delta_x = lu_solve((lu, piv), -jac(w))
            w += alpha * delta_x
            mom = w - pre_w
            history.append(f(w))
            if np.linalg.norm(delta_x) < tol: break
        except np.linalg.LinAlgError: break
    return {"x": w, "history": history, "iterations": ite + 1}
