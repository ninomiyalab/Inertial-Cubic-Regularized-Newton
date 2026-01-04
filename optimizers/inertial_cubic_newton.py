import numpy as np
from scipy.linalg import lu_factor, lu_solve

def inertial_cubic_newton(f, jac, hess, w_init, mu=0.75, eps=1.0, max_iter=100, tol=1e-6):
    """Inertial Cubic Newton Method"""
    w = np.array(w_init, dtype=float)
    identity = np.identity(len(w))
    mom = np.zeros_like(w)
    history = [f(w)]
    for ite in range(max_iter):
        pre_w = w.copy()
        w = w + mu * mom
        grad = jac(w)
        cubic_term = eps * np.sqrt(np.linalg.norm(grad))
        try:
            lu, piv = lu_factor(hess(w) + cubic_term * identity)
            delta_x = lu_solve((lu, piv), -grad)
            w += delta_x
            mom = w - pre_w
            history.append(f(w))
            if np.linalg.norm(delta_x) < tol: break
        except np.linalg.LinAlgError: break
    return {"x": w, "history": history, "iterations": ite + 1}
