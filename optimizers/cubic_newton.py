import numpy as np
from scipy.linalg import lu_factor, lu_solve

def cubic_newton(f, jac, hess, w_init, eps=1.0, max_iter=100, tol=1e-6):
    """Cubic Regularized Newton Method"""
    w = np.array(w_init, dtype=float)
    identity = np.identity(len(w))
    history = [f(w)]
    for ite in range(max_iter):
        grad = jac(w)
        cubic_term = eps * np.sqrt(np.linalg.norm(grad))
        try:
            lu, piv = lu_factor(hess(w) + cubic_term * identity)
            delta_x = lu_solve((lu, piv), -grad)
            w += delta_x
            history.append(f(w))
            if np.linalg.norm(delta_x) < tol: break
        except np.linalg.LinAlgError: break
    return {"x": w, "history": history, "iterations": ite + 1}
