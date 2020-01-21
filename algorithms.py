import warnings

import numpy as np
import cvxpy as cp

from scipy.linalg import solve_discrete_are


def policy_fitting(L, r, xs, us_observed):
    """
    Policy fitting.

    Args:
        - L: function that takes in a cvxpy Variable
            and returns a cvxpy expression representing the objective.
        - r: function that takes in a cvxpy Variable
            and returns a cvxpy expression and a list of constraints
            representing the regularization function.
        - xs: N x n matrix of states.
        - us_observed: N x m matrix of inputs.

    Returns:
        - Kpf: m x n gain matrix found by policy fitting. 
    """
    n = xs.shape[1]
    m = us_observed.shape[1]
    Kpf = cp.Variable((m, n))
    r_obj, r_cons = r(Kpf)
    cp.Problem(cp.Minimize(L(Kpf) + r_obj), r_cons).solve()

    return Kpf.value


def _policy_fitting_with_a_kalman_constraint(L, r, xs, us_observed, A, B, P, Q, R, niter=50, rho=1):
    """
    Policy fitting with a Kalman constraint.

    Args:
        - L: function that takes in a cvxpy Variable
            and returns a cvxpy expression representing the objective.
        - r: function that takes in a cvxpy Variable
            and returns a cvxpy expression and a list of constraints
            representing the regularization function.
        - xs: N x n matrix of states.
        - us_observed: N x m matrix of inputs.
        - A: n x n dynamics matrix.
        - B: n x m dynamics matrix.
        - P: n x n PSD matrix, the initial PSD cost-to-go coefficient.
        - Q: n x n PSD matrix, the initial state cost coefficient.
        - R: n x n PD matrix, the initial input cost coefficient.
        - niter: int (optional). Number of iterations (default=50).
        - rho: double (optional). Penalty parameter (default=1).

    Returns:
        - K: m x n gain matrix found by policy fitting with a Kalman constraint. 
    """
    n, m = B.shape

    K = np.zeros((m, n))
    Y = np.zeros((n + m, n))

    try:
        import mosek
        solver = cp.MOSEK
    except:
        warnings.warn("Solver MOSEK is not installed, falling back to SCS.")
        solver = cp.SCS

    for k in range(niter):
        # K step
        Kcp = cp.Variable((m, n))
        r_obj, r_cons = r(Kcp)
        M = cp.vstack([
            Q + A.T @ P @ (A + B @ Kcp) - P,
            R @ Kcp + B.T @ P @ (A + B @ Kcp)
        ])
        objective = cp.Minimize(L(Kcp) + r_obj + cp.trace(Y.T @ M) + rho / 2 * cp.sum_squares(M))
        prob = cp.Problem(objective, r_cons)
        prob.solve()
        K = Kcp.value

        # P, Q, R step
        Pcp = cp.Variable((n, n), PSD=True)
        Qcp = cp.Variable((n, n), PSD=True)
        Rcp = cp.Variable((m, m), PSD=True)
        M = cp.vstack([
            Qcp + A.T @ Pcp @ (A + B @ K) - Pcp,
            Rcp @ K + B.T @ Pcp @ (B @ K + A)
        ])
        objective = cp.Minimize(cp.trace(Y.T @ M) + rho / 2 * cp.sum_squares(M))
        prob = cp.Problem(objective, [Pcp >> 0, Qcp >> 0, Rcp >> np.eye(m)])
        try:
            prob.solve(solver=solver)
        except:
            prob.solve(solver=cp.SCS, acceleration_lookback=0, max_iters=10000)
        P = Pcp.value
        Q = Qcp.value
        R = Rcp.value

        # Y step
        residual = np.vstack([
            Q + A.T @ P @ (A + B @ K) - P,
            R @ K + B.T @ P @ (A + B @ K)
        ])
        Y = Y + rho * residual

    R = (R + R.T) / 2
    Q = (Q + Q.T) / 2

    w, v = np.linalg.eigh(R)
    w[w < 1e-6] = 1e-6
    R = v @ np.diag(w) @ v.T

    w, v = np.linalg.eigh(Q)
    w[w < 0] = 0
    Q = v @ np.diag(w) @ v.T

    P = solve_discrete_are(A, B, Q, R)

    return -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def policy_fitting_with_a_kalman_constraint(L, r, xs, us_observed, A, B, n_random=5, niter=50, rho=1):
    """
    Wrapper around _policy_fitting_with_a_kalman_constraint.
    """
    n, m = B.shape

    def evaluate_L(K):
        Kcp = cp.Variable((m, n))
        Kcp.value = K
        return L(Kcp).value

    # solve with zero initialization
    P = np.zeros((n, n))
    Q = np.zeros((n, n))
    R = np.zeros((m, m))
    K = _policy_fitting_with_a_kalman_constraint(
        L, r, xs, us_observed, A, B, P, Q, R, niter=niter, rho=rho)

    best_K = K
    best_L = evaluate_L(K)

    # run n_random random initializations; keep best
    for _ in range(n_random):
        P = 1. / np.sqrt(n) * np.random.randn(n, n)
        Q = 1. / np.sqrt(n) * np.random.randn(n, n)
        R = 1. / np.sqrt(m) * np.random.randn(m, m)
        K = _policy_fitting_with_a_kalman_constraint(L, r, xs, us_observed, A, B, P.T@P, Q.T@Q, R.T@R, niter=niter, rho=rho)
        L_K = evaluate_L(K)
        if L_K < best_L:
            best_L = L_K
            best_K = K

    return best_K
