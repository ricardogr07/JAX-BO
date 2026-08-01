from scipy.optimize import minimize
import numpy as np
import jax.numpy as jnp
from jax import lax
from typing import Callable, Tuple, Optional, List, Union


def minimize_lbfgs(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    verbose: bool = False,
    maxfun: int = 15000,
    bnds: Optional[Union[Tuple[Tuple[float, float]], List[Tuple[float, float]]]] = None,
) -> Tuple[np.ndarray, float]:
    """
    Optimize a scalar-valued function using the L-BFGS-B algorithm with numerical gradients.

    This function is suitable when your objective function does NOT return gradients.
    It will internally approximate gradients using the "2-point" finite difference method.

    Parameters
    ----------
    objective : Callable[[np.ndarray], float]
        A function that takes a 1D NumPy array `x` as input and returns a scalar float.
        It must NOT return gradients. Only `f(x)` should be returned.
        Example:
            def obj(x):
                return np.sum(x**2)

    x0 : np.ndarray
        A 1D NumPy array of shape (D,) specifying the initial guess for the parameters.
        Must be within bounds if `bnds` are provided.

    verbose : bool, optional (default=False)
        If True, prints the loss value at each optimization step.

    maxfun : int, optional (default=15000)
        Maximum number of function evaluations allowed during optimization.

    bnds : list or tuple of (float, float), optional
        Bounds for each parameter dimension.
        Must be the same length as `x0`. Each element is a (min, max) tuple.
        Example: bnds = [(0.0, 1.0), (0.0, 2.0)]

    Returns
    -------
    x_opt : np.ndarray
        The optimized input parameters that minimize the objective function.

    f_opt : float
        The scalar objective value at `x_opt`.

    Notes
    -----
    This version does NOT use gradient information from the objective.
    For differentiable models (e.g., JAX or autograd), prefer `minimize_lbfgs_grad`.
    """

    if verbose:

        def callback_fn(params):
            print(
                "Loss: {}".format(
                    objective(params)[0]
                    if isinstance(objective(params), tuple)
                    else objective(params)
                )
            )

    else:
        callback_fn = None

    result = minimize(
        objective,
        x0,
        jac="2-point",  # Approximate gradient numerically
        method="L-BFGS-B",
        bounds=bnds,
        callback=callback_fn,
        options={"maxfun": maxfun},
    )

    print(f"optimization success: {result.success}")
    print(result.message)
    print(f"nit (iterations): {result.nit}")

    return result.x, result.fun


def minimize_lbfgs_grad(
    objective: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    x0: np.ndarray,
    verbose: bool = False,
    maxfun: int = 15000,
    bnds: Optional[Union[Tuple[Tuple[float, float]], List[Tuple[float, float]]]] = None,
) -> Tuple[np.ndarray, float]:
    """
    Optimize a scalar-valued function using the L-BFGS-B algorithm with **analytic gradients**.

    This function requires your objective function to return both the loss and its gradient.

    Parameters
    ----------
    objective : Callable[[np.ndarray], Tuple[float, np.ndarray]]
        A function that takes a 1D NumPy array `x` as input and returns a tuple:
        (scalar loss, gradient array of shape (D,))
        Example:
            def obj(x):
                loss = np.sum(x**2)
                grad = 2 * x
                return loss, grad

    x0 : np.ndarray
        A 1D NumPy array of shape (D,) specifying the initial guess for the parameters.
        Must be within bounds if `bnds` are provided.

    verbose : bool, optional (default=False)
        If True, prints the loss value at each optimization step.

    maxfun : int, optional (default=15000)
        Maximum number of function evaluations allowed during optimization.

    bnds : list or tuple of (float, float), optional
        Bounds for each parameter dimension.
        Must be the same length as `x0`. Each element is a (min, max) tuple.
        Example: bnds = [(0.0, 1.0), (0.0, 2.0)]

    Returns
    -------
    x_opt : np.ndarray
        The optimized input parameters that minimize the objective function.

    f_opt : float
        The scalar objective value at `x_opt`.

    Notes
    -----
    This version is faster and more accurate when analytic gradients are available.
    It is ideal for use with JAX, autograd, or PyTorch.
    """

    if verbose:

        def callback_fn(params):
            print("Loss: {}".format(objective(params)[0]))

    else:
        callback_fn = None

    result = minimize(
        objective,
        x0,
        jac=True,  # Use analytic gradients
        method="L-BFGS-B",
        bounds=bnds,
        callback=callback_fn,
        options={"maxfun": maxfun, "gtol": 1e-8},
    )

    return result.x, result.fun


def minimize_bfgs_jax(
    value_and_grad: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
    x0: jnp.ndarray,
    maxiter: int = 500,
    gtol: float = 1e-8,
    maxls: int = 30,
    c1: float = 1e-4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Minimize an unbounded scalar function fully on device with BFGS and a
    backtracking (Armijo) line search.

    Built from `lax.while_loop` and `jax.numpy` only, so it is jit- and
    vmap-compatible and never leaves the device: the intended use is mapping
    many small optimizations (e.g. multi-start GP hyperparameter training)
    in a single compiled computation.

    The line search only ever accepts steps whose value and gradient are
    finite and satisfy the Armijo decrease condition, so the returned pair
    is always consistent: `f_opt` is the objective at `x_opt`, values are
    monotonically nonincreasing, and an objective that is non-finite at
    `x0` returns `x0` with its non-finite value for the caller to discard.
    This is why `jax.scipy.optimize.minimize(method="BFGS")` is not used
    here: on a line-search failure its final state takes the unvalidated
    trial step, so it can report an `x` inconsistent with its reported
    `fun` (observed on NLML surfaces whose optimum borders a Cholesky
    breakdown region).

    Parameters
    ----------
    value_and_grad : Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
        A function of a 1D parameter array returning (scalar value, gradient
        array of the same shape as the input). Must be traceable by jax.

    x0 : jnp.ndarray
        A 1D array of shape (D,), the initial guess.

    maxiter : int, optional (default=500)
        Maximum number of BFGS iterations.

    gtol : float, optional (default=1e-8)
        Terminate when the max-norm of the gradient drops below this value.

    maxls : int, optional (default=30)
        Maximum number of step halvings per line search; the search fails,
        ending the optimization at the current iterate, if no acceptable
        step is found within this budget.

    c1 : float, optional (default=1e-4)
        Armijo sufficient-decrease constant.

    Returns
    -------
    x_opt : jnp.ndarray
        The final iterate, shape (D,).

    f_opt : jnp.ndarray
        The scalar objective value at `x_opt`.
    """
    f0, g0 = value_and_grad(x0)
    eye = jnp.eye(x0.shape[0], dtype=x0.dtype)

    def ls_cond(carry):
        _, _, _, j, accepted, *_ = carry
        return (~accepted) & (j < maxls)

    def ls_body(carry):
        t, f_t, g_t, j, _, x, f, slope, p = carry
        f_try, g_try = value_and_grad(x + t * p)
        ok = (
            jnp.isfinite(f_try)
            & jnp.all(jnp.isfinite(g_try))
            & (f_try <= f + c1 * t * slope)
        )
        f_t = jnp.where(ok, f_try, f_t)
        g_t = jnp.where(ok, g_try, g_t)
        t_next = jnp.where(ok, t, t * 0.5)
        return (t_next, f_t, g_t, j + 1, ok, x, f, slope, p)

    def cond(state):
        *_, k, done = state
        return (~done) & (k < maxiter)

    def body(state):
        x, f, g, inv_hessian, scaled, k, done = state
        p = -inv_hessian @ g
        slope = jnp.dot(p, g)
        # A non-descent direction means the curvature approximation broke
        # down numerically; fall back to steepest descent for this step.
        descent = slope < 0
        p = jnp.where(descent, p, -g)
        slope = jnp.where(descent, slope, -jnp.dot(g, g))

        t0 = jnp.asarray(1.0, dtype=x.dtype)
        carry = (t0, f, g, 0, jnp.asarray(False), x, f, slope, p)
        t, f_new, g_new, _, accepted, *_ = lax.while_loop(ls_cond, ls_body, carry)

        s = t * p
        y = g_new - g
        ys = jnp.dot(y, s)
        # The BFGS update keeps the inverse Hessian positive definite only
        # under positive curvature; skip it otherwise (cautious BFGS).
        ok_upd = accepted & jnp.isfinite(ys) & (ys > 0)
        # Nocedal and Wright eq. 6.20: rescale the identity right before
        # the first update so the initial step sizes match the local scale.
        yy = jnp.dot(y, y)
        scale = jnp.where(scaled | ~ok_upd, 1.0, ys / jnp.where(yy > 0, yy, 1.0))
        rho = jnp.where(ok_upd, 1.0 / jnp.where(ys != 0, ys, 1.0), 0.0)
        w = eye - rho * jnp.outer(s, y)
        updated = w @ (inv_hessian * scale) @ w.T + rho * jnp.outer(s, s)
        inv_hessian = jnp.where(ok_upd, updated, inv_hessian)

        x = jnp.where(accepted, x + s, x)
        f = jnp.where(accepted, f_new, f)
        g = jnp.where(accepted, g_new, g)
        done = (~accepted) | (jnp.max(jnp.abs(g)) < gtol)
        return (x, f, g, inv_hessian, scaled | ok_upd, k + 1, done)

    state = (x0, f0, g0, eye, jnp.asarray(False), 0, ~jnp.isfinite(f0))
    x_opt, f_opt, *_ = lax.while_loop(cond, body, state)
    return x_opt, f_opt
