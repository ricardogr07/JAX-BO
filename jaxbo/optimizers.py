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
    maxiter: int = 100,
    gtol: Optional[float] = None,
    ftol: Optional[float] = None,
    maxls: int = 5,
    c1: float = 1e-4,
    c2: float = 0.9,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Minimize an unbounded scalar function fully on device with BFGS and a
    strong Wolfe line search (bracketing by doubling, then bisection zoom).

    Built from `lax.while_loop` and `jax.numpy` only, so it is jit- and
    vmap-compatible and never leaves the device: the intended use is mapping
    many small optimizations (e.g. multi-start GP hyperparameter training)
    in a single compiled computation.

    The line search only ever accepts steps whose value and gradient are
    finite and satisfy at least the Armijo decrease condition, so the
    returned pair is always consistent: `f_opt` is the objective at
    `x_opt` and values are monotonically nonincreasing. The one deliberate
    exception is a run that cannot start: if either the value or the
    gradient is non-finite at `x0`, the run returns `x0` with a non-finite
    score for the caller to discard, even when the value alone was finite
    (a non-finite gradient leaves the line search unable to accept any
    step, so such a lane has optimized nothing and must not compete for
    selection on the strength of its starting value). This is why
    `jax.scipy.optimize.minimize(method="BFGS")` is not used here: on a
    line-search failure its final state takes the unvalidated trial step,
    so it can report an `x` inconsistent with its reported `fun` (observed
    on NLML surfaces whose optimum borders a Cholesky breakdown region).

    Two ingredients make the search basin-robust on multi-modal surfaces
    (platform-level rounding differences must not flip which optimum a
    seeded restart converges to):

    - The strong Wolfe curvature condition |phi'(t)| <= c2 * |phi'(0)|
      rejects trial points where the objective is still steep, so a step
      cannot dive down a cliff into a distant basin merely because the
      value there happens to satisfy the decrease condition; it also keeps
      every inverse-Hessian update well-scaled. When no trial point can
      certify curvature within the budgets, the run ends at the current
      iterate, matching scipy L-BFGS-B's ABNORMAL_TERMINATION_IN_LNSRCH
      semantics: that situation means the objective is locally linear
      along the direction (an improper ridge, e.g. the zero-noise ridge
      of a noiseless GP likelihood, or the dtype's resolution floor), and
      continuing to harvest decrease there converges to improper optima
      the reference implementation never reaches.
    - The first trial step of the run is gradient-scaled,
      t = min(1, 1 / ||g0||), so a steep initialization takes a unit-length
      first move instead of a ||g0||-length leap; after the first
      inverse-Hessian update the natural step t = 1 is well-scaled (see
      the Nocedal and Wright eq. 6.20 rescale below).

    Parameters
    ----------
    value_and_grad : Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
        A function of a 1D parameter array returning (scalar value, gradient
        array of the same shape as the input). Must be traceable by jax.

    x0 : jnp.ndarray
        A 1D array of shape (D,), the initial guess.

    maxiter : int, optional (default=100)
        Maximum number of BFGS iterations. BFGS converges superlinearly on
        the tiny smooth problems this solver targets (GP hyperparameter
        surfaces of dimension D + 2 finish in under 25 iterations in
        float64, and the ftol floor ends float32 runs well before 100), so
        100 is generous headroom whose real job is bounding the cost of a
        pathological run: under vmap every lane steps until the slowest
        lane terminates.

    gtol : float, optional (default=None)
        Terminate when the max-norm of the gradient drops below this value.
        Defaults to sqrt(eps) of the parameter dtype: about 1.5e-8 in
        float64 (matching the 1e-8 the scipy L-BFGS-B path used) and about
        3.5e-4 in float32, where gradient norms cannot reach float64-scale
        tolerances.

    ftol : float, optional (default=None)
        Terminate when an accepted step's decrease satisfies
        (f_k - f_{k+1}) <= ftol * max(|f_k|, |f_{k+1}|, 1), the same
        criterion scipy's L-BFGS-B applies. Defaults to 1e5 * eps of the
        parameter dtype (factr semantics): about 2.2e-11 in float64,
        tighter than scipy's default 2.2e-9, and about 1.2e-2 in float32.
        This floor is what terminates float32 runs: near the optimum the
        Armijo threshold `f + c1 * t * slope` rounds to `f`, so zero
        progress steps keep being accepted and, without this criterion,
        the loop would spin until maxiter.

    maxls : int, optional (default=5)
        Maximum number of bisection steps in the zoom phase of the line
        search. Deliberately small: in a curved basin the zoom certifies a
        strong Wolfe point within a few bisections, while on flat or
        rounding-noise stretches no budget certifies one and every extra
        bisection is a wasted full objective evaluation (measured 3x train
        wall time at 512 training points between 5 and 30, with identical
        selected optima). When the budgets end without a certified point,
        the search fails and the optimization ends at the current iterate.

    c1 : float, optional (default=1e-4)
        Armijo sufficient-decrease constant.

    c2 : float, optional (default=0.9)
        Strong Wolfe curvature constant (0.9 is the standard quasi-Newton
        choice, matching scipy).

    Returns
    -------
    x_opt : jnp.ndarray
        The final iterate, shape (D,).

    f_opt : jnp.ndarray
        The scalar objective value at `x_opt`.
    """
    eps = float(jnp.finfo(jnp.asarray(x0).dtype).eps)
    if gtol is None:
        gtol = eps**0.5
    if ftol is None:
        # float64: factr 1e7, exactly scipy L-BFGS-B's default, so runs
        # stop at scipy-comparable depth (tighter riding of improper NLML
        # ridges is worse, not better; see the GP gap fixture). float32:
        # 1e7 ULPs exceeds the 23-bit mantissa, so the parity constant is
        # meaningless there; 1e5 is the measured precision floor.
        ftol = (1e7 if eps < 1e-10 else 1e5) * eps

    f0, g0 = value_and_grad(x0)
    # A run that cannot start is reported with a non-finite score so the
    # caller discards the lane. The gradient half matters as much as the
    # value half: with a finite f0 and a non-finite g0 the search direction
    # is non-finite, every trial fails `usable`, and the run would otherwise
    # return x0 with its finite f0, which `GP.train`'s nanargmin would then
    # accept as a converged restart (it is the true objective at x0, but
    # nothing was optimized).
    bad_start = ~jnp.isfinite(f0) | ~jnp.all(jnp.isfinite(g0))
    f0 = jnp.where(bad_start, jnp.nan, f0)
    eye = jnp.eye(x0.shape[0], dtype=x0.dtype)
    # Doubling budget of the bracket phase. Small on purpose: bracketing
    # exists to correct the step SCALE (a few octaves), not to traverse
    # flat improper ridges of the objective, where unbounded growth lets
    # a single line search harvest gains that keep the outer ftol test
    # from ever firing (scipy's bounded per-iteration travel is what made
    # the L-BFGS-B path stop at sane optima on such surfaces).
    n_bracket = 3

    def line_search(x, f, g, p, slope, t_init):
        """Strong Wolfe search: bracket by doubling, then bisection zoom.

        Returns (accepted, t, f_t, g_t). Invariant: the returned point
        satisfies at least Armijo with finite value and gradient, or
        accepted is False and the caller keeps the current iterate.
        """

        def eval_t(t):
            f_t, g_t = value_and_grad(x + t * p)
            return f_t, g_t, jnp.dot(g_t, p)

        def usable(t, phi, grad):
            # Finite and Armijo: a non-finite trial is simply "too far"
            # and gets bracketed away, which is how the scipy path
            # recovered from NaN regions.
            return (
                jnp.isfinite(phi)
                & jnp.all(jnp.isfinite(grad))
                & (phi <= f + c1 * t * slope)
            )

        wolfe_thresh = -c2 * slope
        zero_t = jnp.zeros_like(t_init)

        def b_cond(carry):
            j, wolfe, brack = carry[0], carry[1], carry[2]
            return (~wolfe) & (~brack) & (j < n_bracket)

        def b_body(carry):
            (
                j,
                wolfe,
                brack,
                t_lo,
                phi_lo,
                dphi_lo,
                g_lo,
                t_hi,
                phi_hi,
                t,
                t_st,
                phi_st,
                g_st,
            ) = carry
            phi_t, g_t, dphi_t = eval_t(t)
            to_hi = (~usable(t, phi_t, g_t)) | ((phi_t >= phi_lo) & (j > 0))
            wolfe = (~to_hi) & (jnp.abs(dphi_t) <= wolfe_thresh)
            pos = (~to_hi) & (~wolfe) & (dphi_t >= 0)
            grow = (~to_hi) & (~wolfe) & (~pos)
            brack = to_hi | pos
            # Bracket is [lo, t] when the trial is unusable or higher than
            # lo, and [t, old lo] when the slope turned positive (the old
            # lo values move to hi before lo is overwritten).
            t_hi = jnp.where(to_hi, t, jnp.where(pos, t_lo, t_hi))
            phi_hi = jnp.where(to_hi, phi_t, jnp.where(pos, phi_lo, phi_hi))
            move = pos | grow
            g_lo = jnp.where(move, g_t, g_lo)
            t_lo = jnp.where(move, t, t_lo)
            phi_lo = jnp.where(move, phi_t, phi_lo)
            dphi_lo = jnp.where(move, dphi_t, dphi_lo)
            g_st = jnp.where(wolfe, g_t, g_st)
            t_st = jnp.where(wolfe, t, t_st)
            phi_st = jnp.where(wolfe, phi_t, phi_st)
            t = jnp.where(grow, 2.0 * t, t)
            return (
                j + 1,
                wolfe,
                brack,
                t_lo,
                phi_lo,
                dphi_lo,
                g_lo,
                t_hi,
                phi_hi,
                t,
                t_st,
                phi_st,
                g_st,
            )

        carry = (
            0,
            jnp.asarray(False),
            jnp.asarray(False),
            zero_t,
            f,
            slope,
            g,
            zero_t,
            f,
            t_init,
            zero_t,
            f,
            g,
        )
        (
            _,
            wolfe,
            brack,
            t_lo,
            phi_lo,
            dphi_lo,
            g_lo,
            t_hi,
            phi_hi,
            _,
            t_st,
            phi_st,
            g_st,
        ) = lax.while_loop(b_cond, b_body, carry)

        def z_cond(carry):
            k, wolfe, dead = carry[0], carry[1], carry[2]
            return brack & (~wolfe) & (~dead) & (k < maxls)

        def z_body(carry):
            (
                k,
                wolfe,
                dead,
                t_lo,
                phi_lo,
                dphi_lo,
                g_lo,
                t_hi,
                phi_hi,
                t_st,
                phi_st,
                g_st,
            ) = carry
            t_m = 0.5 * (t_lo + t_hi)
            phi_m, g_m, dphi_m = eval_t(t_m)
            to_hi = (~usable(t_m, phi_m, g_m)) | (phi_m >= phi_lo)
            wolfe = (~to_hi) & (jnp.abs(dphi_m) <= wolfe_thresh)
            flip = (~to_hi) & (~wolfe) & (dphi_m * (t_hi - t_lo) >= 0)
            to_lo = (~to_hi) & (~wolfe)
            # Flat at working precision across lo, mid, and hi: no
            # representable decrease is left in the bracket, so stop
            # instead of bisecting the budget away (this is the float32
            # endgame; the caller's ftol floor then ends the run).
            dead = (phi_m == phi_lo) & (phi_m == phi_hi)
            g_st = jnp.where(wolfe, g_m, g_st)
            t_st = jnp.where(wolfe, t_m, t_st)
            phi_st = jnp.where(wolfe, phi_m, phi_st)
            t_hi_new = jnp.where(to_hi, t_m, jnp.where(flip, t_lo, t_hi))
            phi_hi_new = jnp.where(to_hi, phi_m, jnp.where(flip, phi_lo, phi_hi))
            g_lo = jnp.where(to_lo, g_m, g_lo)
            t_lo = jnp.where(to_lo, t_m, t_lo)
            phi_lo = jnp.where(to_lo, phi_m, phi_lo)
            dphi_lo = jnp.where(to_lo, dphi_m, dphi_lo)
            return (
                k + 1,
                wolfe,
                dead,
                t_lo,
                phi_lo,
                dphi_lo,
                g_lo,
                t_hi_new,
                phi_hi_new,
                t_st,
                phi_st,
                g_st,
            )

        zoom_carry = (
            0,
            wolfe,
            jnp.asarray(False),
            t_lo,
            phi_lo,
            dphi_lo,
            g_lo,
            t_hi,
            phi_hi,
            t_st,
            phi_st,
            g_st,
        )
        (_, wolfe, _, t_lo, phi_lo, _, g_lo, _, _, t_st, phi_st, g_st) = lax.while_loop(
            z_cond, z_body, zoom_carry
        )

        # Only a certified strong Wolfe point is accepted. No fallback to
        # the best Armijo point on purpose: failing to certify curvature
        # within budget means the objective is locally linear along the
        # direction (an improper ridge, e.g. the zero-noise ridge of a
        # noiseless GP likelihood, or the dtype's resolution floor), and
        # scipy's L-BFGS-B stops the whole run there
        # (ABNORMAL_TERMINATION_IN_LNSRCH). An Armijo fallback would keep
        # harvesting ridge decrease scipy never harvests and converge to
        # improper optima the reference path never reaches. Non-finite
        # trials are not fatal by themselves: like dcsrch, the search
        # brackets away from them and may still certify a point at a
        # smaller step.
        return wolfe, t_st, phi_st, g_st

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

        one = jnp.asarray(1.0, dtype=x.dtype)
        gnorm = jnp.sqrt(jnp.dot(g, g))
        # Gradient-scaled first trial step until the first inverse-Hessian
        # update lands (see the docstring's basin-robustness note).
        t_init = jnp.where(
            scaled, one, jnp.minimum(one, one / jnp.maximum(gnorm, one * eps))
        )
        accepted, t, f_new, g_new = line_search(x, f, g, p, slope, t_init)

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

        f_prev = f
        x = jnp.where(accepted, x + s, x)
        f = jnp.where(accepted, f_new, f)
        g = jnp.where(accepted, g_new, g)
        # Three exits: line search out of acceptable steps, relative
        # function decrease below the ftol floor (the criterion that ends
        # float32 runs), or gradient below gtol.
        floor = ftol * jnp.maximum(jnp.maximum(jnp.abs(f_prev), jnp.abs(f)), 1.0)
        done = (~accepted) | (f_prev - f <= floor) | (jnp.max(jnp.abs(g)) < gtol)
        return (x, f, g, inv_hessian, scaled | ok_upd, k + 1, done)

    state = (x0, f0, g0, eye, jnp.asarray(False), 0, bad_start)
    x_opt, f_opt, *_ = lax.while_loop(cond, body, state)
    return x_opt, f_opt
