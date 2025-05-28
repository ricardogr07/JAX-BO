from scipy.optimize import minimize
import numpy as np
from typing import Callable, Tuple, Optional, List, Union

def minimize_lbfgs(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    verbose: bool = False,
    maxfun: int = 15000,
    bnds: Optional[Union[Tuple[Tuple[float, float]], List[Tuple[float, float]]]] = None
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
            print("Loss: {}".format(objective(params)[0] if isinstance(objective(params), tuple) else objective(params)))
    else:
        callback_fn = None

    result = minimize(
        objective,
        x0,
        jac="2-point",  # Approximate gradient numerically
        method='L-BFGS-B',
        bounds=bnds,
        callback=callback_fn,
        options={'maxfun': maxfun}
    )

    print(f"optimization success: {result.success}")
    print(result.message)
    print(f"nit (iterations): {result.nit}")

    return result.x, result.fun


from scipy.optimize import minimize
import numpy as np
from typing import Callable, Tuple, Optional, List, Union

def minimize_lbfgs_grad(
    objective: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    x0: np.ndarray,
    verbose: bool = False,
    maxfun: int = 15000,
    bnds: Optional[Union[Tuple[Tuple[float, float]], List[Tuple[float, float]]]] = None
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
        method='L-BFGS-B',
        bounds=bnds,
        callback=callback_fn,
        options={'maxfun': maxfun, 'gtol': 1e-8}
    )

    return result.x, result.fun

