import numpy as onp
import jax.numpy as np
from typing import List, Dict, Any, Tuple
import numpy as onp
import numpy.typing as npt

def serializable_MF(
    opt_params_list: List[npt.NDArray],
    X_f_L: npt.NDArray,
    y_f_L: npt.NDArray,
    X_f_H: npt.NDArray,
    y_f_H: npt.NDArray,
    X_c_L_list: List[npt.NDArray],
    y_c_L_list: List[npt.NDArray],
    X_c_H_list: List[npt.NDArray],
    y_c_H_list: List[npt.NDArray],
    bounds: Dict[str, npt.NDArray],
    gmm_vars: List[npt.NDArray]
) -> List[Any]:
    """
    Converts all multi-fidelity optimization objects into JSON-serializable Python lists.

    Args:
        opt_params_list (List[np.ndarray]): List of parameter arrays (e.g. optimal GP hyperparameters).
        X_f_L (np.ndarray): Low-fidelity function inputs.
        y_f_L (np.ndarray): Low-fidelity function outputs.
        X_f_H (np.ndarray): High-fidelity function inputs.
        y_f_H (np.ndarray): High-fidelity function outputs.
        X_c_L_list (List[np.ndarray]): List of constraint inputs (low fidelity).
        y_c_L_list (List[np.ndarray]): List of constraint outputs (low fidelity).
        X_c_H_list (List[np.ndarray]): List of constraint inputs (high fidelity).
        y_c_H_list (List[np.ndarray]): List of constraint outputs (high fidelity).
        bounds (Dict[str, np.ndarray]): Dictionary with "lb" and "ub" keys for domain bounds.
        gmm_vars (List[np.ndarray]): List of GMM parameter arrays (e.g., weights, means, covariances).

    Returns:
        List: A list containing:
            - Serialized optimization parameters,
            - Function data,
            - Constraints data,
            - Domain bounds,
            - GMM variables.
    """
    # Serialize optimization parameters
    serialized_params = [p.tolist() for p in opt_params_list]

    # Serialize function observations
    serialized_data = {
        "X_f_L": X_f_L.tolist(),
        "y_f_L": y_f_L.tolist(),
        "X_f_H": X_f_H.tolist(),
        "y_f_H": y_f_H.tolist(),
    }

    # Serialize constraints
    serialized_constraints = [
        {
            "X_c_L": X_c_L_list[k].tolist(),
            "y_c_L": y_c_L_list[k].tolist(),
            "X_c_H": X_c_H_list[k].tolist(),
            "y_c_H": y_c_H_list[k].tolist()
        }
        for k in range(len(X_c_L_list))
    ]

    # Serialize bounds
    serialized_bounds = {
        "lb": bounds["lb"].tolist(),
        "ub": bounds["ub"].tolist()
    }

    # Serialize GMM parameters
    serialized_gmm_vars = [var.tolist() for var in gmm_vars]

    # Final exportable structure
    return [serialized_params, serialized_data, serialized_constraints, serialized_bounds, serialized_gmm_vars]

def deserializable_MF(serialized: List[Any]) -> Tuple[
    List[np.ndarray],  # opt_params_list
    np.ndarray,        # X_f_L
    np.ndarray,        # y_f_L
    np.ndarray,        # X_f_H
    np.ndarray,        # y_f_H
    List[np.ndarray],  # X_c_L_list
    List[np.ndarray],  # y_c_L_list
    List[np.ndarray],  # X_c_H_list
    List[np.ndarray],  # y_c_H_list
    Dict[str, np.ndarray],  # bounds
    List[np.ndarray]   # gmm_vars
]:
    """
    Deserializes a previously serialized multi-fidelity dataset and model configuration.

    Args:
        serialized (List): A list containing serialized items in the following order:
            - opt_params_list (List of lists)
            - return_data (dict with keys: X_f_L, y_f_L, X_f_H, y_f_H)
            - return_constraints (List of dicts with constraint data)
            - return_bounds (dict with 'lb' and 'ub')
            - return_gmm_vars (List of lists)

    Returns:
        Tuple containing:
            - List of optimization parameter arrays,
            - Low- and high-fidelity function data (X and y),
            - Lists of low- and high-fidelity constraint data (X and y),
            - Bounds dictionary,
            - GMM parameter arrays.
    """
    return_params, return_data, return_constraints, return_bounds, return_gmm_vars = serialized

    # Reconstruct optimization parameters
    opt_params_list = [np.array(p) for p in return_params]

    # Reconstruct function data
    X_f_L = np.array(return_data["X_f_L"])
    y_f_L = np.array(return_data["y_f_L"])
    X_f_H = np.array(return_data["X_f_H"])
    y_f_H = np.array(return_data["y_f_H"])

    # Reconstruct constraint data
    X_c_L_list = [np.array(item["X_c_L"]) for item in return_constraints]
    y_c_L_list = [np.array(item["y_c_L"]) for item in return_constraints]
    X_c_H_list = [np.array(item["X_c_H"]) for item in return_constraints]
    y_c_H_list = [np.array(item["y_c_H"]) for item in return_constraints]

    # Reconstruct bounds
    bounds = {
        "lb": np.array(return_bounds["lb"]),
        "ub": np.array(return_bounds["ub"])
    }

    # Reconstruct GMM parameters
    gmm_vars = [np.array(var) for var in return_gmm_vars]

    return (
        opt_params_list, X_f_L, y_f_L, X_f_H, y_f_H,
        X_c_L_list, y_c_L_list, X_c_H_list, y_c_H_list,
        bounds, gmm_vars
    )
