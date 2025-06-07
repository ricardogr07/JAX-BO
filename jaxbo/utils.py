import numpy as onp
import jax.numpy as np
from jax import jit, vmap, random
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Tanh
from jax.nn.initializers import glorot_normal, normal
from jax.scipy.stats import multivariate_normal
from KDEpy import FFTKDE
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

@jit
def normalize(X, y, bounds):
    """
    Normalizes input features X and target values y using provided bounds and statistics.

    Args:
        X (jax.numpy.ndarray): Input features to be normalized. Shape: (n_samples, n_features).
        y (jax.numpy.ndarray): Target values to be normalized. Shape: (n_samples,) or (n_samples, n_targets).
        bounds (dict): Dictionary containing 'lb' (lower bounds) and 'ub' (upper bounds) for each feature in X.
            - 'lb' (jax.numpy.ndarray): Lower bounds for X. Shape: (n_features,).
            - 'ub' (jax.numpy.ndarray): Upper bounds for X. Shape: (n_features,).

    Returns:
        tuple:
            - batch (dict): Dictionary containing normalized 'X' and 'y'.
            - norm_const (dict): Dictionary containing normalization constants:
                - 'mu_y': Mean of y before normalization.
                - 'sigma_y': Standard deviation of y before normalization.

    Notes:
        - X is normalized to the [0, 1] range using the provided bounds.
        - y is normalized to have zero mean and unit variance.
    """
    mu_y, sigma_y = y.mean(0), y.std(0)
    X = (X - bounds['lb'])/(bounds['ub']-bounds['lb'])
    y = (y - mu_y)/sigma_y
    batch = {'X': X, 'y': y}
    norm_const = {'mu_y': mu_y, 'sigma_y': sigma_y}
    return batch, norm_const

@jit
def normalize_MultifidelityGP(XL, yL, XH, yH, bounds):
    """
    Normalizes input and output data for a multi-fidelity Gaussian Process (GP) model.

    This function takes in low-fidelity and high-fidelity input/output pairs, along with their bounds,
    and normalizes both the input features and output targets. The inputs are scaled to the [0, 1] range
    based on the provided bounds, and the outputs are standardized to have zero mean and unit variance
    using statistics computed from the concatenated outputs.

    Args:
        XL (np.ndarray): Low-fidelity input data of shape (n_L, d).
        yL (np.ndarray): Low-fidelity output data of shape (n_L,).
        XH (np.ndarray): High-fidelity input data of shape (n_H, d).
        yH (np.ndarray): High-fidelity output data of shape (n_H,).
        bounds (dict): Dictionary with keys 'lb' and 'ub' representing lower and upper bounds
            for input normalization. Each should be an array of shape (d,).

    Returns:
        batch (dict): Dictionary containing normalized data:
            - 'XL': Normalized low-fidelity inputs.
            - 'XH': Normalized high-fidelity inputs.
            - 'y':  Normalized concatenated outputs.
            - 'yL': Normalized low-fidelity outputs.
            - 'yH': Normalized high-fidelity outputs.
        norm_const (dict): Dictionary containing normalization constants:
            - 'mu_y': Mean of concatenated outputs before normalization.
            - 'sigma_y': Standard deviation of concatenated outputs before normalization.
    """
    y = np.concatenate([yL, yH], axis = 0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    XL = (XL - bounds['lb'])/(bounds['ub']-bounds['lb'])
    XH = (XH - bounds['lb'])/(bounds['ub']-bounds['lb'])
    yL = (yL - mu_y)/sigma_y
    yH = (yH - mu_y)/sigma_y
    y = (y - mu_y)/sigma_y
    batch = {'XL': XL, 'XH': XH, 'y': y, 'yL': yL, 'yH': yH}
    norm_const = {'mu_y': mu_y, 'sigma_y': sigma_y}
    return batch, norm_const

@jit
def normalize_GradientGP(XF, yF, XG, yG):
    """
    Normalizes the inputs and outputs for a Gradient Gaussian Process (GP) model.

    Args:
        XF (array-like): Feature matrix for function observations.
        yF (array-like): Output vector for function observations.
        XG (array-like): Feature matrix for gradient observations.
        yG (array-like): Output vector for gradient observations.

    Returns:
        tuple: A tuple containing:
            - batch (dict): Dictionary with keys 'XF', 'XG', 'yF', 'yG', and 'y' (concatenated outputs).
            - norm_const (dict): Dictionary with normalization constants for inputs and outputs, 
            including 'mu_X', 'sigma_X', 'mu_y', and 'sigma_y'.
    """
    y = np.concatenate([yF, yG], axis = 0)
    batch = {'XF': XF, 'XG': XG, 'yF': yF, 'yG': yG, 'y': y}
    norm_const = {'mu_X': 0.0, 'sigma_X': 1.0,
                  'mu_y': 0.0, 'sigma_y': 1.0}
    return batch, norm_const

@jit
def normalize_HeterogeneousMultifidelityGP(XL, yL, XH, yH, bounds):
    """
    Normalizes input and output data for heterogeneous multifidelity Gaussian Process (GP) models.

    This function standardizes the low-fidelity inputs (XL) and outputs (yL), and high-fidelity outputs (yH)
    using the mean and standard deviation of the low-fidelity data. The high-fidelity inputs (XH) are normalized
    to the [0, 1] range using the provided bounds.

    Args:
        XL (np.ndarray): Low-fidelity input data of shape (n_low, d).
        yL (np.ndarray): Low-fidelity output data of shape (n_low, 1) or (n_low,).
        XH (np.ndarray): High-fidelity input data of shape (n_high, d).
        yH (np.ndarray): High-fidelity output data of shape (n_high, 1) or (n_high,).
        bounds (dict): Dictionary with keys 'lb' and 'ub' representing the lower and upper bounds for normalization
                       of high-fidelity inputs. Each should be an array of shape (d,).

    Returns:
        batch (dict): Dictionary containing normalized data with keys:
            - 'XL': Normalized low-fidelity inputs.
            - 'XH': Normalized high-fidelity inputs.
            - 'y':  Concatenated and normalized outputs.
            - 'yL': Normalized low-fidelity outputs.
            - 'yH': Normalized high-fidelity outputs.
        norm_const (dict): Dictionary containing normalization constants with keys:
            - 'mu_X': Mean of low-fidelity inputs.
            - 'sigma_X': Standard deviation of low-fidelity inputs.
            - 'mu_y': Mean of concatenated outputs.
            - 'sigma_y': Standard deviation of concatenated outputs.
    """
    y = np.concatenate([yL, yH], axis = 0)
    mu_X, sigma_X = XL.mean(0), XL.std(0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    XL = (XL - mu_X)/sigma_X
    XH = (XH - bounds['lb'])/(bounds['ub']-bounds['lb'])
    yL = (yL - mu_y)/sigma_y
    yH = (yH - mu_y)/sigma_y
    y = (y - mu_y)/sigma_y
    batch = {'XL': XL, 'XH': XH, 'y': y, 'yL': yL, 'yH': yH}
    norm_const = {'mu_X': mu_X, 'sigma_X': sigma_X,
                  'mu_y': mu_y, 'sigma_y': sigma_y}
    return batch, norm_const

@jit
def standardize(X, y):
    """
    Standardizes the input features X and target y to have zero mean and unit variance.

    Parameters
    ----------
    X : np.ndarray
        Input features, where each row is a sample and each column is a feature.
    y : np.ndarray
        Target values, can be a 1D or 2D array.

    Returns
    -------
    batch : dict
        Dictionary containing the standardized 'X' and 'y'.
    norm_const : dict
        Dictionary containing the means ('mu_X', 'mu_y') and standard deviations ('sigma_X', 'sigma_y')
        used for standardization.
    """
    mu_X, sigma_X = X.mean(0), X.std(0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    X = (X - mu_X)/sigma_X
    y = (y - mu_y)/sigma_y
    batch = {'X': X, 'y': y}
    norm_const = {'mu_X': mu_X, 'sigma_X': sigma_X,
                  'mu_y': mu_y, 'sigma_y': sigma_y}
    return batch, norm_const

@jit
def standardize_MultifidelityGP(XL, yL, XH, yH):
    """
    Standardizes input and output data for multi-fidelity Gaussian Process modeling.

    This function concatenates low-fidelity (XL, yL) and high-fidelity (XH, yH) datasets,
    computes the mean and standard deviation for both inputs and outputs, and standardizes
    each dataset accordingly. The standardized datasets and normalization constants are returned.

    Args:
        XL (np.ndarray): Low-fidelity input data of shape (n_low, d).
        yL (np.ndarray): Low-fidelity output data of shape (n_low,).
        XH (np.ndarray): High-fidelity input data of shape (n_high, d).
        yH (np.ndarray): High-fidelity output data of shape (n_high,).

    Returns:
        batch (dict): Dictionary containing standardized datasets:
            - 'XL': Standardized low-fidelity inputs.
            - 'XH': Standardized high-fidelity inputs.
            - 'y': Standardized concatenated outputs.
            - 'yL': Standardized low-fidelity outputs.
            - 'yH': Standardized high-fidelity outputs.
        norm_const (dict): Dictionary containing normalization constants:
            - 'mu_X': Mean of concatenated inputs.
            - 'sigma_X': Standard deviation of concatenated inputs.
            - 'mu_y': Mean of concatenated outputs.
            - 'sigma_y': Standard deviation of concatenated outputs.
    """
    X = np.concatenate([XL, XH], axis = 0)
    y = np.concatenate([yL, yH], axis = 0)
    mu_X, sigma_X = X.mean(0), X.std(0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    XL = (XL - mu_X)/sigma_X
    XH = (XH - mu_X)/sigma_X
    yL = (yL - mu_y)/sigma_y
    yH = (yH - mu_y)/sigma_y
    y = (y - mu_y)/sigma_y
    batch = {'XL': XL, 'XH': XH, 'y': y, 'yL': yL, 'yH': yH}
    norm_const = {'mu_X': mu_X, 'sigma_X': sigma_X,
                  'mu_y': mu_y, 'sigma_y': sigma_y}
    return batch, norm_const

@jit
def standardize_HeterogeneousMultifidelityGP(XL, yL, XH, yH):
    """
    Standardizes and normalizes input and output data for heterogeneous multifidelity Gaussian Process models.

    This function applies standardization to low-fidelity inputs (XL) and normalization to high-fidelity inputs (XH).
    The outputs (yL, yH) are standardized using the mean and standard deviation computed from the concatenated outputs.
    The function returns the standardized/normalized data and the normalization constants used.

    Args:
        XL (np.ndarray): Low-fidelity input data of shape (n_L, d).
        yL (np.ndarray): Low-fidelity output data of shape (n_L,).
        XH (np.ndarray): High-fidelity input data of shape (n_H, d).
        yH (np.ndarray): High-fidelity output data of shape (n_H,).

    Returns:
        batch (dict): Dictionary containing standardized/normalized data:
            - 'XL': Standardized low-fidelity inputs.
            - 'XH': Normalized high-fidelity inputs.
            - 'y': Standardized concatenated outputs.
            - 'yL': Standardized low-fidelity outputs.
            - 'yH': Standardized high-fidelity outputs.
        norm_const (dict): Dictionary containing normalization constants:
            - 'mu_XL': Mean of XL.
            - 'sigma_XL': Standard deviation of XL.
            - 'min_XH': Minimum of XH.
            - 'max_XH': Maximum of XH.
            - 'mu_y': Mean of concatenated outputs.
            - 'sigma_y': Standard deviation of concatenated outputs.
    """
    y = np.concatenate([yL, yH], axis = 0)
    mu_XL, sigma_XL = XL.mean(0), XL.std(0)
    min_XH, max_XH = XH.min(0), XH.max(0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    XL = (XL - mu_XL)/sigma_XL
    XH = (XH - min_XH)/(max_XH-min_XH)
    yL = (yL - mu_y)/sigma_y
    yH = (yH - mu_y)/sigma_y
    y = (y - mu_y)/sigma_y
    batch = {'XL': XL, 'XH': XH, 'y': y, 'yL': yL, 'yH': yH}
    norm_const = {'mu_XL': mu_XL, 'sigma_XL': sigma_XL,
                  'min_XH': min_XH, 'max_XH': max_XH,
                  'mu_y': mu_y, 'sigma_y': sigma_y}
    return batch, norm_const

@jit
def compute_w_gmm(x, **kwargs):
    """
    Computes the weighted sum of Gaussian Mixture Model (GMM) components at a given point.

    This function normalizes the input `x` to the unit hypercube defined by the provided bounds,
    then evaluates the weighted sum of multivariate normal probability density functions (PDFs)
    at the normalized point using the parameters of the GMM.

    Args:
        x (np.ndarray): The input point(s) at which to evaluate the GMM, shape (..., D).
        **kwargs: Additional keyword arguments, including:
            - 'bounds' (dict): Dictionary with keys 'lb' and 'ub' for lower and upper bounds (arrays).
            - 'gmm_vars' (tuple): Tuple containing (weights, means, covariances) of the GMM:
                - weights (np.ndarray): Weights of the GMM components, shape (K,).
                - means (np.ndarray): Means of the GMM components, shape (K, D).
                - covs (np.ndarray): Covariance matrices of the GMM components, shape (K, D, D).

    Returns:
        float or np.ndarray: The weighted sum of GMM component PDFs evaluated at `x`.
    """
    bounds = kwargs['bounds']
    lb = bounds['lb']
    ub = bounds['ub']
    x = (x - lb) / (ub - lb)
    weights, means, covs = kwargs['gmm_vars']
    gmm_mode = lambda w, mu, cov:  w*multivariate_normal.pdf(x, mu, cov)
    w = np.sum(vmap(gmm_mode)(weights, means, covs), axis = 0)
    return w

def fit_kernel_density(X, xi, weights=None, bw=None):
    """Fit a kernel density estimator and evaluate its PDF at ``xi``.

    Parameters
    ----------
    X : array-like
        Input data used to fit the kernel density estimator. Should be a 1D
        array.
    xi : array-like
        Points at which to evaluate the estimated PDF.
    weights : array-like, optional
        Weights for each data point in ``X``. If ``None``, equal weighting is
        assumed.
    bw : float, optional
        Bandwidth for the KDE. If ``None``, the bandwidth is estimated
        automatically.

    Returns
    -------
    pdf : ndarray
        The estimated PDF values at the points specified by ``xi``. Values are
        clipped to be non-negative and a small epsilon (``1e-8``) is added for
        numerical stability.

    Notes
    -----
    - Uses :class:`FFTKDE` for fast kernel density estimation.
    - If bandwidth estimation fails or results in a very small value, a default
      of ``1.0`` is used.
    - The output PDF is interpolated using a linear interpolation and
      extrapolated as needed.
    """

    X, weights = onp.array(X), onp.array(weights)
    X = X.flatten()
    if bw is None:
        try:
            sc = gaussian_kde(X, weights=weights)
            bw = onp.sqrt(sc.covariance).flatten()[0]
        except:
            bw = 1.0
        if bw < 1e-8:
            bw = 1.0


    kde_pdf_x, kde_pdf_y = FFTKDE(bw=bw).fit(X, weights).evaluate()

    # Define the interpolation function
    interp1d_fun = interp1d(kde_pdf_x,
                            kde_pdf_y,
                            kind = 'linear',
                            fill_value = 'extrapolate')

    # Evaluate the weights on the input data
    pdf = interp1d_fun(xi)
    return np.clip(pdf, a_min=0.0) + 1e-8

def init_NN(Q):
    """
    Initializes a feedforward neural network using the stax API.

    Args:
        Q (list or tuple of int): A sequence specifying the number of units in each layer of the network.
            The length of Q determines the number of layers, where Q[0] is the input dimension and Q[-1] is the output dimension.

    Returns:
        net_init (callable): A function to initialize the network parameters.
        net_apply (callable): A function to apply the network to inputs.

    Notes:
        - Each hidden layer uses a Dense layer followed by a Tanh activation.
        - The output layer is a Dense layer without an activation.
        - Weights are initialized using Glorot normal initialization, and biases are initialized with a normal distribution, both with dtype float64.
    """
    layers = []
    num_layers = len(Q)
    for i in range(0, num_layers-2):
        layers.append(Dense(Q[i+1],
                            W_init=glorot_normal(dtype=np.float64),
                            b_init=normal(dtype=np.float64)))
        layers.append(Tanh)
    layers.append(Dense(Q[-1],
                  W_init=glorot_normal(dtype=np.float64),
                  b_init=normal(dtype=np.float64)))
    net_init, net_apply = stax.serial(*layers)
    return net_init, net_apply

def init_ResNet(layers, depth, is_spect):
    """
    Initializes a residual neural network (ResNet) with configurable depth, layer sizes, and optional spectral normalization.
    Args:
        layers (list of int): List specifying the number of units in each layer of the network.
        depth (int): Number of residual blocks to apply in the network.
        is_spect (int): If set to 1, applies spectral normalization and normalization parameters to the network; otherwise, standard initialization is used.
    Returns:
        init (callable): A function that takes a JAX random key and returns initialized network parameters.
        apply (callable): A function that applies the ResNet to input data using the initialized parameters.
    Notes:
        - The network uses tanh activations and residual connections.
        - If `is_spect` is enabled, spectral normalization is applied to the weights, and additional normalization parameters (gamma, beta) are included.
        - The `apply` function performs normalization on the inputs if `is_spect` is enabled, otherwise applies standard residual blocks.
    """
    ''' MLP blocks with residual connections'''
    def init(rng_key):
        # Initialize neural net params
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            
            #W = random.normal(k1, (d_in, d_out))
            #b = random.normal(k2, (d_out,))
            
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev*random.normal(k1, (d_in, d_out))
            if is_spect == 1:
                W = W/np.linalg.norm(W)
            
            b = np.zeros(d_out)
            
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        if is_spect == 1:
            gamma = np.ones(layers[0])
            beta = np.zeros(layers[0])
            params.append(gamma)
            params.append(beta)
        return params
    def mlp(params, inputs):
        for W, b in params:
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs
    if is_spect == 1:
        def apply(params, inputs):
            inputs = params[-2]/np.sqrt(np.var(inputs, axis=0))*(inputs-np.mean(inputs, axis=0))+params[-1]
            for i in range(depth):
                #outputs = mlp(params, inputs) + inputs
                inputs = mlp(params[:-2], inputs) + inputs
            return inputs
    else:
        def apply(params, inputs):
            for i in range(depth):
                inputs = mlp(params, inputs) + inputs
            return inputs
    return init, apply

def init_MomentumResNet(layers, depth, vel_zeros=0, gamma=0.9):
    """
    Initializes a MomentumResNet, a multi-layer perceptron (MLP) with residual connections and momentum-based updates.

    Args:
        layers (list of int): List specifying the number of units in each layer of the MLP.
        depth (int): Number of residual/momentum update steps to apply.
        vel_zeros (int, optional): If 1, initializes the velocity vector to zeros; otherwise, initializes using the MLP output. Default is 0.
        gamma (float, optional): Momentum parameter controlling the contribution of previous velocity. Default is 0.9.

    Returns:
        init (callable): Function that takes a JAX PRNG key and returns initialized network parameters.
        apply (callable): Function that applies the MomentumResNet to input data, given parameters and inputs.

    Notes:
        - The network uses tanh activations in each layer.
        - The residual connection is implemented via a velocity vector updated with momentum.
        - The apply function's behavior depends on the value of `vel_zeros`.
    """
    ''' MLP blocks with residual connections'''
    def init(rng_key):
        # Initialize neural net params
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W = random.normal(k1, (d_in, d_out))
            b = random.normal(k2, (d_out,))
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params
    def mlp(params, inputs):
        for W, b in params:
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs
    if vel_zeros == 1:
        def apply(params, inputs):
            velocity = np.zeros_like(inputs)
            for i in range(depth):
                velocity = gamma*velocity + (1.0-gamma)*mlp(params, inputs)
                inputs = inputs + velocity
            return inputs
    else:
        def apply(params, inputs):
            velocity = mlp(params, inputs)
            for i in range(depth):
                velocity = gamma*velocity + (1.0-gamma)*mlp(params, inputs)
                inputs = inputs + velocity
            return inputs
    return init, apply
