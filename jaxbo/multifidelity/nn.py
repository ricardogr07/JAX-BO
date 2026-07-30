"""Neural network feature maps for the deep and manifold multifidelity models.

Holds the ``jax.example_libraries.stax`` surface (MLP, ResNet, and
MomentumResNet initializers) that warps inputs before the GP kernel. It
belongs to the ``[multifidelity]`` extra (SCOPE.md section 3), so the jaxbo
core never imports it eagerly; the historical ``jaxbo.utils.init_*`` paths
keep working through a lazy forward.
"""

from typing import Callable, List, Sequence, Tuple

import jax.numpy as np
from jax import random
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Tanh
from jax.nn.initializers import glorot_normal, normal


def init_NN(Q: Sequence[int]) -> Tuple[Callable, Callable]:
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
    for i in range(0, num_layers - 2):
        layers.append(
            Dense(
                Q[i + 1],
                W_init=glorot_normal(dtype=np.float64),
                b_init=normal(dtype=np.float64),
            )
        )
        layers.append(Tanh)
    layers.append(
        Dense(
            Q[-1],
            W_init=glorot_normal(dtype=np.float64),
            b_init=normal(dtype=np.float64),
        )
    )
    net_init, net_apply = stax.serial(*layers)
    return net_init, net_apply


def init_ResNet(
    layers: List[int], depth: int, is_spect: int
) -> Tuple[Callable, Callable]:
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

    def init(rng_key):
        """Initialize the ResNet parameters from a PRNG key."""

        def init_layer(key, d_in, d_out):
            """Draw one layer's Glorot-scaled weights and zero bias."""
            k1, k2 = random.split(key)

            glorot_stddev = 1.0 / np.sqrt((d_in + d_out) / 2.0)
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            if is_spect == 1:
                W = W / np.linalg.norm(W)

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
        """One tanh MLP pass over the parameter list."""
        for W, b in params:
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs

    if is_spect == 1:

        def apply(params, inputs):
            """Apply the spectrally normalized residual blocks."""
            inputs = (
                params[-2]
                / np.sqrt(np.var(inputs, axis=0))
                * (inputs - np.mean(inputs, axis=0))
                + params[-1]
            )
            for i in range(depth):
                # outputs = mlp(params, inputs) + inputs
                inputs = mlp(params[:-2], inputs) + inputs
            return inputs

    else:

        def apply(params, inputs):
            """Apply the residual blocks."""
            for i in range(depth):
                inputs = mlp(params, inputs) + inputs
            return inputs

    return init, apply


def init_MomentumResNet(
    layers: List[int], depth: int, vel_zeros: int = 0, gamma: float = 0.9
) -> Tuple[Callable, Callable]:
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

    def init(rng_key):
        """Initialize the MomentumResNet parameters from a PRNG key."""

        def init_layer(key, d_in, d_out):
            """Draw one layer's normal weights and bias."""
            k1, k2 = random.split(key)
            W = random.normal(k1, (d_in, d_out))
            b = random.normal(k2, (d_out,))
            return W, b

        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params

    def mlp(params, inputs):
        """One tanh MLP pass over the parameter list."""
        for W, b in params:
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs

    if vel_zeros == 1:

        def apply(params, inputs):
            """Apply momentum updates from a zero-initialized velocity."""
            velocity = np.zeros_like(inputs)
            for i in range(depth):
                velocity = gamma * velocity + (1.0 - gamma) * mlp(params, inputs)
                inputs = inputs + velocity
            return inputs

    else:

        def apply(params, inputs):
            """Apply momentum updates from an MLP-initialized velocity."""
            velocity = mlp(params, inputs)
            for i in range(depth):
                velocity = gamma * velocity + (1.0 - gamma) * mlp(params, inputs)
                inputs = inputs + velocity
            return inputs

    return init, apply
