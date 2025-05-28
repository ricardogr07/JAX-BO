import jax.numpy as np
import jax.random as random
from jax import vmap, jit
from jax.scipy.linalg import cholesky, solve_triangular
from jax.scipy.special import expit as sigmoid

from jaxbo.models import GPmodel
import jaxbo.kernels as kernels

from numpyro import sample, deterministic, handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from functools import partial
from typing import Any, Dict, Callable

class MCMCmodel(GPmodel):
    """
    Base class for MCMC-based models, inheriting from GPmodel.
    Provides methods for training with HMC/NUTS and making predictions using posterior samples.
    """

    def __init__(self, options: Dict[str, Any]):
        """
        Initialize the MCMCmodel.

        Args:
            options (Dict[str, Any]): Model options.
        """
        super().__init__(options)

    def train(
        self,
        batch: Dict[str, np.ndarray],
        rng_key: random.PRNGKey,
        settings: Dict[str, Any],
        verbose: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Run MCMC inference using NUTS.

        Args:
            batch (Dict[str, np.ndarray]): Training data batch.
            rng_key (jax.random.PRNGKey): Random key for JAX.
            settings (Dict[str, Any]): MCMC settings (e.g., num_samples, num_warmup).
            verbose (bool): If True, print MCMC summary.

        Returns:
            Dict[str, np.ndarray]: Posterior samples.
        """
        kernel = NUTS(self.model, target_accept_prob=settings['target_accept_prob'])
        mcmc = MCMC(
            kernel,
            num_warmup=settings['num_warmup'],
            num_samples=settings['num_samples'],
            num_chains=settings['num_chains'],
            progress_bar=True,
            jit_model_args=True
        )
        mcmc.run(rng_key, batch)
        if verbose:
            mcmc.print_summary()
        return mcmc.get_samples()

    @partial(jit, static_argnums=(0,))
    def predict(
        self,
        X_star: np.ndarray,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions at new input locations using posterior samples.

        Args:
            X_star (np.ndarray): Test input locations.
            **kwargs: Additional arguments, must include:
                - bounds (Dict[str, np.ndarray]): Normalization bounds.
                - rng_keys (np.ndarray): Array of random keys for sampling.
                - samples (Any): Posterior samples.
                - Any other arguments required by posterior_sample.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean and standard deviation of predictions.
        """
        # Normalize to [0,1]
        bounds = kwargs['bounds']
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        # Vectorized predictions
        rng_keys = kwargs['rng_keys']
        samples = kwargs['samples']
        sample_fn = lambda key, sample: self.posterior_sample(key, sample, X_star, **kwargs)
        means, predictions = vmap(sample_fn)(rng_keys, samples)
        mean_prediction = np.mean(means, axis=0)
        std_prediction = np.std(predictions, axis=0)
        return mean_prediction, std_prediction

class GP(MCMCmodel):
    """
    Gaussian Process (GP) model for MCMC-based Bayesian inference.
    This class implements a Gaussian Process model with log-normal priors over kernel parameters and noise,
    suitable for use with Markov Chain Monte Carlo (MCMC) methods. It provides methods for defining the model,
    computing the Cholesky decomposition of the kernel matrix, and drawing posterior predictive samples.
    Args:
        options (dict): Configuration options for the GP model.
    Methods:
        model(batch: dict) -> None:
            Defines the probabilistic model for the GP, including priors and likelihood.
            Args:
                batch (dict): Dictionary containing training data with keys 'X' (inputs, shape [N, D])
                              and 'y' (targets, shape [N]).
        compute_cholesky(params: np.ndarray, batch: dict) -> np.ndarray:
            Computes the lower-triangular Cholesky decomposition of the kernel matrix.
            Args:
                params (np.ndarray): Array of kernel and noise parameters.
                batch (dict): Dictionary containing training data with key 'X'.
            Returns:
                np.ndarray: Lower-triangular Cholesky factor of the kernel matrix.
        posterior_sample(
            key: jax.random.PRNGKey,
            sample: dict,
            X_star: np.ndarray,
            **kwargs
        ) -> tuple[np.ndarray, np.ndarray]:
            Draws a sample from the GP posterior predictive distribution at test points.
            Args:
                key (jax.random.PRNGKey): JAX random key for sampling.
                sample (dict): Dictionary of sampled kernel and noise parameters.
                X_star (np.ndarray): Test input locations, shape [N*, D].
                **kwargs: Additional arguments, must include:
                    - 'norm_const' (dict): Normalization constants with keys 'mu_y' and 'sigma_y'.
                    - 'batch' (dict): Training data with keys 'X' and 'y'.
            Returns:
                tuple[np.ndarray, np.ndarray]: Predictive mean and sampled function values at X_star.
    """
    # Initialize the class
    def __init__(self, options):
        super().__init__(options)

    def model(self, batch):
        """
        Defines a probabilistic model for Gaussian Process regression using log-normal priors on kernel variance, kernel lengthscales, and noise variance.

        Args:
            batch (dict): A dictionary containing:
                - 'X' (np.ndarray): Input features of shape (N, D).
                - 'y' (np.ndarray): Observed targets of shape (N,).

        Model Details:
            - Places log-normal priors on the kernel variance (`kernel_var`), kernel lengthscales (`kernel_length`), and noise variance (`noise_var`).
            - Constructs the kernel matrix using the provided kernel function and sampled hyperparameters.
            - Adds noise variance and a small jitter (1e-8) to the diagonal for numerical stability.
            - Observes the targets `y` under a multivariate normal distribution with zero mean and the computed covariance matrix.

        Returns:
            None. The function is intended for use within a probabilistic programming framework (e.g., NumPyro) to define the model structure.
        """
        X = batch['X']
        y = batch['y']
        N, D = X.shape
        # set uninformative log-normal priors
        var = sample('kernel_var', dist.LogNormal(0.0, 10.0))
        length = sample('kernel_length', dist.LogNormal(np.zeros(D), 10.0*np.ones(D)))
        noise = sample('noise_var', dist.LogNormal(0.0, 10.0))
        theta = np.concatenate([np.array([var]), np.array(length)])
        # compute kernel
        K = self.kernel(X, X, theta) + np.eye(N)*(noise + 1e-8)
        # sample Y according to the standard gaussian process formula
        sample("y", dist.MultivariateNormal(loc=np.zeros(N), covariance_matrix=K), obs=y)

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        """
        Computes the Cholesky decomposition of the kernel matrix for a given batch of data.

        Args:
            params (array-like): Model parameters, where the last element is the noise variance (sigma_n)
                and the preceding elements are kernel hyperparameters (theta).
            batch (dict): A dictionary containing the batch data with key 'X' representing the input data
                of shape (N, D), where N is the number of data points and D is the input dimension.

        Returns:
            ndarray: The lower-triangular Cholesky factor (L) of the kernel matrix K, where
                K = kernel(X, X, theta) + (sigma_n + 1e-8) * I_N.

        Notes:
            - The kernel matrix is regularized by adding a small jitter (1e-8) to the diagonal for numerical stability.
            - This method is JIT-compiled with the first argument (self) as a static argument.
        """
        X = batch['X']
        N, D = X.shape
        # Fetch params
        sigma_n = params[-1]
        theta = params[:-1]
        # Compute kernel
        K = self.kernel(X, X, theta) + np.eye(N)*(sigma_n + 1e-8)
        L = cholesky(K, lower=True)
        return L

    @partial(jit, static_argnums=(0,))
    def posterior_sample(self, key, sample, X_star, **kwargs):
        """
        Draws a sample from the posterior predictive distribution of a Gaussian Process at new input locations.

        Args:
            key: A JAX PRNG key for random number generation.
            sample (dict): Dictionary containing sampled kernel hyperparameters with keys:
                - 'kernel_var': Kernel variance (float).
                - 'kernel_length': Kernel lengthscale(s) (array-like).
                - 'noise_var': Observation noise variance (float).
            X_star (np.ndarray): Test input locations of shape (n_star, d).
            **kwargs: Additional keyword arguments, including:
                - 'norm_const' (dict): Normalization constants with keys 'mu_y' and 'sigma_y'.
                - 'batch' (dict): Training data with keys 'X' (inputs) and 'y' (targets).

        Returns:
            tuple:
                mu (np.ndarray): Posterior predictive mean at X_star, denormalized.
                sample (np.ndarray): Posterior predictive sample at X_star, denormalized.
        """
        # Fetch training data
        norm_const = kwargs['norm_const']
        batch = kwargs['batch']
        X, y = batch['X'], batch['y']
        # Fetch params
        var = sample['kernel_var']
        length = sample['kernel_length']
        noise = sample['noise_var']
        params = np.concatenate([np.array([var]), np.array(length), np.array([noise])])
        theta = params[:-1]
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0])*(noise + 1e-8)
        k_pX = self.kernel(X_star, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta  = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        sample = mu + std * random.normal(key, mu.shape)
        mu = mu*norm_const['sigma_y'] + norm_const['mu_y']
        sample = sample*norm_const['sigma_y'] + norm_const['mu_y']
        return mu, sample

class GPclassifier(MCMCmodel):
    """
    Gaussian Process Classifier using MCMC inference.
    This class implements a Gaussian Process (GP) classifier with log-normal priors on kernel parameters,
    and a Bernoulli likelihood for binary classification. The model is designed for use with MCMC sampling
    frameworks and supports posterior predictive sampling.
    Args:
        options (dict): Configuration options for the model and MCMC sampling.
    Methods:
        model(batch): Defines the probabilistic model for the GP classifier.
        posterior_sample(key, sample, X_star, **kwargs): Draws samples from the posterior predictive distribution.
    """

    def __init__(self, options):
        """
        Initialize the GPclassifier.
        Args:
            options (dict): Configuration options for the model and MCMC sampling.
        """
        """
        Defines the probabilistic model for the GP classifier.
        Sets log-normal priors on the kernel variance and lengthscales, constructs the GP kernel,
        samples latent function values, and specifies a Bernoulli likelihood for observed labels.
        Args:
            batch (dict): Dictionary containing training data with keys:
                - 'X': Input features, shape (N, D)
                - 'y': Binary labels, shape (N,)
        """
        super().__init__(options)

    def model(self, batch):
        """
        Defines a probabilistic model for Bayesian inference using Gaussian Processes with a Bernoulli likelihood.
        Args:
            batch (dict): A dictionary containing:
                - 'X' (np.ndarray): Input features of shape (N, D).
                - 'y' (np.ndarray): Binary target labels of shape (N,).
        Model Details:
            - Places log-normal priors on the kernel variance ('kernel_var') and lengthscales ('kernel_length').
            - Constructs the kernel matrix using the provided kernel function and adds jitter for numerical stability.
            - Samples latent function values using a standard normal prior for 'beta' (intercept) and 'eta' (latent variables).
            - Computes the latent function 'f' as a linear combination of the Cholesky factor of the kernel and 'eta', plus 'beta'.
            - Observes the binary targets 'y' under a Bernoulli likelihood parameterized by the logits 'f'.
        """
        
        X = batch['X']
        y = batch['y']
        N, D = X.shape
        # set uninformative log-normal priors
        var = sample('kernel_var', dist.LogNormal(0.0, 1.0), sample_shape = (1,))
        length = sample('kernel_length', dist.LogNormal(0.0, 1.0), sample_shape = (D,))
        theta = np.concatenate([var, length])
        # compute kernel
        K = self.kernel(X, X, theta) + np.eye(N)*1e-8
        L = cholesky(K, lower=True)
        # Generate latent function
        beta = sample('beta', dist.Normal(0.0, 1.0))
        eta = sample('eta', dist.Normal(0.0, 1.0), sample_shape=(N,))
        f = np.matmul(L, eta) + beta
        # Bernoulli likelihood
        sample('y', dist.Bernoulli(logits=f), obs=y)

    @partial(jit, static_argnums=(0,))
    def posterior_sample(self, key, sample, X_star, **kwargs):
        """
        Draws samples from the posterior predictive distribution at new input locations.
        Computes the predictive mean and samples from the posterior GP at the test points X_star,
        given MCMC samples of the model parameters and latent variables.
        Args:
            key: JAX PRNG key for randomness.
            sample (dict): Dictionary of sampled model parameters and latent variables.
            X_star (array): Test input locations, shape (N*, D).
            **kwargs: Additional keyword arguments, must include:
                - 'batch': Dictionary with training data ('X').
        Returns:
            mu (array): Predictive mean at X_star.
            sample (array): Posterior predictive sample at X_star.
        """
        # Fetch training data
        batch = kwargs['batch']
        X = batch['X']
        # Fetch params
        var = sample['kernel_var']
        length = sample['kernel_length']
        beta = sample['beta']
        eta = sample['eta']
        theta = np.concatenate([var, length])
        # Compute kernels
        K_xx = self.kernel(X, X, theta) + np.eye(X.shape[0])*1e-8
        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0])*1e-8
        k_pX = self.kernel(X_star, X, theta)
        L = cholesky(K_xx, lower=True)
        f = np.matmul(L, eta) + beta
        tmp_1 = solve_triangular(L.T,solve_triangular(L, f, lower=True))
        tmp_2  = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean
        mu = np.matmul(k_pX, tmp_1)
        cov = k_pp - np.matmul(k_pX, tmp_2)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        sample = mu + std * random.normal(key, mu.shape)
        return mu, sample

class MultifidelityGPclassifier(MCMCmodel):
    """
    MultifidelityGPclassifier implements a multi-fidelity Gaussian Process (GP) classifier using MCMC inference.
    This model supports classification tasks where data is available at multiple fidelity levels (e.g., low- and high-fidelity observations).
    It defines a hierarchical GP prior over latent functions, with separate kernels for each fidelity and a correlation parameter (rho) linking them.
    The model uses log-normal priors for kernel hyperparameters and a normal prior for the fidelity correlation.
    Args:
        options (dict): Configuration options for the model and MCMC inference.
    Methods:
        model(batch):
            Defines the probabilistic model for multi-fidelity GP classification.
            Args:
                batch (dict): Dictionary containing:
                    - 'XL': Low-fidelity input data (array of shape [NL, D])
                    - 'XH': High-fidelity input data (array of shape [NH, D])
                    - 'y': Observed binary labels (array of shape [NL+NH])
            Returns:
                None. Defines the model in the probabilistic programming context.
        posterior_sample(key, sample, X_star, **kwargs):
            Draws a posterior predictive sample at new input locations.
            Args:
                key: JAX PRNG key for randomness.
                sample (dict): Dictionary of posterior samples for model parameters.
                X_star (array): New input locations for prediction (shape [N*, D]).
                **kwargs: Must include 'batch' with training data as in model().
            Returns:
                mu (array): Predictive mean at X_star.
                sample (array): Posterior predictive sample at X_star.
    Attributes:
        Inherits all attributes from MCMCmodel.
    Notes:
        - The kernel function must be defined in the parent class.
        - Uses Cholesky decomposition for efficient GP computations.
        - Assumes binary classification with Bernoulli likelihood (logits parameterization).
    """

    def __init__(self, options):
        super().__init__(options)

    def model(self, batch):
        """
        Bayesian multi-fidelity classification model using Gaussian Processes with log-normal and normal priors.

        Args:
            batch (dict): A dictionary containing:
                - 'XL' (np.ndarray): Low-fidelity input data of shape (NL, D).
                - 'XH' (np.ndarray): High-fidelity input data of shape (NH, D).
                - 'y' (np.ndarray): Observed binary labels for both fidelities, concatenated.

        Model Details:
            - Places log-normal priors on the variance and lengthscale parameters of both low- and high-fidelity kernels.
            - Places a normal prior on the fidelity correlation parameter rho.
            - Constructs a multi-fidelity covariance matrix using the specified kernels and rho.
            - Samples latent function values using a Cholesky decomposition of the covariance matrix.
            - Includes normal priors for mean offsets (beta_L, beta_H) and latent noise (eta_L, eta_H).
            - Observes binary labels via a Bernoulli likelihood with logits given by the latent function.

        Returns:
            None. Defines the probabilistic model for use in MCMC or variational inference frameworks.
        """
        XL, XH = batch['XL'], batch['XH']
        y = batch['y']
        NL, NH = XL.shape[0], XH.shape[0]
        D = XH.shape[1]
        # set uninformative log-normal priors for low-fidelity kernel
        var_L = sample('kernel_var_L', dist.LogNormal(0.0, 1.0), sample_shape = (1,))
        length_L = sample('kernel_length_L', dist.LogNormal(0.0, 1.0), sample_shape = (D,))
        theta_L = np.concatenate([var_L, length_L])
        # set uninformative log-normal priors for high-fidelity kernel
        var_H = sample('kernel_var_H', dist.LogNormal(0.0, 1.0), sample_shape = (1,))
        length_H = sample('kernel_length_H', dist.LogNormal(0.0, 1.0), sample_shape = (D,))
        theta_H = np.concatenate([var_H, length_H])
        # prior for rho
        rho = sample('rho', dist.Normal(0.0, 10.0), sample_shape = (1,))
        # Compute kernels
        K_LL = self.kernel(XL, XL, theta_L) + np.eye(NL)*1e-8
        K_LH = rho*self.kernel(XL, XH, theta_L)
        K_HH = rho**2 * self.kernel(XH, XH, theta_L) + \
                        self.kernel(XH, XH, theta_H) + np.eye(NH)*1e-8
        K = np.vstack((np.hstack((K_LL,K_LH)),
                       np.hstack((K_LH.T,K_HH))))
        L = cholesky(K, lower=True)
        # Generate latent function
        beta_L = sample('beta_L', dist.Normal(0.0, 1.0))
        beta_H = sample('beta_H', dist.Normal(0.0, 1.0))
        eta_L = sample('eta_L', dist.Normal(0.0, 1.0), sample_shape=(NL,))
        eta_H = sample('eta_H', dist.Normal(0.0, 1.0), sample_shape=(NH,))
        beta = np.concatenate([beta_L*np.ones(NL), beta_H*np.ones(NH)])
        eta = np.concatenate([eta_L, eta_H])
        f = np.matmul(L, eta) + beta
        # Bernoulli likelihood
        sample('y', dist.Bernoulli(logits=f), obs=y)

    @partial(jit, static_argnums=(0,))
    def posterior_sample(self, key, sample, X_star, **kwargs):
        """
        Draws a posterior predictive sample from the multi-fidelity Gaussian Process model at new input locations.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for generating samples.
        sample : dict
            Dictionary containing sampled kernel hyperparameters and latent variables. Expected keys include:
                - 'kernel_var_L', 'kernel_var_H': Kernel variances for low- and high-fidelity.
                - 'kernel_length_L', 'kernel_length_H': Kernel lengthscales for low- and high-fidelity.
                - 'beta_L', 'beta_H': Mean function parameters for low- and high-fidelity.
                - 'eta_L', 'eta_H': Latent variables for low- and high-fidelity.
                - 'rho': Correlation parameter between fidelities.
        X_star : np.ndarray
            Test input locations at which to sample the posterior, shape (N*, D).
        **kwargs : dict
            Additional keyword arguments. Must contain:
                - 'batch': dict with keys:
                    - 'XL': Low-fidelity training inputs, shape (NL, D).
                    - 'XH': High-fidelity training inputs, shape (NH, D).

        Returns
        -------
        mu : np.ndarray
            Posterior predictive mean at X_star, shape (N*,).
        sample : np.ndarray
            Posterior predictive sample at X_star, shape (N*,).

        Notes
        -----
        This method computes the posterior mean and draws a sample from the predictive distribution at the test points X_star,
        using the multi-fidelity GP model with the provided hyperparameters and latent variables.
        """
        # Fetch training data
        batch = kwargs['batch']
        XL, XH = batch['XL'], batch['XH']
        NL, NH = XL.shape[0], XH.shape[0]
        # Fetch params
        var_L = sample['kernel_var_L']
        var_H = sample['kernel_var_H']
        length_L = sample['kernel_length_L']
        length_H = sample['kernel_length_H']
        beta_L = sample['beta_L']
        beta_H = sample['beta_H']
        eta_L = sample['eta_L']
        eta_H = sample['eta_H']
        rho = sample['rho']
        theta_L = np.concatenate([var_L, length_L])
        theta_H = np.concatenate([var_H, length_H])
        beta = np.concatenate([beta_L*np.ones(NL), beta_H*np.ones(NH)])
        eta = np.concatenate([eta_L, eta_H])
        # Compute kernels
        k_pp = rho**2 * self.kernel(X_star, X_star, theta_L) + \
                        self.kernel(X_star, X_star, theta_H) + \
                        np.eye(X_star.shape[0])*1e-8
        psi1 = rho*self.kernel(X_star, XL, theta_L)
        psi2 = rho**2 * self.kernel(X_star, XH, theta_L) + \
                        self.kernel(X_star, XH, theta_H)
        k_pX = np.hstack((psi1,psi2))
        # Compute K_xx
        K_LL = self.kernel(XL, XL, theta_L) + np.eye(NL)*1e-8
        K_LH = rho*self.kernel(XL, XH, theta_L)
        K_HH = rho**2 * self.kernel(XH, XH, theta_L) + \
                        self.kernel(XH, XH, theta_H) + np.eye(NH)*1e-8
        K_xx = np.vstack((np.hstack((K_LL,K_LH)),
                       np.hstack((K_LH.T,K_HH))))
        L = cholesky(K_xx, lower=True)
        # Sample latent function
        f = np.matmul(L, eta) + beta
        tmp_1 = solve_triangular(L.T,solve_triangular(L, f, lower=True))
        tmp_2  = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean
        mu = np.matmul(k_pX, tmp_1)
        cov = k_pp - np.matmul(k_pX, tmp_2)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        sample = mu + std * random.normal(key, mu.shape)
        return mu, sample

class BayesianMLP(MCMCmodel):
    """
    BayesianMLP is a Bayesian Multi-Layer Perceptron model for regression tasks using MCMC inference.
    Args:
        options (dict): Configuration options for the MCMC model.
        layers (list): List specifying the number of units in each layer of the MLP.
    Methods:
        model(batch):
            Probabilistic model definition for the Bayesian MLP.
            Args:
                batch (dict): Dictionary containing input features 'X' (array) and targets 'y' (array).
            Defines priors over weights and biases for each layer, computes the forward pass,
            and specifies the likelihood for observed data.
        forward(H, sample):
            Performs a deterministic forward pass through the network using sampled weights and biases.
            Args:
                H (array): Input features.
                sample (dict): Dictionary of sampled weights and biases.
            Returns:
                mu (array): Predicted mean values.
                sigma (array): Predicted standard deviations.
        posterior_sample(key, sample, X_star, **kwargs):
            Generates a posterior predictive sample for new inputs.
            Args:
                key: JAX random key.
                sample (dict): Dictionary of sampled weights and biases.
                X_star (array): New input features for prediction.
                **kwargs: Additional arguments, expects 'norm_const' for de-normalization.
            Returns:
                mu (array): De-normalized predictive mean.
                sample (array): De-normalized posterior predictive sample.
    """

    def __init__(self, options, layers):
        super().__init__(options)
        self.layers = layers

    def model(self, batch):
        """
        Defines a probabilistic neural network model for regression using Bayesian inference.

        Args:
            batch (dict): A dictionary containing input data with keys:
                - 'X': Input features, shape (N, D)
                - 'y': Target values, shape (N,)

        Model Structure:
            - The model consists of multiple fully connected layers as specified by self.layers.
            - Each hidden layer uses weights and biases sampled from standard normal distributions.
            - Hidden activations use the tanh nonlinearity.
            - The output layer predicts both the mean (mu) and standard deviation (sigma) of the target distribution.
            - The output mean and standard deviation are parameterized by separate sets of weights and biases, each sampled from standard normal distributions.
            - The likelihood of the observed targets y is modeled as a Normal distribution with the predicted mean and standard deviation.

        Returns:
            None. The function defines the probabilistic model for use with probabilistic programming frameworks (e.g., NumPyro).
        """
        X = batch['X']
        y = batch['y']
        N, D = X.shape
        H = X
        # Forward pass
        num_layers = len(self.layers)
        for l in range(0,num_layers-2):
            D_X, D_H = self.layers[l], self.layers[l+1]
            W = sample('w%d' % (l+1), dist.Normal(np.zeros((D_X, D_H)), np.ones((D_X, D_H))))
            b = sample('b%d' % (l+1), dist.Normal(np.zeros(D_H), np.ones(D_H)))
            H = np.tanh(np.add(np.matmul(H, W), b))
        D_X, D_H = self.layers[-2], self.layers[-1]
        # Output mean
        W = sample('w%d_mu' % (num_layers-1), dist.Normal(np.zeros((D_X, D_H)), np.ones((D_X, D_H))))
        b = sample('b%d_mu' % (num_layers-1), dist.Normal(np.zeros(D_H), np.ones(D_H)))
        mu = np.add(np.matmul(H, W), b)
        # Output std
        W = sample('w%d_std' % (num_layers-1), dist.Normal(np.zeros((D_X, D_H)), np.ones((D_X, D_H))))
        b = sample('b%d_std' % (num_layers-1), dist.Normal(np.zeros(D_H), np.ones(D_H)))
        sigma = np.exp(np.add(np.matmul(H, W), b))
        mu, sigma = mu.flatten(), sigma.flatten()
        # Likelihood
        sample("y", dist.Normal(mu, sigma), obs=y)

    @partial(jit, static_argnums=(0,))
    def forward(self, H, sample):
        """
        Performs a forward pass through a multi-layer neural network using the provided weights and biases.

        Args:
            H (np.ndarray): Input data or activations from the previous layer.
            sample (dict): Dictionary containing weight and bias parameters for each layer. 
                - For hidden layers: keys 'w{l}' and 'b{l}' for layer l (1-indexed).
                - For output layer: keys 'w{num_layers-1}_mu', 'b{num_layers-1}_mu', 
                  'w{num_layers-1}_std', and 'b{num_layers-1}_std'.

        Returns:
            tuple: 
                - mu (np.ndarray): The mean output of the network.
                - sigma (np.ndarray): The standard deviation output of the network (after applying exp).
        """
        num_layers = len(self.layers)
        for l in range(0,num_layers-2):
            W = sample['w%d'%(l+1)]
            b = sample['b%d'%(l+1)]
            H = np.tanh(np.add(np.matmul(H, W), b))
        W = sample['w%d_mu'%(num_layers-1)]
        b = sample['b%d_mu'%(num_layers-1)]
        mu = np.add(np.matmul(H, W), b)
        W = sample['w%d_std'%(num_layers-1)]
        b = sample['b%d_std'%(num_layers-1)]
        sigma = np.exp(np.add(np.matmul(H, W), b))
        return mu, sigma

    @partial(jit, static_argnums=(0,))
    def posterior_sample(self, key, sample, X_star, **kwargs):
        """
        Draws a sample from the posterior predictive distribution at the given input locations.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for generating random numbers.
        sample : array-like
            Posterior sample of model parameters.
        X_star : array-like
            Input locations at which to evaluate the posterior predictive distribution.
        **kwargs : dict
            Additional keyword arguments. Must include:
                - 'norm_const': dict with keys 'mu_y' and 'sigma_y' for de-normalization.

        Returns
        -------
        mu : numpy.ndarray
            De-normalized predictive mean at X_star, flattened to 1D.
        sample : numpy.ndarray
            De-normalized posterior predictive sample at X_star, flattened to 1D.
        """
        mu, sigma = self.forward(X_star, sample)
        sample = mu + np.sqrt(sigma) * random.normal(key, mu.shape)
        # De-normalize
        norm_const = kwargs['norm_const']
        mu = mu*norm_const['sigma_y'] + norm_const['mu_y']
        sample = sample*norm_const['sigma_y'] + norm_const['mu_y']
        return mu.flatten(), sample.flatten()

# Work in progress..
class MissingInputsGP(MCMCmodel):
    """
    A Gaussian Process (GP) model for handling missing inputs using Markov Chain Monte Carlo (MCMC) inference.
    This model augments observed input data with latent variables to account for missing features, and performs
    Bayesian inference over both the GP hyperparameters and the latent inputs.
    Args:
        options (dict): Configuration options for the MCMC model.
        dim_H (int): The dimensionality of the latent (missing) input space.
        latent_bounds (tuple): Bounds for the latent variables.
    Methods:
        model(batch):
            Defines the probabilistic model for the GP with missing inputs.
            - batch (dict): Dictionary containing 'X' (inputs) and 'y' (targets).
            - Samples latent inputs, GP hyperparameters, and models the likelihood of observed data.
        compute_cholesky(params, batch):
            Computes the Cholesky decomposition of the GP covariance matrix for efficient inference.
            - params (array): GP hyperparameters and noise variance.
            - batch (dict): Dictionary containing 'X' (inputs).
            - Returns: Lower-triangular Cholesky factor.
        posterior_sample(key, sample, X_star, **kwargs):
            Draws samples from the GP posterior predictive distribution at new input locations.
            - key: PRNG key for randomness.
            - sample (dict): Dictionary of sampled latent variables and GP hyperparameters.
            - X_star (array): New input locations for prediction.
            - kwargs: Additional arguments, including 'batch' and 'norm_const' for normalization.
            - Returns: Tuple of (predictive mean, predictive sample) at X_star.
    """
    # Initialize the class
    def __init__(self, options, dim_H, latent_bounds):
        super().__init__(options)
        self.dim_H = dim_H
        self.latent_bounds = latent_bounds

    def model(self, batch):
        """
        Defines a probabilistic model for Gaussian Process Latent Variable Model (GPLVM) using MCMC.

        Args:
            batch (dict): A dictionary containing:
                - 'X' (np.ndarray): Observed input data of shape (N, dim_X).
                - 'y' (np.ndarray): Observed output data of shape (N,).

        Model Details:
            - Augments observed inputs X with latent variables H sampled from a standard normal distribution.
            - Places log-normal priors on the GP kernel variance, lengthscales, and noise variance.
            - Constructs the GP kernel matrix using the augmented inputs and sampled hyperparameters.
            - Models the observed outputs y with a multivariate normal likelihood parameterized by the GP kernel.

        Returns:
            None. The function defines the probabilistic model structure for use in MCMC inference.
        """
        X = batch['X']
        y = batch['y']
        N = y.shape[0]
        dim_X = X.shape[1]
        dim_H = self.dim_H
        D = dim_X + dim_H
        # Generate latent inputs
        H = sample('H', dist.Normal(np.zeros((N, dim_H)), np.ones((N, dim_H))))
        X = np.concatenate([X, H], axis = 1)
        # set uninformative log-normal priors on GP hyperparameters
        var = sample('kernel_var', dist.LogNormal(0.0, 10.0))
        length = sample('kernel_length', dist.LogNormal(np.zeros(D), 10.0*np.ones(D)))
        noise = sample('noise_var', dist.LogNormal(0.0, 10.0))
        theta = np.concatenate([np.array([var]), np.array(length)])
        # compute kernel
        K = self.kernel(X, X, theta) + np.eye(N)*(noise + 1e-8)
        # sample Y according to the GP likelihood
        sample("y", dist.MultivariateNormal(loc=np.zeros(N), covariance_matrix=K), obs=y)

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        """
        Computes the Cholesky decomposition of the kernel matrix for a given batch and parameters.

        Parameters:
            params (array-like): Model parameters, where the last element is the noise variance (sigma_n)
                                 and the preceding elements are kernel hyperparameters (theta).
            batch (dict): A dictionary containing input data with key 'X', where 'X' is an (N, D) array
                          of N data points with D features.

        Returns:
            ndarray: The lower-triangular Cholesky factor (L) of the kernel matrix K.

        Notes:
            - The kernel matrix K is computed as kernel(X, X, theta) + I * (sigma_n + 1e-8),
              where I is the identity matrix and 1e-8 is added for numerical stability.
            - Assumes that self.kernel is a callable kernel function and cholesky is available.
        """
        X = batch['X']
        N, D = X.shape
        # Fetch params
        sigma_n = params[-1]
        theta = params[:-1]
        # Compute kernel
        K = self.kernel(X, X, theta) + np.eye(N)*(sigma_n + 1e-8)
        L = cholesky(K, lower=True)
        return L

    @partial(jit, static_argnums=(0,))
    def posterior_sample(self, key, sample, X_star, **kwargs):
        """
        Draws a posterior predictive sample from the Gaussian Process (GP) model at new input locations.

        Args:
            key: A JAX PRNG key for random number generation.
            sample (dict): Dictionary containing GP hyperparameters and latent variables, including:
                - 'H': Missing input features to concatenate with X.
                - 'kernel_var': Kernel variance parameter.
                - 'kernel_length': Kernel lengthscale(s).
                - 'noise_var': Observation noise variance.
            X_star (np.ndarray): Test input locations at which to sample the posterior, shape (n*, d).
            **kwargs: Additional keyword arguments, including:
                - 'batch' (dict): Dictionary with training data:
                    - 'X': Training inputs, shape (n, d).
                    - 'y': Training targets, shape (n,).
                - 'norm_const' (dict): Normalization constants with keys:
                    - 'mu_y': Mean of training targets.
                    - 'sigma_y': Standard deviation of training targets.

        Returns:
            mu (np.ndarray): Posterior predictive mean at X_star, de-normalized.
            sample (np.ndarray): Posterior predictive sample at X_star, de-normalized.
        """
        batch = kwargs['batch']
        X, y = batch['X'], batch['y']
        # Fetch missing inputs
        H = sample['H']
        X = np.concatenate([X, H], axis=1)
        # Fetch GP params
        var = sample['kernel_var']
        length = sample['kernel_length']
        noise = sample['noise_var']
        params = np.concatenate([np.array([var]), np.array(length), np.array([noise])])
        theta = params[:-1]
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0])*(noise + 1e-8)
        k_pX = self.kernel(X_star, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta  = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        sample = mu + std * random.normal(key, mu.shape)
        # De-normalize
        norm_const = kwargs['norm_const']
        mu = mu*norm_const['sigma_y'] + norm_const['mu_y']
        sample = sample*norm_const['sigma_y'] + norm_const['mu_y']
        return mu, sample
