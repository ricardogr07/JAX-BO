"""Compatibility locks for the 0.2.0 core restructure (SCOPE.md section 2).

Two contracts are locked here:

1. The historical import paths keep working after the 2a layout moves
   (``jaxbo.models``, ``jaxbo.models.gp_model``, ``jaxbo.models.base_gpmodel``,
   ``jaxbo.input_priors``, ``jaxbo.utils`` forwarding).
2. The shelter-pulse call pattern runs unchanged: ``GP`` dict options,
   ``train(batch, rng_key, num_restarts)``,
   ``predict(X_star, params=, batch=, bounds=)``, ``EI(mu, std, best)``.

Plus the 2a acceptance gate: ``import jaxbo`` must not transitively import
numpyro, scikit-learn, KDEpy, or ``jax.example_libraries.stax``.
"""

import subprocess
import sys

import jax.numpy as jnp
from jax import random

# Old import paths, exactly as consumers wrote them before the restructure.
from jaxbo import acquisitions
from jaxbo.input_priors import uniform_prior
from jaxbo.models import GP
from jaxbo.utils import normalize


def test_core_import_graph_is_clean():
    """import jaxbo must not reach numpyro, sklearn, KDEpy, or stax.

    Runs in a fresh interpreter so imports made by other tests cannot mask
    or fake the result.
    """
    code = (
        "import jaxbo, sys\n"
        "leaked = [m for m in sys.modules"
        " if m.startswith(('numpyro', 'sklearn', 'KDEpy'))"
        " or m == 'jax.example_libraries.stax']\n"
        "assert not leaked, f'core import graph leaked: {leaked}'\n"
        "assert 'jaxbo.mcmc_models' not in sys.modules, 'mcmc_models imported eagerly'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_old_import_paths_still_resolve():
    """The pre-0.2.0 import paths are re-exports of the new layout."""
    from jaxbo.gp import GP as GP_new
    from jaxbo.gp import GPmodel as GPmodel_new
    from jaxbo.models import GPmodel
    from jaxbo.models.base_gpmodel import GPmodel as GPmodel_old
    from jaxbo.models.gp_model import GP as GP_old
    from jaxbo.priors import uniform_prior as uniform_prior_new

    assert GP is GP_old is GP_new
    assert GPmodel is GPmodel_old is GPmodel_new
    assert uniform_prior is uniform_prior_new
    assert callable(acquisitions.EI)


def test_lazy_namespaces_resolve_on_access():
    """Research models and mcmc_models stay reachable, just lazily."""
    import jaxbo
    import jaxbo.utils as utils

    from jaxbo.models import MultifidelityGP  # noqa: F401  lazy attribute

    assert callable(utils.fit_kernel_density)
    assert callable(utils.init_NN)
    mcmc = jaxbo.mcmc_models
    assert hasattr(mcmc, "MCMCGP")
    # The rename is deliberate: the numpyro GP no longer shadows jaxbo.gp.GP.
    assert not hasattr(mcmc, "GP")


def test_shelter_pulse_call_pattern():
    """The exact consumer pattern from shelterpulse/optimize/jaxbo_optimizer.py.

    Normalization contract: train takes the ALREADY NORMALIZED batch from
    jaxbo.utils.normalize; predict takes RAW domain X_star plus bounds.
    """
    lb, ub = jnp.zeros(1), jnp.ones(1)
    bounds = {"lb": lb, "ub": ub}
    prior = uniform_prior(lb, ub)

    gp = GP({"kernel": "RBF", "input_prior": prior, "criterion": "EI"})

    X = jnp.linspace(0.0, 1.0, 8)[:, None]
    y = ((X.flatten() - 0.3) ** 2) + 0.1
    batch, norm_const = normalize(X, y, bounds)

    rng_key = random.PRNGKey(0)
    opt_params = gp.train(batch, rng_key, num_restarts=2)
    assert jnp.all(jnp.isfinite(opt_params))

    # Raw-domain candidates, normalized internally by predict against bounds.
    X_star = jnp.linspace(0.0, 1.0, 16)[:, None]
    mu, std = gp.predict(X_star, params=opt_params, batch=batch, bounds=bounds)
    assert mu.shape[0] == 16
    assert std.shape == (16,)
    assert jnp.all(jnp.isfinite(mu))
    assert jnp.all(std >= 0.0)

    # shelter-pulse scores candidates one at a time with EI.
    best = jnp.min(batch["y"])
    scores = [acquisitions.EI(mu[i : i + 1], std[i : i + 1], best) for i in range(4)]
    assert all(jnp.isfinite(s) for s in scores)
