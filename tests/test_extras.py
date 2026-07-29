"""Contracts of the 2b extras split (SCOPE.md sections 3 and 4, decision 7).

Locks four things:

1. The extras import lazily: ``import jaxbo`` succeeds with the extras'
   dependencies BLOCKED, and ``jaxbo.multifidelity`` imports with core deps
   only.
2. Importing an extra without its dependency raises an ImportError naming
   the ``pip install jaxbo[extra]`` fix, including through the compat shims
   and the method-level weighted surface.
3. Every historical import path re-exports the new layout object identically.
4. The extracted weighted surface still works end to end when the
   dependencies are installed (fit_gmm -> gmm_vars -> LW-LCB acquisition).
"""

import importlib
import subprocess
import sys
import textwrap

import jax.numpy as jnp
from jax import random

# Class name -> historical jaxbo.models submodule, exactly the pre-split map.
MODEL_SHIMS = {
    "MultifidelityGP": "multifidelity_gp",
    "DeepMultifidelityGP": "deep_multifidelity_gp",
    "DeepMultifidelityGP_MultiOutputs": "deep_multifidelity_gp_multioutputs",
    "HeterogeneousMultifidelityGP": "heterogeneous_multifidelity_gp",
    "ManifoldGP": "manifold_gp_model",
    "ManifoldGP_MultiOutputs": "manifold_gp_multioutputs",
    "GradientGP": "gradient_gp",
    "MultipleIndependentMFGP": "multiple_independent_mfgp",
    "MultipleIndependentHeterogeneousMFGP": "multiple_independent_heterogeneous_mfgp",
    "MultipleIndependentOutputsGP": "multiple_independent_output_gp_model",
}


def test_old_import_paths_reexport_new_layout():
    """Every pre-split path resolves to the same object as the new home."""
    import jaxbo
    import jaxbo.multifidelity as mf
    import jaxbo.utils as utils
    import jaxbo.weights as weights
    from jaxbo.multifidelity import nn

    # jaxbo.models attribute access and direct submodule paths
    for cls_name, mod_name in MODEL_SHIMS.items():
        new = getattr(mf, cls_name)
        assert getattr(jaxbo.models, cls_name) is new
        shim = importlib.import_module(f"jaxbo.models.{mod_name}")
        assert getattr(shim, cls_name) is new

    # mcmc_models shim mirrors jaxbo.mcmc
    assert jaxbo.mcmc_models.MCMCGP is jaxbo.mcmc.MCMCGP
    assert jaxbo.mcmc_models.MCMCmodel is jaxbo.mcmc.MCMCmodel
    # The 2a rename holds: no GP name in the mcmc namespace
    assert not hasattr(jaxbo.mcmc, "GP")

    # serializable shim mirrors jaxbo.multifidelity.serializable
    assert jaxbo.serializable.serializable_MF is mf.serializable_MF
    assert jaxbo.serializable.deserializable_MF is mf.deserializable_MF

    # utils forwards the moved weighted and nn surfaces
    assert utils.fit_kernel_density is weights.fit_kernel_density
    assert utils.compute_w_gmm is weights.compute_w_gmm
    assert utils.init_NN is nn.init_NN
    assert utils.init_ResNet is nn.init_ResNet
    assert utils.init_MomentumResNet is nn.init_MomentumResNet


def test_missing_extra_raises_install_hint():
    """With the extras' deps blocked, guards raise the pip install hint.

    Runs in a fresh interpreter with a meta_path hook blocking numpyro,
    sklearn, and KDEpy, so the result holds regardless of what this
    environment has installed.
    """
    code = textwrap.dedent(
        """
        import sys

        class BlockExtras:
            blocked = ("numpyro", "sklearn", "KDEpy")

            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in self.blocked:
                    raise ImportError(f"blocked for test: {name}")
                return None

        sys.meta_path.insert(0, BlockExtras())

        # Core and the dependency-free extra import clean without any of them
        import jaxbo
        import jaxbo.multifidelity

        # Star import stays core-only: __all__ must not advertise the lazy
        # extras, or this line would import them (adversarial review finding)
        exec("from jaxbo import *", {})

        try:
            import jaxbo.mcmc
        except ImportError as e:
            assert "pip install jaxbo[mcmc]" in str(e), str(e)
        else:
            raise SystemExit("jaxbo.mcmc imported without numpyro")

        try:
            import jaxbo.mcmc_models
        except ImportError as e:
            assert "pip install jaxbo[mcmc]" in str(e), str(e)
        else:
            raise SystemExit("jaxbo.mcmc_models imported without numpyro")

        try:
            import jaxbo.weights
        except ImportError as e:
            assert "pip install jaxbo[weighted]" in str(e), str(e)
        else:
            raise SystemExit("jaxbo.weights imported without sklearn/KDEpy")

        # Method-level weighted surface gives the same hint
        import jax.numpy as jnp
        from jaxbo.gp import GP
        from jaxbo.input_priors import uniform_prior

        lb, ub = jnp.zeros(1), jnp.ones(1)
        gp = GP({"kernel": "RBF", "input_prior": uniform_prior(lb, ub),
                 "criterion": "LW-LCB"})
        try:
            gp.fit_gmm(bounds={"lb": lb, "ub": ub}, rng_key=None)
        except ImportError as e:
            assert "jaxbo[weighted]" in str(e), str(e)
        else:
            raise SystemExit("fit_gmm ran without the [weighted] extra")
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_one_model_import_loads_only_its_module():
    """A historical single-model import must not execute sibling modules.

    Regression for the adversarial review finding: an eager
    jaxbo.multifidelity ``__init__`` made ``from jaxbo.models import
    MultifidelityGP`` import every research model plus stax. Runs in a fresh
    interpreter so this suite's own imports cannot mask the result.
    """
    code = textwrap.dedent(
        """
        import sys

        from jaxbo.models import MultifidelityGP  # noqa: F401

        loaded = {m for m in sys.modules if m.startswith("jaxbo.multifidelity")}
        allowed = {"jaxbo.multifidelity", "jaxbo.multifidelity.multifidelity_gp"}
        assert loaded <= allowed, f"eager sibling imports: {sorted(loaded - allowed)}"
        assert "jax.example_libraries.stax" not in sys.modules, "stax leaked"
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_weighted_surface_end_to_end():
    """fit_gmm produces gmm_vars that drive the LW-LCB criterion."""
    from jaxbo.input_priors import uniform_prior
    from jaxbo.models import GP
    from jaxbo.utils import normalize

    lb, ub = jnp.zeros(1), jnp.ones(1)
    bounds = {"lb": lb, "ub": ub}
    prior = uniform_prior(lb, ub)
    gp = GP({"kernel": "RBF", "input_prior": prior, "criterion": "LW-LCB"})

    X = jnp.linspace(0.0, 1.0, 8)[:, None]
    y = ((X.flatten() - 0.3) ** 2) + 0.1
    batch, norm_const = normalize(X, y, bounds)
    rng_key = random.PRNGKey(0)
    opt_params = gp.train(batch, rng_key, num_restarts=2)

    kwargs = dict(
        params=opt_params,
        batch=batch,
        norm_const=norm_const,
        bounds=bounds,
        rng_key=rng_key,
    )
    weights, means, covs = gp.fit_gmm(num_comp=2, N_samples=200, **kwargs)
    assert weights.shape == (2,)
    assert means.shape == (2, 1)
    assert covs.shape == (2, 1, 1)

    acq = gp.acquisition(
        jnp.array([0.5]),
        gmm_vars=(jnp.array(weights), jnp.array(means), jnp.array(covs)),
        kappa=2.0,
        **kwargs,
    )
    assert jnp.isfinite(acq)


def test_constrained_fit_gmm_delegates(monkeypatch):
    """Both constrained models route fit_gmm through the shared extraction."""
    import jaxbo.weights as weights
    from jaxbo.input_priors import uniform_prior
    from jaxbo.multifidelity import (
        MultipleIndependentHeterogeneousMFGP,
        MultipleIndependentMFGP,
    )

    lb, ub = jnp.zeros(2), jnp.ones(2)
    prior = uniform_prior(lb, ub)
    calls = []

    def fake(model, num_comp=2, N_samples=10000, **kwargs):
        calls.append((model, num_comp, N_samples, kwargs))
        return "sentinel"

    monkeypatch.setattr(weights, "fit_gmm_constrained", fake)

    m1 = MultipleIndependentMFGP({"kernel": "RBF", "input_prior": prior})
    m2 = MultipleIndependentHeterogeneousMFGP(
        {"kernel": "RBF", "input_prior": prior}, layers=[2, 4, 2]
    )
    assert m1.fit_gmm(num_comp=3, N_samples=7, rng_key="k") == "sentinel"
    assert m2.fit_gmm(num_comp=4, N_samples=9, rng_key="k") == "sentinel"

    assert calls[0][0] is m1 and calls[0][1:3] == (3, 7)
    assert calls[1][0] is m2 and calls[1][1:3] == (4, 9)
