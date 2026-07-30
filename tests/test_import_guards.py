"""Extras import guards beyond the plain import check (slice 2d).

``tests/test_compat.py::test_core_import_graph_is_clean`` already proves
that ``import jaxbo`` alone pulls no extras dependency. This complements it
in a fresh interpreter: USING the core GP surface (model construction plus
``normalize``) still keeps numpyro, scikit-learn, KDEpy, and stax out, and
the lazy boundary is real, touching a staged name actually pulls its
dependency in rather than the name having been smuggled into the core.
"""

import subprocess
import sys
import textwrap


def test_core_gp_usage_keeps_extras_out():
    """Core GP construction and normalization must not load any extras dep."""
    code = textwrap.dedent(
        """
        import sys

        import jax.numpy as jnp
        import jaxbo
        from jaxbo.input_priors import uniform_prior
        from jaxbo.models import GP
        from jaxbo.utils import normalize

        lb, ub = jnp.zeros(1), jnp.ones(1)
        gp = GP({"kernel": "RBF", "input_prior": uniform_prior(lb, ub),
                 "criterion": "EI"})
        X = jnp.linspace(0.0, 1.0, 4)[:, None]
        batch, norm_const = normalize(X, X.flatten() ** 2, {"lb": lb, "ub": ub})

        extras = [m for m in sys.modules
                  if m.split(".")[0] in ("numpyro", "sklearn", "KDEpy")
                  or m == "jax.example_libraries.stax"]
        assert not extras, f"core GP usage leaked extras deps: {extras}"
        assert "jaxbo.mcmc_models" not in sys.modules, "mcmc_models loaded eagerly"
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_lazy_boundary_actually_defers_kdepy():
    """The staged KDE surface lives behind the boundary, not inside the core.

    Accessing ``jaxbo.utils.fit_kernel_density`` must be what first pulls
    KDEpy into the process; if KDEpy were already loaded (or never loaded),
    the lazy forwarding would be fake or broken.
    """
    code = textwrap.dedent(
        """
        import sys

        import jaxbo.utils

        assert not any(m.split(".")[0] == "KDEpy" for m in sys.modules), \\
            "KDEpy loaded before the staged name was touched"
        assert callable(jaxbo.utils.fit_kernel_density)
        assert any(m.split(".")[0] == "KDEpy" for m in sys.modules), \\
            "fit_kernel_density did not route through the KDEpy staging module"
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
