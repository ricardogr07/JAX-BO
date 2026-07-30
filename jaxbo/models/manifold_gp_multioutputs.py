"""Compatibility shim: ``ManifoldGP_MultiOutputs`` now lives in :mod:`jaxbo.multifidelity`.

Kept so the historical import path
``from jaxbo.models.manifold_gp_multioutputs import ManifoldGP_MultiOutputs`` keeps working.
New code should import from :mod:`jaxbo.multifidelity`.
"""

from jaxbo.multifidelity.manifold_gp_multioutputs import ManifoldGP_MultiOutputs

__all__ = ["ManifoldGP_MultiOutputs"]
