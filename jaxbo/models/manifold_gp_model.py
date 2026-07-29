"""Compatibility shim: ``ManifoldGP`` now lives in :mod:`jaxbo.multifidelity`.

Kept so the historical import path
``from jaxbo.models.manifold_gp_model import ManifoldGP`` keeps working.
New code should import from :mod:`jaxbo.multifidelity`.
"""

from jaxbo.multifidelity.manifold_gp_model import ManifoldGP

__all__ = ["ManifoldGP"]
