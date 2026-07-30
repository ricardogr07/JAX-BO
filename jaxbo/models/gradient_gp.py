"""Compatibility shim: ``GradientGP`` now lives in :mod:`jaxbo.multifidelity`.

Kept so the historical import path
``from jaxbo.models.gradient_gp import GradientGP`` keeps working.
New code should import from :mod:`jaxbo.multifidelity`.
"""

from jaxbo.multifidelity.gradient_gp import GradientGP

__all__ = ["GradientGP"]
