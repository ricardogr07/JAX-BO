"""Compatibility shim: ``MultifidelityGP`` now lives in :mod:`jaxbo.multifidelity`.

Kept so the historical import path
``from jaxbo.models.multifidelity_gp import MultifidelityGP`` keeps working.
New code should import from :mod:`jaxbo.multifidelity`.
"""

from jaxbo.multifidelity.multifidelity_gp import MultifidelityGP

__all__ = ["MultifidelityGP"]
