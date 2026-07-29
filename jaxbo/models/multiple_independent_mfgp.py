"""Compatibility shim: ``MultipleIndependentMFGP`` now lives in :mod:`jaxbo.multifidelity`.

Kept so the historical import path
``from jaxbo.models.multiple_independent_mfgp import MultipleIndependentMFGP`` keeps working.
New code should import from :mod:`jaxbo.multifidelity`.
"""

from jaxbo.multifidelity.multiple_independent_mfgp import MultipleIndependentMFGP

__all__ = ["MultipleIndependentMFGP"]
