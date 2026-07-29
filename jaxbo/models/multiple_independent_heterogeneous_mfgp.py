"""Compatibility shim: ``MultipleIndependentHeterogeneousMFGP`` now lives in :mod:`jaxbo.multifidelity`.

Kept so the historical import path
``from jaxbo.models.multiple_independent_heterogeneous_mfgp import MultipleIndependentHeterogeneousMFGP`` keeps working.
New code should import from :mod:`jaxbo.multifidelity`.
"""

from jaxbo.multifidelity.multiple_independent_heterogeneous_mfgp import (
    MultipleIndependentHeterogeneousMFGP,
)

__all__ = ["MultipleIndependentHeterogeneousMFGP"]
