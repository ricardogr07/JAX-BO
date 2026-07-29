"""Compatibility shim: ``HeterogeneousMultifidelityGP`` now lives in :mod:`jaxbo.multifidelity`.

Kept so the historical import path
``from jaxbo.models.heterogeneous_multifidelity_gp import HeterogeneousMultifidelityGP`` keeps working.
New code should import from :mod:`jaxbo.multifidelity`.
"""

from jaxbo.multifidelity.heterogeneous_multifidelity_gp import (
    HeterogeneousMultifidelityGP,
)

__all__ = ["HeterogeneousMultifidelityGP"]
