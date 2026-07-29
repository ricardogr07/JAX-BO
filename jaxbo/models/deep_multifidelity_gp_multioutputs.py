"""Compatibility shim: ``DeepMultifidelityGP_MultiOutputs`` now lives in :mod:`jaxbo.multifidelity`.

Kept so the historical import path
``from jaxbo.models.deep_multifidelity_gp_multioutputs import DeepMultifidelityGP_MultiOutputs`` keeps working.
New code should import from :mod:`jaxbo.multifidelity`.
"""

from jaxbo.multifidelity.deep_multifidelity_gp_multioutputs import (
    DeepMultifidelityGP_MultiOutputs,
)

__all__ = ["DeepMultifidelityGP_MultiOutputs"]
