"""Compatibility shim: ``MultipleIndependentOutputsGP`` now lives in :mod:`jaxbo.multifidelity`.

Kept so the historical import path
``from jaxbo.models.multiple_independent_output_gp_model import MultipleIndependentOutputsGP`` keeps working.
New code should import from :mod:`jaxbo.multifidelity`.
"""

from jaxbo.multifidelity.multiple_independent_output_gp_model import (
    MultipleIndependentOutputsGP,
)

__all__ = ["MultipleIndependentOutputsGP"]
