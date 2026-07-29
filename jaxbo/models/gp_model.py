"""Compatibility shim: ``GP`` now lives in :mod:`jaxbo.gp`.

Kept so the historical import path ``from jaxbo.models.gp_model import GP``
keeps working. New code should import from :mod:`jaxbo.gp`.
"""

from jaxbo.gp import GP

__all__ = ["GP"]
