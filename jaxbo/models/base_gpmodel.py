"""Compatibility shim: ``GPmodel`` now lives in :mod:`jaxbo.gp`.

Kept so the historical import path
``from jaxbo.models.base_gpmodel import GPmodel`` keeps working. New code
should import from :mod:`jaxbo.gp`.
"""

from jaxbo.gp import GPmodel

__all__ = ["GPmodel"]
