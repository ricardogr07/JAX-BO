"""Compatibility shim: the MF serialization helpers moved to
:mod:`jaxbo.multifidelity.serializable` with the extras split (the
section 3). Importing this path pulls the [multifidelity] extra's package,
which needs no dependencies beyond the jaxbo core.
"""

from jaxbo.multifidelity.serializable import deserializable_MF, serializable_MF

__all__ = ["deserializable_MF", "serializable_MF"]
