"""Multifidelity, manifold, gradient, and multi-output GP models.

Home of the ``[multifidelity]`` optional extra (SCOPE.md section 3): the six
multifidelity models, the two manifold models, ``GradientGP``, and
``MultipleIndependentOutputsGP``, all built on the shared
:class:`jaxbo.gp.GPmodel` base, plus the neural network feature maps
(:mod:`jaxbo.multifidelity.nn`) and the multifidelity parameter
serialization helpers (:mod:`jaxbo.multifidelity.serializable`).

This family needs no third-party packages beyond the jaxbo core (the feature
maps use ``jax.example_libraries.stax``, which ships with jax), so the extra
currently pins nothing; it exists so ``pip install jaxbo[multifidelity]``
stays a stable name if that ever changes. The GMM-weighted acquisition
surface these models can drive (``fit_gmm``, ``LW_*`` criteria) lives behind
the separate ``[weighted]`` extra and raises its own install hint when
scikit-learn or KDEpy is missing. The jaxbo core never imports this package
eagerly; ``jaxbo.models`` keeps the historical import paths alive as shims.
"""

from jaxbo.multifidelity.deep_multifidelity_gp import DeepMultifidelityGP
from jaxbo.multifidelity.deep_multifidelity_gp_multioutputs import (
    DeepMultifidelityGP_MultiOutputs,
)
from jaxbo.multifidelity.gradient_gp import GradientGP
from jaxbo.multifidelity.heterogeneous_multifidelity_gp import (
    HeterogeneousMultifidelityGP,
)
from jaxbo.multifidelity.manifold_gp_model import ManifoldGP
from jaxbo.multifidelity.manifold_gp_multioutputs import ManifoldGP_MultiOutputs
from jaxbo.multifidelity.multifidelity_gp import MultifidelityGP
from jaxbo.multifidelity.multiple_independent_heterogeneous_mfgp import (
    MultipleIndependentHeterogeneousMFGP,
)
from jaxbo.multifidelity.multiple_independent_mfgp import MultipleIndependentMFGP
from jaxbo.multifidelity.multiple_independent_output_gp_model import (
    MultipleIndependentOutputsGP,
)
from jaxbo.multifidelity.serializable import deserializable_MF, serializable_MF

__all__ = [
    "DeepMultifidelityGP",
    "DeepMultifidelityGP_MultiOutputs",
    "GradientGP",
    "HeterogeneousMultifidelityGP",
    "ManifoldGP",
    "ManifoldGP_MultiOutputs",
    "MultifidelityGP",
    "MultipleIndependentHeterogeneousMFGP",
    "MultipleIndependentMFGP",
    "MultipleIndependentOutputsGP",
    "deserializable_MF",
    "serializable_MF",
]
