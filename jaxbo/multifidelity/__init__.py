"""Multifidelity, manifold, gradient, and multi-output GP models.

Home of the ``[multifidelity]`` optional extra: the six
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
scikit-learn or KDEpy is missing.

Every class resolves lazily (PEP 562) so importing one model, here or
through the historical ``jaxbo.models`` shims, never executes the sibling
model modules. The jaxbo core never imports this package eagerly.
"""

from typing import List

# Attribute name to its submodule; each loads only on first access.
_LAZY_ATTRS = {
    "DeepMultifidelityGP": "deep_multifidelity_gp",
    "DeepMultifidelityGP_MultiOutputs": "deep_multifidelity_gp_multioutputs",
    "GradientGP": "gradient_gp",
    "HeterogeneousMultifidelityGP": "heterogeneous_multifidelity_gp",
    "ManifoldGP": "manifold_gp_model",
    "ManifoldGP_MultiOutputs": "manifold_gp_multioutputs",
    "MultifidelityGP": "multifidelity_gp",
    "MultipleIndependentHeterogeneousMFGP": "multiple_independent_heterogeneous_mfgp",
    "MultipleIndependentMFGP": "multiple_independent_mfgp",
    "MultipleIndependentOutputsGP": "multiple_independent_output_gp_model",
    "deserializable_MF": "serializable",
    "serializable_MF": "serializable",
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name: str):
    """Resolve model classes and helpers lazily (PEP 562)."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    obj = getattr(importlib.import_module(f"jaxbo.multifidelity.{module_name}"), name)
    globals()[name] = obj  # cache so __getattr__ runs once per name
    return obj


def __dir__() -> List[str]:
    """Advertise the lazily resolved names alongside the loaded ones."""
    return sorted(set(globals()) | set(_LAZY_ATTRS))
