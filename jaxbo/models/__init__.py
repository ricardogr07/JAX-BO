# Forked and modified by Ricardo García Ramírez (2025)
# Original Copyright 2019 Predictive Intelligence Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model namespace for jaxbo.

The core exposes exactly two classes eagerly: :class:`jaxbo.gp.GPmodel` and
:class:`jaxbo.gp.GP` (SCOPE.md section 3). The multifidelity, manifold,
gradient, and multiple-output research models moved to the
:mod:`jaxbo.multifidelity` extra; they remain importable from this namespace
for backward compatibility, resolved lazily on first access so the core
import graph never pays for them.
"""

from typing import List

from jaxbo.gp import GP, GPmodel

# Lazily resolved research models: attribute name to its jaxbo.multifidelity
# home (SCOPE.md sections 3 and 7). Nothing in the core imports them eagerly.
_LAZY_MODELS = {
    "MultipleIndependentOutputsGP": "multiple_independent_output_gp_model",
    "ManifoldGP": "manifold_gp_model",
    "ManifoldGP_MultiOutputs": "manifold_gp_multioutputs",
    "MultifidelityGP": "multifidelity_gp",
    "DeepMultifidelityGP": "deep_multifidelity_gp",
    "DeepMultifidelityGP_MultiOutputs": "deep_multifidelity_gp_multioutputs",
    "GradientGP": "gradient_gp",
    "MultipleIndependentMFGP": "multiple_independent_mfgp",
    "HeterogeneousMultifidelityGP": "heterogeneous_multifidelity_gp",
    "MultipleIndependentHeterogeneousMFGP": "multiple_independent_heterogeneous_mfgp",
}

__all__ = ["GPmodel", "GP"]


def __getattr__(name: str):
    """Resolve research model classes lazily (PEP 562).

    Keeps historical imports such as ``from jaxbo.models import
    MultifidelityGP`` working without dragging their heavier dependency
    graph into ``import jaxbo``.
    """
    module_name = _LAZY_MODELS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    cls = getattr(importlib.import_module(f"jaxbo.multifidelity.{module_name}"), name)
    globals()[name] = cls  # cache so __getattr__ runs once per name
    return cls


def __dir__() -> List[str]:
    """Advertise both the eager core names and the lazy research models."""
    return sorted(set(globals()) | set(_LAZY_MODELS))
