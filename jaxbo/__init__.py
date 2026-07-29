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

"""jaxbo: Bayesian optimization on JAX.

The eager namespace below is the core (SCOPE.md section 3): it imports with
only the core dependencies (jax, jaxlib, numpy, scipy). ``mcmc_models``
(numpyro) is resolved lazily on first attribute access so ``import jaxbo``
never pays for it; slice 2b moves it into the ``[mcmc]`` optional extra.
"""

from typing import List

from jaxbo import (
    acquisitions,
    gp,
    initializers,
    input_priors,
    kernels,
    models,
    optimizers,
    priors,
    test_functions,
    utils,
)

# Modules resolved lazily via __getattr__ (PEP 562): extras staging, never
# imported eagerly by the core.
_LAZY_MODULES = ("mcmc_models",)

__all__ = [
    "acquisitions",
    "gp",
    "initializers",
    "input_priors",
    "kernels",
    "mcmc_models",
    "models",
    "optimizers",
    "priors",
    "test_functions",
    "utils",
]


def __getattr__(name: str):
    """Import extras-staged submodules on first access (PEP 562)."""
    if name in _LAZY_MODULES:
        import importlib

        module = importlib.import_module(f"jaxbo.{name}")
        globals()[name] = module  # cache so __getattr__ runs once per name
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    """Advertise the eager core namespace plus the lazy submodules."""
    return sorted(set(globals()) | set(_LAZY_MODULES))
