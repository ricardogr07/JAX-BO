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

from .base_gpmodel import GPmodel
from .gp_model import GP
from .multiple_independent_output_gp_model import MultipleIndependentOutputsGP
from .manifold_gp_model import ManifoldGP
from .manifold_gp_multioutputs import ManifoldGP_MultiOutputs
from .multifidelity_gp import MultifidelityGP
from .deep_multifidelity_gp import DeepMultifidelityGP
from .deep_multifidelity_gp_multioutputs import DeepMultifidelityGP_MultiOutputs
from .gradient_gp import GradientGP
from .multiple_independent_mfgp import MultipleIndependentMFGP
from .heterogeneous_multifidelity_gp import HeterogeneousMultifidelityGP
from .multiple_independent_heterogeneous_mfgp import MultipleIndependentHeterogeneousMFGP

__all__ = [
    'GPmodel',
    'GP',
    'MultipleIndependentOutputsGP', 
    'ManifoldGP', 
    'ManifoldGP_MultiOutputs',
    'MultifidelityGP',
    'DeepMultifidelityGP',
    'DeepMultifidelityGP_MultiOutputs', 
    'GradientGP', 
    'MultipleIndependentMFGP',
    'HeterogeneousMultifidelityGP',
    'MultipleIndependentHeterogeneousMFGP'
]