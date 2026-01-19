# Copyright 2024 Haowen Yu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .dvsgesture.dataset import DVSGesture
from .dvsgesture.dataset_precomputed import DVSGesturePrecomputed, collate_fn
from .SparseVKMEncoder import VecKMSparse
from .create_datasets import create_dvsgesture_datasets, create_dvsgesture_dataloaders


def get_dataset_class(name: str):
    if name.lower() == 'dvsgesture':
        from .dvsgesture.dataset import DVSGesture
        return DVSGesture
    # Placeholder for HARDVS
    # elif name.lower() == 'hardvs':
    #     from .hardvs.dataset import HARDVS
    #     return HARDVS
    else:
        raise ValueError(f"Unknown dataset: {name}")

__all__ = [
    'DVSGesture',
    'DVSGesturePrecomputed',
    'VecKMSparse',
    'collate_fn',
    'create_dvsgesture_datasets',
    'create_dvsgesture_dataloaders',
    'get_dataset_class',
]
