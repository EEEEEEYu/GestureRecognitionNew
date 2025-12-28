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

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PrecomputingConfig:
    """Configuration for preprocessing/precomputing data."""
    dataset_dir: str = ""
    output_dir: str = ""
    accumulation_interval_ms: float = 100.0
    ratio_of_vectors: float = 0.1
    encoding_dim: int = 64
    temporal_length: float = 100.0
    kernel_size: int = 17
    T_scale: float = 25.0
    S_scale: float = 25.0
    height: int = 128
    width: int = 128
    num_workers: int = 4
    checkpoint_every_n_samples: int = 50
