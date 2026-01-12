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
from typing import List, Optional, Dict, Any


@dataclass
class BenchmarkingConfig:
    """Configuration for benchmarking denoising and sampling parameters."""
    num_samples: int = 100
    
    # Denoising benchmark parameter ranges
    denoising: Optional[Dict[str, Any]] = None
    
    # Sampling benchmark parameter ranges
    sampling: Optional[Dict[str, Any]] = None
    
    # Intermediate denoised data storage
    denoised_cache_dir: str = ""
