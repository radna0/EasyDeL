# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""EasyDeL Inference Module.

This module provides high-performance inference engines and APIs for running
large language models and speech recognition workloads efficiently on JAX/XLA.

Key Components:
    - eSurge: Unified attention inference engine with paged KV caches
    - vWhisper: Speech recognition and transcription engine
    - OpenAI API compatibility layer for seamless integration

Example:
    Running an OpenAI-compatible API server:

    >>> from easydel.inference import eSurgeApiServer, eSurge
    >>> surge = eSurge(...)
    >>> server = eSurgeApiServer(surge)
    >>> server.run(host="0.0.0.0", port=8000)

Attributes:
    FunctionCallFormat: Enum defining function call formatting styles
    FunctionCallFormatter: Utility for formatting function calls in prompts
    InferenceApiRouter: FastAPI router for OpenAI API compatibility
    JitableSamplingParams: JAX-compatible sampling parameters
    SamplingParams: High-level sampling configuration
    vWhisperInference: Speech recognition engine
    vWhisperInferenceConfig: Configuration for vWhisper
"""

from .sampling_params import JitableSamplingParams, SamplingParams
from .speculative.dflash import dflash_accept_len_and_bonus
from .speculative.dflash_decode import dflash_cached_decode_blockverify

# Optional imports (avoid forcing server deps like fastapi during TPU runs).
try:  # pragma: no cover
    from .esurge import EngineRequest, EngineRequestStatus, eSurge, eSurgeApiServer, eSurgeRunner
except Exception:  # pragma: no cover
    EngineRequest = EngineRequestStatus = eSurge = eSurgeApiServer = eSurgeRunner = None  # type: ignore

try:  # pragma: no cover
    from .oai_proxies import InferenceApiRouter
except Exception:  # pragma: no cover
    InferenceApiRouter = None  # type: ignore

try:  # pragma: no cover
    from .tools import ToolParser, ToolParserManager
except Exception:  # pragma: no cover
    ToolParser = ToolParserManager = None  # type: ignore

try:  # pragma: no cover
    from .vwhisper import vWhisperInference, vWhisperInferenceConfig
except Exception:  # pragma: no cover
    vWhisperInference = vWhisperInferenceConfig = None  # type: ignore

__all__ = (
    "JitableSamplingParams",
    "SamplingParams",
    "dflash_accept_len_and_bonus",
    "dflash_cached_decode_blockverify",
    # Optional exports (may be None if deps missing)
    "EngineRequest",
    "EngineRequestStatus",
    "InferenceApiRouter",
    "ToolParser",
    "ToolParserManager",
    "eSurge",
    "eSurgeApiServer",
    "eSurgeRunner",
    "vWhisperInference",
    "vWhisperInferenceConfig",
)
