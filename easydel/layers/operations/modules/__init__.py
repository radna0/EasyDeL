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

from .blocksparse_attention import BlockSparseAttn
from .decode_attention import AutoRegressiveDecodeAttn
from .flash_attention import FlashAttn
from .ragged_page_attention import RaggedPageAttnV2, RaggedPageAttnV3
from .ring_attention import RingAttn
from .scaled_dot_product_attention import ScaledDotProductAttn
from .unified_attention import UnifiedAttn
from .vanilla_attention import VanillaAttn

__all__ = [
    "AutoRegressiveDecodeAttn",
    "BlockSparseAttn",
    "FlashAttn",
    "RaggedPageAttnV2",
    "RaggedPageAttnV3",
    "RingAttn",
    "ScaledDotProductAttn",
    "UnifiedAttn",
    "VanillaAttn",
]

# Optional modules (skip if kernel deps not installed).
try:  # pragma: no cover
    from .gated_delta_rule import GatedDeltaRuleOp, GatedDeltaRuleOutput

    __all__ += ["GatedDeltaRuleOp", "GatedDeltaRuleOutput"]
except Exception:  # pragma: no cover
    class GatedDeltaRuleOp:  # type: ignore[dead-code]
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("GatedDeltaRuleOp requires optional ejkernel deps.")

    class GatedDeltaRuleOutput:  # type: ignore[dead-code]
        pass

    __all__ += ["GatedDeltaRuleOp", "GatedDeltaRuleOutput"]

try:  # pragma: no cover
    from .kda import KDAOutput, KernelDeltaAttnOp, fused_kda_gate

    __all__ += ["KDAOutput", "KernelDeltaAttnOp", "fused_kda_gate"]
except Exception:  # pragma: no cover
    class KernelDeltaAttnOp:  # type: ignore[dead-code]
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("KernelDeltaAttnOp requires optional ejkernel deps.")

    class KDAOutput:  # type: ignore[dead-code]
        pass

    def fused_kda_gate(*_args, **_kwargs):  # type: ignore[dead-code]
        raise NotImplementedError("fused_kda_gate requires optional ejkernel deps.")

    __all__ += ["KDAOutput", "KernelDeltaAttnOp", "fused_kda_gate"]

try:  # pragma: no cover
    from .ssm1 import SSM1Op, SSM1Output

    __all__ += ["SSM1Op", "SSM1Output"]
except Exception:  # pragma: no cover
    class SSM1Op:  # type: ignore[dead-code]
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("SSM1Op requires optional ejkernel deps.")

    class SSM1Output:  # type: ignore[dead-code]
        pass

    __all__ += ["SSM1Op", "SSM1Output"]

try:  # pragma: no cover
    from .ssm2 import SSM2Op, SSM2Output

    __all__ += ["SSM2Op", "SSM2Output"]
except Exception:  # pragma: no cover
    class SSM2Op:  # type: ignore[dead-code]
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("SSM2Op requires optional ejkernel deps.")

    class SSM2Output:  # type: ignore[dead-code]
        pass

    __all__ += ["SSM2Op", "SSM2Output"]

__all__ = tuple(__all__)
