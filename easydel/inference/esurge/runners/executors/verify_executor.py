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

"""Verify-step compilation/execution for eSurge.

This executor is used by speculative decoding algorithms (e.g. DFlash) that
need logits for *every* token in a small verification window, rather than the
single next-token row used by normal sampling.

Key behavior:
  - Runs the model forward with KV-cache updates (same as the normal model step).
  - Returns:
      - hidden_states:     [num_tokens, hidden_dim]
      - greedy_token_ids:  [num_tokens]

Performance note:
  - This is intended for *small* token buckets (e.g. block_size * batch_size),
    not long prefill. Avoid calling it with large `num_tokens`.
"""

from __future__ import annotations

import typing as tp
from collections import OrderedDict

import jax
from eformer import escale as es
from flax import nnx as nn
from jax import numpy as jnp

from easydel.layers.caching import (
    HybridCache,
    RaggedPagesCache,
    RaggedPagesCacheConfig,
    RaggedPagesMetadata,
    UnifiedAttentionCache,
    UnifiedAttentionCacheConfig,
)
from easydel.utils import ejit

from ..execution_types import BatchMetadata, StepFunctionInputs, VerifyStepOutputs

if tp.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule


class VerifyStepExecutor:
    """Compile/cache and execute verify-mode model forward steps."""

    def __init__(
        self,
        *,
        model: "EasyDeLBaseModule",
        mesh: tp.Any,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
        kv_pages_template: HybridCache | RaggedPagesCache | UnifiedAttentionCache,
        graphstate_template: tp.Any,
        graphother_template: tp.Any,
        max_num_reqs: int,
        graphdef: tp.Any,
        empty_sharding: jax.sharding.Sharding,
        use_aot_forward: bool,
        cache_capacity: int = 16,
        target_layer_ids: list[int] | None = None,
        add_one_for_pre_layer_capture: bool = True,
        maybe_implicit: tp.Callable[[tp.Callable[..., tp.Any]], tp.Callable[..., tp.Any]] | None = None,
    ) -> None:
        self.model = model
        self.mesh = mesh
        self.metadata = metadata
        self.max_num_reqs = int(max_num_reqs)
        self.graphdef = graphdef
        self._metadata_version = metadata.version
        self._use_slot_mapping = self._metadata_version == "v2"
        self._empty_sharding = empty_sharding
        self.use_aot_forward = bool(use_aot_forward)
        self._cache_capacity = int(cache_capacity)
        self._maybe_implicit = maybe_implicit or (lambda f: f)
        self._target_layer_ids = list(target_layer_ids) if target_layer_ids is not None else None
        self._add_one_for_pre_layer_capture = bool(add_one_for_pre_layer_capture)

        self._verify_step_fn = self._build_verify_step_fn(
            kv_pages_template=kv_pages_template,
            graphstate_template=graphstate_template,
            graphother_template=graphother_template,
        )
        self._cache: OrderedDict[tuple[int, int, str], tp.Any] = OrderedDict()

    def clear_cache(self) -> None:
        self._cache.clear()

    def _cache_put(self, key: tuple[int, int, str], value: tp.Any) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_capacity:
            self._cache.popitem(last=False)

    def _cache_get(self, key: tuple[int, int, str]) -> tp.Any:
        value = self._cache[key]
        self._cache.move_to_end(key)
        return value

    def has(self, key: tuple[int, int, str]) -> bool:
        return key in self._cache

    def get_compiled(self, *, num_tokens: int, padded_num_reqs: int) -> tp.Any:
        mode = "aot" if self.use_aot_forward else "jit"
        key = (int(num_tokens), int(padded_num_reqs), mode)
        return self._cache_get(key)

    def compile(
        self,
        *,
        num_tokens: int,
        padded_num_reqs: int,
        graphdef: tp.Any,
        graphstate: tp.Any,
        graphother: tp.Any,
        inputs: StepFunctionInputs,
    ) -> VerifyStepOutputs | None:
        self.graphdef = graphdef
        mode = "aot" if self.use_aot_forward else "jit"
        key = (int(num_tokens), int(padded_num_reqs), mode)
        if key in self._cache:
            return None

        if self.use_aot_forward:
            compiled = self._verify_step_fn.lower(
                *(graphdef, graphstate, graphother, inputs.kv_pages, inputs.batch_metadata)
            ).compile()
            self._cache_put(key, compiled)
            return None

        def wrapped(graphstate_, graphother_, kv_pages_, metadata_):
            return self._verify_step_fn(self.graphdef, graphstate_, graphother_, kv_pages_, metadata_)

        out = wrapped(graphstate, graphother, inputs.kv_pages, inputs.batch_metadata)
        self._cache_put(key, wrapped)
        return out

    def _build_verify_step_fn(
        self,
        *,
        kv_pages_template: HybridCache | RaggedPagesCache | UnifiedAttentionCache,
        graphstate_template: tp.Any,
        graphother_template: tp.Any,
    ) -> tp.Callable[..., VerifyStepOutputs]:
        max_num_reqs = int(self.max_num_reqs)
        num_reqs_max_model_len = min(int(self.metadata.get_max_num_seqs()), max_num_reqs)

        metadata_sharding = BatchMetadata(
            packed_qsl_seqlens=self._empty_sharding,
            packed_i32_padded=self._empty_sharding,
            packed_f32_padded=self._empty_sharding,
            packed_misc_i32=self._empty_sharding,
            pages_tables=self._empty_sharding,
            input_ids_buf=self._empty_sharding,
            position_ids_buf=self._empty_sharding,
            slot_mapping=self._empty_sharding if self._use_slot_mapping else None,
            num_kv_update_slices=self._empty_sharding if self._use_slot_mapping else None,
            pixel_values=None,
            image_grid_thw=None,
            pixel_values_videos=None,
            video_grid_thw=None,
        )

        kv_pages_sharding = es.extract_shardings(kv_pages_template, self.mesh)

        outputs_shardings = VerifyStepOutputs(
            kv_pages=es.extract_shardings(kv_pages_template, self.mesh),
            context_features=self._empty_sharding,
            greedy_token_ids=self._empty_sharding,
        )

        @ejit(
            static_argnums=(0,),
            # Verify mode is called in irregular patterns (spec decode), and we
            # also JIT-compile on demand. Donating KV buffers here can make the
            # compile-time trace consume the live KV buffer and leave it invalid.
            # Keep it non-donating for correctness/stability on TPU.
            donate_argnames=[],
            in_shardings=(
                es.extract_shardings(graphstate_template, self.mesh),
                es.extract_shardings(graphother_template, self.mesh),
                kv_pages_sharding,
                metadata_sharding,
            ),
            out_shardings=outputs_shardings,
        )
        @self._maybe_implicit
        def _verify_step(
            graphdef,
            graphstate,
            graphother,
            kv_pages: HybridCache | RaggedPagesCache | UnifiedAttentionCache,
            metadata: BatchMetadata,
        ) -> VerifyStepOutputs:
            from easydel.inference.speculative.dflash import extract_dflash_context_features_from_hidden_states

            with self.model.mesh:
                model: "EasyDeLBaseModule" = nn.merge(graphdef, graphstate, graphother)
                input_ids_view = metadata.input_ids_buf
                position_ids_view = metadata.position_ids_buf

                cache_metadata = RaggedPagesMetadata(
                    pages_tables=metadata.pages_tables,
                    context_lens=metadata.seq_lens[:num_reqs_max_model_len],
                    query_start_loc=metadata.query_start_loc[: num_reqs_max_model_len + 1],
                    num_seqs=jnp.array([metadata.num_requests], dtype=jnp.int32),
                    num_slices_per_kv_cache_update_page=self.metadata.num_slices_per_kv_cache_update_page,
                    page_size=self.metadata.page_size,
                    request_distribution=metadata.request_distribution,
                    slot_mapping=metadata.slot_mapping,
                    num_kv_update_slices=metadata.num_kv_update_slices,
                    version=self._metadata_version,
                )

                # Keep the verify step minimal: no VLM/DeepStack extras for now.
                model_inputs = {"input_ids": jnp.expand_dims(input_ids_view, 0)}
                output = model(
                    **model_inputs,
                    position_ids=jnp.expand_dims(position_ids_view, 0),
                    past_key_values=kv_pages,
                    cache_metadata=cache_metadata,
                    apply_lm_head=True,
                    output_hidden_states=True,
                )
                hs = output.last_hidden_state.squeeze(0)

                if output.hidden_states is None:
                    raise ValueError("Verify mode requires output_hidden_states=True to provide context features.")
                if self._target_layer_ids is None:
                    raise ValueError("VerifyStepExecutor requires target_layer_ids for DFlash context features.")
                ctx = extract_dflash_context_features_from_hidden_states(
                    hidden_states=output.hidden_states,
                    target_layer_ids=self._target_layer_ids,
                    add_one_for_pre_layer_capture=self._add_one_for_pre_layer_capture,
                ).squeeze(0)
                if output.logits is None:
                    raise ValueError("Verify mode requires apply_lm_head=True to produce logits for greedy verification.")
                greedy_token_ids = jnp.argmax(output.logits.squeeze(0), axis=-1).astype(jnp.int32)

                return VerifyStepOutputs(
                    kv_pages=output.past_key_values,
                    context_features=ctx,
                    greedy_token_ids=greedy_token_ids,
                )

        return _verify_step
