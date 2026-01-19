from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import typing as tp


@dataclass(frozen=True)
class DFlashBenchResult:
    mode: str
    prompt_len: int
    max_new_tokens: int
    block_size: int
    accept_rate: float | None
    accepted_tokens: int | None
    proposed_tokens: int | None
    wall_s: float
    output_toks_per_s: float
    accept_len_mean: float | None = None
    accept_len_p50: float | None = None
    accept_len_p90: float | None = None
    draft_mask_rate: float | None = None


@dataclass(frozen=True)
class DFlashDecodeOutputs:
    dflash: DFlashBenchResult
    dflash_token_ids: list[int]
    baseline: DFlashBenchResult | None
    baseline_token_ids: list[int] | None


def _lm_head_logits(*, hidden, lm_head):
    """Project hidden states to vocab logits, handling kernel layout [H,V] vs [V,H]."""
    import jax
    import jax.numpy as jnp

    kernel = jax.lax.stop_gradient(lm_head.kernel.value)
    bias = jax.lax.stop_gradient(lm_head.bias.value) if getattr(lm_head, "bias", None) is not None else None
    if int(kernel.shape[0]) == int(hidden.shape[-1]):
        logits = jnp.einsum("bsh,hv->bsv", hidden, kernel, precision=jax.lax.Precision.HIGHEST)
    else:
        logits = jnp.einsum("bsh,vh->bsv", hidden, kernel, precision=jax.lax.Precision.HIGHEST)
    if bias is not None:
        logits = logits + bias[None, None, :]
    return logits


def _tp_greedy_argmax_from_vocab_shard(*, logits_sharded, mesh) -> "jax.Array":
    """Greedy argmax for vocab-parallel logits on TPU.

    `ColumnParallelLinear` (used for LM heads under TP) returns logits sharded on
    the vocab dimension across `mesh` axis `tp`. To get global argmax without
    gathering logits, do:
      - local argmax + local max
      - pmax over tp for global max
      - pmax over tp for global argmax among shards that hit global max

    Returns global token ids (int32) with shape [...], replicated across tp.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from jax.experimental.shard_map import shard_map

    if "tp" not in getattr(mesh, "axis_names", ()):
        # No TP axis: normal argmax.
        return jnp.argmax(logits_sharded, axis=-1).astype(jnp.int32)

    tp_size = int(getattr(mesh, "shape", {}).get("tp", 1) or 1)
    if tp_size <= 1:
        return jnp.argmax(logits_sharded, axis=-1).astype(jnp.int32)

    def _per_shard_argmax(x_local):
        x_f = x_local.astype(jnp.float32)
        local_max = jnp.max(x_f, axis=-1)
        local_arg = jnp.argmax(x_f, axis=-1).astype(jnp.int32)
        v_local = int(x_local.shape[-1])
        tp_idx = jax.lax.axis_index("tp").astype(jnp.int32)
        arg_global = local_arg + tp_idx * jnp.asarray(v_local, dtype=jnp.int32)
        global_max = jax.lax.pmax(local_max, "tp")
        score = jnp.where(local_max == global_max, arg_global, jnp.asarray(-1, dtype=jnp.int32))
        return jax.lax.pmax(score, "tp").astype(jnp.int32)

    # Assume logits are sharded on the last dim (vocab) by tp.
    in_specs = P(None, None, "tp") if getattr(logits_sharded, "ndim", 0) == 3 else P(None, "tp")
    out_specs = P(None, None) if getattr(logits_sharded, "ndim", 0) == 3 else P(None)
    return shard_map(_per_shard_argmax, mesh=mesh, in_specs=in_specs, out_specs=out_specs)(logits_sharded)


def _chunked_argmax_from_lm_head_weight(*, hidden, lm_w, vocab_chunk: int) -> "jax.Array":
    """Argmax over vocab using frozen LM head weight [V,H], optionally chunked.

    This matches the training-side logic in `DFlashTrainer` and is correct even
    when `lm_w` is sharded across devices (JAX SPMD will lower reductions).
    """
    import jax
    import jax.numpy as jnp

    hs = hidden.astype(jnp.bfloat16)
    vocab_size = int(lm_w.shape[0])
    seq_len = int(hs.shape[1])
    bsz = int(hs.shape[0])

    if int(vocab_chunk) <= 0 or int(vocab_chunk) >= vocab_size:
        logits = jnp.einsum("bsh,vh->bsv", hs, lm_w, precision=jax.lax.Precision.HIGHEST)
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    best_val = jnp.full((bsz, seq_len), -jnp.inf, dtype=jnp.float32)
    best_ids = jnp.zeros((bsz, seq_len), dtype=jnp.int32)
    chunk = int(vocab_chunk)
    for start in range(0, vocab_size, chunk):
        end = start + chunk
        w = lm_w[start:end, :]
        chunk_logits = jnp.einsum(
            "bsh,vh->bsv",
            hs,
            w,
            precision=jax.lax.Precision.HIGHEST,
        ).astype(jnp.float32)
        chunk_best_local = jnp.argmax(chunk_logits, axis=-1).astype(jnp.int32)
        chunk_best_val = jnp.take_along_axis(chunk_logits, chunk_best_local[..., None], axis=-1)[..., 0]
        take_chunk = chunk_best_val > best_val
        best_val = jnp.where(take_chunk, chunk_best_val, best_val)
        best_ids = jnp.where(take_chunk, chunk_best_local + jnp.int32(start), best_ids)
    return best_ids.astype(jnp.int32)


def load_dflash_draft_from_run_dir(*, run_dir: str | Path, cfg, mesh):
    """Load a DFlashDraftModel + graphstate from an EasyDeL run-* directory."""
    from flax import nnx

    from .dflash_checkpoint import load_dflash_graphstate_from_run
    from .dflash_draft_model import DFlashDraftModel

    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"Missing run_dir: {run_path}")

    rngs = nnx.Rngs(0)
    template = DFlashDraftModel(cfg, rngs=rngs)
    graphdef, template_graphstate, graphother = nnx.split(template, nnx.Param, ...)
    graphstate = load_dflash_graphstate_from_run(
        run_dir=run_path,
        mesh=mesh,
        template_graphstate=template_graphstate,
        partition_rules=getattr(cfg, "get_partition_rules", lambda: [])(),
        strict_shapes=True,
    )
    draft = nnx.merge(graphdef, graphstate, graphother)
    return draft


def bench_esurge_dflash_decode_single(
    *,
    teacher,
    draft_run_dir: str | Path,
    draft_cfg,
    target_rope,
    lm_head_weight=None,
    prompt_ids,
    position_offset: int = 0,
    max_new_tokens: int,
    block_size: int,
    max_model_len: int,
    page_size: int,
    hbm_utilization: float,
    also_run_baseline: bool = True,
) -> tuple[DFlashBenchResult, DFlashBenchResult | None]:
    """Backward-compatible wrapper returning only metrics (no token IDs)."""
    outs = esurge_dflash_decode_single(
        teacher=teacher,
        draft_run_dir=draft_run_dir,
        draft_cfg=draft_cfg,
        target_rope=target_rope,
        lm_head_weight=lm_head_weight,
        prompt_ids=prompt_ids,
        position_offset=int(position_offset),
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        max_model_len=max_model_len,
        page_size=page_size,
        hbm_utilization=hbm_utilization,
        also_run_baseline=also_run_baseline,
    )
    return outs.dflash, outs.baseline


def esurge_dflash_decode_single(
    *,
    teacher,
    draft_run_dir: str | Path,
    draft_cfg,
    target_rope,
    lm_head_weight=None,
    prompt_ids,
    position_offset: int = 0,
    max_new_tokens: int,
    block_size: int,
    max_model_len: int,
    page_size: int,
    hbm_utilization: float,
    also_run_baseline: bool = True,
) -> DFlashDecodeOutputs:
    """Single-request TPU DFlash decode using eSurge verify executor.

    Correctness rules match SGLang DFlash spec-v1:
    - Prefill uses verify-mode (so we can capture context features).
    - Draft proposes (block_size-1) tokens.
    - Target verifies the whole block greedily.
    - Accept while draft[t] == target[t-1] consecutively.
    - Commit accepted draft tokens + a bonus target token.
    """
    import numpy as np
    import os
    import jax
    import jax.numpy as jnp

    from easydel.inference.esurge.runners.sequence_buffer import SequenceBuffer
    from easydel.inference.esurge.runners.execution_manager import ExecutionManager
    from easydel.inference.esurge.runners.states import CachedRequestState
    from easydel.inference.sampling_params import SamplingParams

    from .dflash import dflash_accept_len_and_bonus
    from .dflash_kv_cache import (
        append_draft_ctx_kv,
        append_draft_ctx_kv_windowed,
        append_draft_ctx_kv_windowed_committed,
        draft_forward_with_ctx_kv,
        materialize_draft_ctx_kv,
    )

    prompt_ids = np.asarray(prompt_ids, dtype=np.int32)
    if prompt_ids.ndim != 1 or prompt_ids.size < 2:
        raise ValueError("prompt_ids must be 1D and have >=2 tokens")
    if int(block_size) <= 1:
        raise ValueError("block_size must be > 1")

    mesh = teacher.mesh
    text_cfg = teacher.config.get_text_config()
    vocab_size = int(text_cfg.vocab_size)
    empty_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())

    seqbuf = SequenceBuffer(
        max_num_reqs=1,
        max_model_len=int(max_model_len),
        max_num_batched_tokens=int(max_model_len),
        vocab_size=vocab_size,
        page_sizes=[int(page_size)],
        sharding=empty_sharding,
    )
    max_pages_per_req = int(
        getattr(
            teacher.create_ragged_page_cache_config(
                hbm_utilization=float(hbm_utilization),
                page_size=int(page_size),
                max_length=int(max_model_len),
            ),
            "max_num_pages_per_req",
        )
    )
    page_ids = (list(range(max_pages_per_req)),)

    rid = "bench-req"
    sp = SamplingParams(max_tokens=int(max_new_tokens), temperature=0.0, top_k=1, top_p=1.0)
    req_state = CachedRequestState(
        req_id=rid,
        prompt_token_ids=prompt_ids.tolist(),
        sampling_params=sp,
        generator=jax.random.PRNGKey(0),
        page_ids=page_ids,
        num_computed_tokens=0,
        output_token_ids=[],
    )
    seqbuf.add_request(req_state, req_index=0)

    metadata = teacher.create_ragged_page_cache_config(
        hbm_utilization=float(hbm_utilization),
        page_size=int(page_size),
        max_length=int(max_model_len),
    )
    saved_layer_ids = getattr(draft_cfg, "target_layer_ids", None)
    if not saved_layer_ids:
        raise ValueError("draft_cfg.target_layer_ids must be set for eSurge DFlash parity.")
    target_layer_ids = [int(x) for x in saved_layer_ids]

    executor = ExecutionManager(
        model=teacher.esurge_compatible_model,
        use_aot_forward=True,
        min_input_pad=1,
        max_model_len=int(max_model_len),
        max_num_reqs=1,
        max_num_tokens=int(max_model_len),
        metadata=metadata,
        verbose=False,
        verify_target_layer_ids=target_layer_ids,
        verify_add_one_for_pre_layer_capture=bool(getattr(draft_cfg, "add_one_for_pre_layer_capture", True)),
    )

    prompt_len = int(prompt_ids.size)
    total_prefill = int(prompt_len - 1)  # exclude current token (SGLang-style)
    prefill_chunk = int(max(1, int(os.environ.get("DFLASH_PREFILL_CHUNK", "256"))))
    # IMPORTANT (TPU stability): keep verify buckets fixed.
    # Passing `num_tokens=<remainder>` triggers on-demand compilation of a new
    # token bucket (e.g. 255), which can segfault in XLA/jellyfish when
    # pre-layer capture is enabled. Instead, always call verify with a fixed
    # precompiled bucket and use `scheduled_full_cpu` to represent the real
    # (possibly smaller) work.
    prefill_bucket = int(min(int(prefill_chunk), int(max_model_len)))
    executor.compile(
        num_tokens_paddings=sorted({1, int(block_size), int(max(1, prefill_bucket))}),
        num_reqs_max_model_len=1,
        max_pages_per_req=int(metadata.max_num_pages_per_req),
        max_num_reqs=1,
        metadata=metadata,
        num_reqs_paddings=[1],
    )

    input_ids_buf = jax.device_put(jnp.zeros((int(max_model_len),), dtype=jnp.int32), empty_sharding)
    position_ids_buf = jax.device_put(jnp.zeros((int(max_model_len),), dtype=jnp.int32), empty_sharding)
    scheduled_full_cpu = np.zeros((1,), dtype=np.int32)
    active_mask_full_cpu = np.asarray([True], dtype=bool)
    page_table_cpu = seqbuf.page_table[0].get_cpu_tensor()
    page_table_version = getattr(seqbuf.page_table[0], "cpu_version", None)
    pos_off_cpu = np.asarray([np.int32(int(position_offset))], dtype=np.int32)

    # Greedy.
    seqbuf.temperature[0] = 0.0
    seqbuf.top_k[0] = 1
    seqbuf.top_p[0] = 1.0
    seqbuf.min_p[0] = 0.0
    seqbuf.num_tokens[0] = int(prompt_len)
    seqbuf.num_tokens_no_spec[0] = int(prompt_len)

    # ---- Prefill prefix (exclude current token) in chunks; also capture ctx features. ----
    ctx_parts = []
    seqbuf.num_computed_tokens[0] = 0
    done = 0
    while done < total_prefill:
        step = int(min(int(prefill_chunk), total_prefill - done))
        seqbuf.num_computed_tokens[0] = int(done)
        scheduled_full_cpu[0] = int(step)
        ctx_part, _greedy_unused, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
            num_tokens=int(prefill_bucket),
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            input_ids_buf=input_ids_buf,
            position_ids_buf=position_ids_buf,
            padded_num_reqs=1,
            token_ids_cpu=seqbuf.token_ids,
            num_computed_tokens_cpu=seqbuf.num_computed_tokens,
            position_offset_cpu=pos_off_cpu,
            temperature_cpu=seqbuf.temperature,
            top_p_cpu=seqbuf.top_p,
            top_k_cpu=seqbuf.top_k,
            min_p_cpu=seqbuf.min_p,
            page_table_cpu=page_table_cpu,
            page_table_version=page_table_version,
        )
        ctx_parts.append(jnp.asarray(ctx_part)[:step, :])
        done += step

    ctx_prefill_full = jnp.concatenate(ctx_parts, axis=0) if ctx_parts else jnp.zeros((0, 0), dtype=jnp.bfloat16)
    ctx_feat = ctx_prefill_full.reshape((1, int(total_prefill), -1)).astype(jnp.bfloat16)
    # Ensure stable sharding for draft-mode JITs (avoid UnspecifiedValue sharding).
    ctx_feat = jax.device_put(ctx_feat, empty_sharding)
    # Draft mode:
    # - ctx_kv: materialize ctx KV once and append committed tokens (fast, but ctx grows).
    # - direct_window: recompute draft(...) each block over a fixed-size ctx window (training-parity).
    draft_mode = os.environ.get("DFLASH_DRAFT_MODE", "ctx_kv").strip().lower()
    if draft_mode not in ("ctx_kv", "direct_window"):
        raise ValueError(f"Unknown DFLASH_DRAFT_MODE={draft_mode!r} (expected 'ctx_kv' or 'direct_window').")

    # Prepare decode state invariant:
    # num_computed_tokens == num_tokens - 1, pending token at [num_computed_tokens].
    seqbuf.num_computed_tokens[0] = int(prompt_len - 1)
    seqbuf.num_tokens[0] = int(prompt_len)
    seqbuf.num_tokens_no_spec[0] = int(prompt_len)

    draft = load_dflash_draft_from_run_dir(run_dir=draft_run_dir, cfg=draft_cfg, mesh=mesh)
    # Materialize draft ctx KV once; then append committed ctx tokens.
    with mesh:
        ctx_hidden = draft.project_context_features(ctx_feat)
        ctx_kv = None
        ctx_feat_win = ctx_feat
        # By default, keep a fixed-size window equal to the prompt prefix length.
        # For cache-parity training runs, prompt_len == ctx_len + 1 so this matches.
        ctx_window = int(os.environ.get("DFLASH_CTX_WINDOW", str(int(ctx_feat.shape[1]))))
        ctx_window = max(0, min(int(ctx_feat.shape[1]), int(ctx_window)))
        if draft_mode == "direct_window":
            ctx_feat_win = ctx_feat[:, -ctx_window:, :]
            ctx_feat_win = jax.device_put(ctx_feat_win, empty_sharding)
        else:
            # In ctx_kv mode, allocate only what we need for the draft KV cache.
            # Default to a fixed rolling window (training-parity) to keep TPU
            # memory usage stable. Full-length radix-cache behavior is a later
            # step.
            ctxkv_windowed = os.environ.get("DFLASH_CTXKV_WINDOWED", "1").lower() in ("1", "true", "yes", "y", "on")
            kv_ctx_window = int(ctx_window) if bool(ctxkv_windowed) else int(max_model_len)
            kv_max_len = int(kv_ctx_window + int(block_size))
            # We only cache the last `kv_ctx_window` tokens, but they still live
            # at the *end* of the prompt in absolute RoPE space.
            pos_start0 = int(int(position_offset) + max(0, int(total_prefill) - int(kv_ctx_window)))
            ctx_kv = materialize_draft_ctx_kv(
                draft=draft,
                rope=target_rope,
                ctx_hidden=ctx_hidden[:, -kv_ctx_window:, :],
                max_len=int(kv_max_len),
                pos_start=int(pos_start0),
            )

        # JIT the draft propose + ctx-KV append so we don't pay Python dispatch
        # overhead for every verify block. This is required for any realistic
        # throughput measurement on TPU.
        #
        # IMPORTANT (TPU):
        # - Do NOT pass NNX Modules (draft/embedding/lm_head) as JIT arguments.
        #   They are pytrees of Params and become tracers; calling them inside
        #   JIT then fails ("DynamicJaxprTracer is not callable").
        # - Do NOT force replicated in_shardings for large Params (embedding /
        #   lm_head), or we will replicate hundreds of MB/GB across 8 chips.
        #
        # Instead: split modules into (graphdef, graphstate, graphother) once,
        # pass only graphstate (arrays) into JIT, and reconstruct the Module via
        # nnx.merge inside the traced function. This keeps weights as runtime
        # parameters with their existing shardings and avoids XLA constant-bloat.
        from flax import nnx

        embedding_mod = teacher.get_embedding()
        lm_head_mod = teacher.get_lm_head()
        draft_graphdef, draft_graphstate, draft_graphother = nnx.split(draft, nnx.Param, ...)
        embed_graphdef, embed_graphstate, embed_graphother = nnx.split(embedding_mod, nnx.Param, ...)
        head_graphdef, head_graphstate, head_graphother = nnx.split(lm_head_mod, nnx.Param, ...)
        # NOTE:
        # For TPU throughput and correctness, we prefer using the target model's
        # LM head Module (which already handles TP sharding) over capturing a
        # raw `lm_head_weight` array in a jitted function. Closed-over sharded
        # arrays can be treated as XLA constants under `jax.jit`, which can
        # silently reshard/replicate and produce wrong argmax tokens (collapsing
        # accept_len).
        #
        # `DFLASH_DRAFT_USE_LM_W=1` is therefore only supported in *non-jitted*
        # draft mode (DFLASH_JIT_DRAFT=0) for debugging / parity checks.
        use_lm_w = os.environ.get("DFLASH_DRAFT_USE_LM_W", "0").lower() in ("1", "true", "yes", "y", "on")
        lm_w = lm_head_weight if bool(use_lm_w) else None

        def _draft_propose_ctx_kv(draft_state, embed_state, head_state, ctx_kv_in, cur_id_in):
            cur_id_in = jnp.asarray(cur_id_in, dtype=jnp.int32)
            draft_mod = nnx.merge(draft_graphdef, draft_state, draft_graphother)
            embed_mod = nnx.merge(embed_graphdef, embed_state, embed_graphother)
            head_mod = nnx.merge(head_graphdef, head_state, head_graphother)
            anchor_emb = embed_mod(cur_id_in.reshape((1, 1)))[:, 0, :].astype(jnp.bfloat16)
            d_hidden = draft_forward_with_ctx_kv(
                draft=draft_mod,
                rope=target_rope,
                cache=ctx_kv_in,
                anchor_embedding=anchor_emb,
                mask_embedding=draft_mod.mask_embedding.value.astype(jnp.bfloat16),
                block_size=int(block_size),
            )
            hs_d = d_hidden[:, 1:, :]  # [1, B-1, hidden]
            if lm_w is not None:
                vocab_chunk = int(os.environ.get("DFLASH_LM_W_CHUNK", "32768"))
                toks = _chunked_argmax_from_lm_head_weight(hidden=hs_d, lm_w=lm_w, vocab_chunk=vocab_chunk)
                return toks.astype(jnp.int32)[0]
            logits_local = head_mod(hs_d.astype(jnp.bfloat16))
            toks = _tp_greedy_argmax_from_vocab_shard(logits_sharded=logits_local, mesh=mesh)
            return toks.astype(jnp.int32)[0]  # [B-1]

        def _append_ctx_kv(draft_state, ctx_kv_in, ctx_commit_feat_full_in, commit_len_in):
            draft_mod = nnx.merge(draft_graphdef, draft_state, draft_graphother)
            new_ctx_hidden_full = draft_mod.project_context_features(ctx_commit_feat_full_in.astype(jnp.bfloat16))
            ctxkv_windowed = os.environ.get("DFLASH_CTXKV_WINDOWED", "1").lower() in ("1", "true", "yes", "y", "on")
            if bool(ctxkv_windowed):
                return append_draft_ctx_kv_windowed_committed(
                    draft=draft_mod,
                    rope=target_rope,
                    cache=ctx_kv_in,
                    new_ctx_hidden_full=new_ctx_hidden_full,
                    commit_len=commit_len_in,
                    ctx_window=int(ctx_window),
                )
            raise ValueError("DFLASH_CTXKV_WINDOWED must be enabled for ctx_kv mode on TPU.")

        def _draft_propose_direct(draft_state, embed_state, head_state, ctx_feat_in, cur_id_in, ctx_pos_start_in):
            cur_id_in = jnp.asarray(cur_id_in, dtype=jnp.int32)
            draft_mod = nnx.merge(draft_graphdef, draft_state, draft_graphother)
            embed_mod = nnx.merge(embed_graphdef, embed_state, embed_graphother)
            head_mod = nnx.merge(head_graphdef, head_state, head_graphother)
            anchor_emb = embed_mod(cur_id_in.reshape((1, 1)))[:, 0, :].astype(jnp.bfloat16)
            d_hidden = draft_mod(
                context_features=ctx_feat_in,
                anchor_embedding=anchor_emb,
                rope=target_rope,
                ctx_pos_start=ctx_pos_start_in,
            )
            hs_d = d_hidden[:, 1:, :]  # [1, B-1, hidden]
            if lm_w is not None:
                vocab_chunk = int(os.environ.get("DFLASH_LM_W_CHUNK", "32768"))
                toks = _chunked_argmax_from_lm_head_weight(hidden=hs_d, lm_w=lm_w, vocab_chunk=vocab_chunk)
                return toks.astype(jnp.int32)[0]
            logits_local = head_mod(hs_d.astype(jnp.bfloat16))
            toks = _tp_greedy_argmax_from_vocab_shard(logits_sharded=logits_local, mesh=mesh)
            return toks.astype(jnp.int32)[0]

        def _append_ctx_feat_window(ctx_feat_in, ctx_commit_feat_in):
            ctx_feat_out = jnp.concatenate([ctx_feat_in, ctx_commit_feat_in.astype(jnp.bfloat16)], axis=1)
            if int(ctx_window) <= 0:
                return ctx_feat_out[:, :0, :]
            return ctx_feat_out[:, -int(ctx_window) :, :]

        # JIT is normally required for realistic throughput on TPU, but large
        # draft/KV pytrees can trigger very large XLA programs which then cause
        # verify executables to fail loading (RESOURCE_EXHAUSTED). Allow
        # disabling JIT for debugging correctness and accept_len behavior.
        jit_draft = os.environ.get("DFLASH_JIT_DRAFT", "1").lower() in ("1", "true", "yes", "y", "on")
        if bool(jit_draft) and lm_w is not None:
            # Never allow a raw LM-head weight capture under jit; see note above.
            if jax.process_index() == 0:
                print(
                    "[dflash] disabling DFLASH_DRAFT_USE_LM_W under jit to avoid constant capture",
                    flush=True,
                )
            lm_w = None

        if bool(jit_draft):
            # Let JAX preserve existing parameter shardings for draft/embedding/lm_head
            # (replicating them would explode HBM and/or XLA program size).
            if ctx_kv is not None:
                _draft_propose_ctx_kv = jax.jit(_draft_propose_ctx_kv)
                _append_ctx_kv = jax.jit(_append_ctx_kv)

            _draft_propose_direct = jax.jit(_draft_propose_direct)
            _append_ctx_feat_window = jax.jit(
                _append_ctx_feat_window,
                in_shardings=(empty_sharding, empty_sharding),
                out_shardings=empty_sharding,
            )

    accepted = 0
    proposed = 0
    accept_lens: list[int] = []
    generated = 0
    dflash_out: list[int] = []
    # Absolute position of token 0 in `ctx_feat_win` (direct_window mode).
    ctx_pos_start_win = int(int(position_offset) + max(0, int(total_prefill) - int(ctx_window)))
    log_every_blocks = int(os.environ.get("DFLASH_LOG_EVERY_BLOCKS", "0"))
    debug_tokens = os.environ.get("DFLASH_DEBUG_TOKENS", "0").lower() in ("1", "true", "yes", "y", "on")
    debug_tokens_blocks = int(os.environ.get("DFLASH_DEBUG_TOKENS_BLOCKS", "1"))
    rollback_recompute = os.environ.get("DFLASH_VERIFY_ROLLBACK_RECOMPUTE", "0").lower() in ("1", "true", "yes", "y", "on")

    # ---- DFlash decode (verify blocks) ----
    t0 = time.time()
    while int(generated) < int(max_new_tokens):
        base_len = int(seqbuf.num_computed_tokens[0])
        cur_id = int(seqbuf.token_ids[0, base_len])

        with mesh:
            if draft_mode == "ctx_kv":
                draft_tokens = _draft_propose_ctx_kv(
                    draft_graphstate,
                    embed_graphstate,
                    head_graphstate,
                    ctx_kv,
                    jnp.asarray(cur_id, dtype=jnp.int32),
                )
            else:
                draft_tokens = _draft_propose_direct(
                    draft_graphstate,
                    embed_graphstate,
                    head_graphstate,
                    ctx_feat_win,
                    jnp.asarray(cur_id, dtype=jnp.int32),
                    jnp.asarray(int(ctx_pos_start_win), dtype=jnp.int32),
                )

        # Candidates: [cur] + draft tokens.
        cand = np.empty((int(block_size),), dtype=np.int32)
        cand[0] = np.int32(cur_id)
        # On older JAX builds, `np.asarray(jax.Array)` can crash with
        # `UnspecifiedValue` sharding objects. Always device_get first.
        cand[1:] = np.asarray(jax.device_get(draft_tokens), dtype=np.int32)
        seqbuf.token_ids[0, base_len : base_len + int(block_size)] = cand
        seqbuf.num_tokens[0] = int(base_len + int(block_size))
        seqbuf.num_tokens_no_spec[0] = int(base_len + int(block_size))

        scheduled_full_cpu[0] = int(block_size)
        kv_before = executor.kv_pages
        ctx_part, greedy_ids, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
            num_tokens=int(block_size),
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            input_ids_buf=input_ids_buf,
            position_ids_buf=position_ids_buf,
            padded_num_reqs=1,
            token_ids_cpu=seqbuf.token_ids,
            num_computed_tokens_cpu=seqbuf.num_computed_tokens,
            position_offset_cpu=pos_off_cpu,
            temperature_cpu=seqbuf.temperature,
            top_p_cpu=seqbuf.top_p,
            top_k_cpu=seqbuf.top_k,
            min_p_cpu=seqbuf.min_p,
            page_table_cpu=page_table_cpu,
            page_table_version=page_table_version,
        )
        greedy_ids = jnp.asarray(greedy_ids)[None, :]  # [1,B]
        cand_j = jnp.asarray(cand[None, :], dtype=jnp.int32)  # [1,B]

        accept_len, bonus = dflash_accept_len_and_bonus(candidates=cand_j, target_predict=greedy_ids)
        n_acc = int(jnp.asarray(accept_len)[0])
        accept_lens.append(int(n_acc))
        keep = 1 + n_acc

        # Critical correctness fix (optional): if we don't accept the full draft
        # block, we must NOT leave unaccepted tokens' KV in the target cache.
        #
        # In SGLang DFlash, this is handled by verify-mode cache commit/rollback.
        # EasyDeL's eSurge verify currently always mutates KV for the entire
        # `block_size` window, so we provide a correctness-first fallback:
        #   - rollback KV to pre-verify state
        #   - re-run verify on only the committed prefix (`keep` tokens)
        # This makes the target cache identical to baseline greedy, at the cost
        # of extra work only when keep < block_size.
        ctx_part_used = ctx_part
        if bool(rollback_recompute) and int(keep) < int(block_size):
            executor.kv_pages = kv_before
            seqbuf.num_tokens[0] = int(base_len + int(keep))
            seqbuf.num_tokens_no_spec[0] = int(base_len + int(keep))
            scheduled_full_cpu[0] = int(keep)
            ctx_part_used, _greedy_unused, input_ids_buf, position_ids_buf, _m2 = executor.execute_verify(
                num_tokens=int(block_size),
                scheduled_full_cpu=scheduled_full_cpu,
                active_mask_full_cpu=active_mask_full_cpu,
                input_ids_buf=input_ids_buf,
                position_ids_buf=position_ids_buf,
                padded_num_reqs=1,
                token_ids_cpu=seqbuf.token_ids,
                num_computed_tokens_cpu=seqbuf.num_computed_tokens,
                position_offset_cpu=pos_off_cpu,
                temperature_cpu=seqbuf.temperature,
                top_p_cpu=seqbuf.top_p,
                top_k_cpu=seqbuf.top_k,
                min_p_cpu=seqbuf.min_p,
                page_table_cpu=page_table_cpu,
                page_table_version=page_table_version,
            )
            scheduled_full_cpu[0] = int(block_size)

        if bool(debug_tokens) and int(len(accept_lens)) <= int(debug_tokens_blocks):
            try:
                cand_cpu = np.asarray(cand, dtype=np.int32).tolist()
                greedy_cpu = np.asarray(jax.device_get(greedy_ids[0]), dtype=np.int32).tolist()
                matches_cpu = [int(a == b) for a, b in zip(cand_cpu[1:], greedy_cpu[:-1])]
                print(
                    "[dflash][debug] "
                    f"cand={cand_cpu} "
                    f"greedy={greedy_cpu} "
                    f"matches(cand[1:]==greedy[:-1])={matches_cpu}",
                    flush=True,
                )
            except Exception as e:
                print(f"[dflash][debug] token dump failed: {type(e).__name__}: {e}", flush=True)

        # Append committed tokens' ctx features so drafting conditions on what was actually verified.
        ctx_commit_feat_full = (
            jnp.asarray(ctx_part_used)[: int(block_size), :].reshape((1, int(block_size), -1)).astype(jnp.bfloat16)
        )
        with mesh:
            if draft_mode == "ctx_kv":
                ctx_kv = _append_ctx_kv(
                    draft_graphstate,
                    ctx_kv,
                    ctx_commit_feat_full,
                    jnp.asarray(int(keep), dtype=jnp.int32),
                )
            else:
                # Direct-window mode is for correctness debugging. Keep a
                # separate absolute-position counter so RoPE parity holds.
                cur_len = int(ctx_feat_win.shape[1])
                drop = max(0, int(cur_len + int(keep) - int(ctx_window)))
                ctx_pos_start_win += int(drop)
                ctx_feat_win = _append_ctx_feat_window(ctx_feat_win, ctx_commit_feat_full[:, : int(keep), :])

        # Commit: advance computed tokens and set bonus as the next pending token.
        seqbuf.num_computed_tokens[0] = int(base_len + keep)
        seqbuf.token_ids[0, int(base_len + keep)] = np.int32(jnp.asarray(bonus)[0])
        seqbuf.num_tokens[0] = int(base_len + keep + 1)
        seqbuf.num_tokens_no_spec[0] = int(base_len + keep + 1)

        accepted += n_acc
        proposed += int(block_size) - 1
        generated += int(keep)
        # Track emitted tokens (accepted draft + bonus) for sanity checks.
        # Bonus token is always appended, even if accept_len == 0.
        remaining = int(max_new_tokens) - int(len(dflash_out))
        if remaining > 0:
            emitted: list[int] = []
            if int(n_acc) > 0:
                emitted.extend([int(x) for x in cand[1 : 1 + int(n_acc)].tolist()])
            emitted.append(int(jnp.asarray(bonus)[0]))
            dflash_out.extend(emitted[:remaining])

        if log_every_blocks > 0 and (len(accept_lens) % int(log_every_blocks) == 0):
            kv_pos_start = "n/a"
            kv_ctx_len = "n/a"
            if draft_mode == "ctx_kv" and ctx_kv is not None:
                try:
                    pos = np.asarray(jax.device_get(ctx_kv.pos_start))
                    if getattr(pos, "ndim", 0) == 0:
                        kv_pos_start = str(int(pos))
                    else:
                        kv_pos_start = str(int(pos.reshape((-1,))[0]))
                    kv_ctx_len = str(int(np.asarray(jax.device_get(ctx_kv.ctx_len))))
                except Exception:
                    kv_pos_start = "err"
                    kv_ctx_len = "err"
            print(
                "[dflash] "
                f"block={len(accept_lens)} "
                f"abs_pos={base_len} "
                f"keep={keep} "
                f"accept_len={n_acc} "
                f"ctx_pos_start_win={ctx_pos_start_win if draft_mode!='ctx_kv' else 'n/a'} "
                f"ctx_kv_pos_start={kv_pos_start} "
                f"ctx_kv_len={kv_ctx_len}",
                flush=True,
            )

    dt = max(1e-9, time.time() - t0)
    accept_len_mean = float(sum(accept_lens) / len(accept_lens)) if accept_lens else None
    accept_len_p50 = float(np.percentile(accept_lens, 50)) if accept_lens else None
    accept_len_p90 = float(np.percentile(accept_lens, 90)) if accept_lens else None
    dflash_res = DFlashBenchResult(
        mode="dflash_blockverify_tpu",
        prompt_len=int(prompt_len),
        max_new_tokens=int(max_new_tokens),
        block_size=int(block_size),
        accept_rate=float(accepted) / float(max(1, proposed)),
        accepted_tokens=int(accepted),
        proposed_tokens=int(proposed),
        wall_s=float(dt),
        output_toks_per_s=float(int(generated)) / float(dt),
        accept_len_mean=accept_len_mean,
        accept_len_p50=accept_len_p50,
        accept_len_p90=accept_len_p90,
        draft_mask_rate=0.0,
    )

    if not bool(also_run_baseline):
        return DFlashDecodeOutputs(
            dflash=dflash_res,
            dflash_token_ids=dflash_out,
            baseline=None,
            baseline_token_ids=None,
        )

    # ---- Baseline greedy (verify-mode, 1 token step) ----
    executor = ExecutionManager(
        model=teacher.esurge_compatible_model,
        use_aot_forward=True,
        min_input_pad=1,
        max_model_len=int(max_model_len),
        max_num_reqs=1,
        max_num_tokens=int(max_model_len),
        metadata=metadata,
        verbose=False,
        verify_target_layer_ids=target_layer_ids,
        verify_add_one_for_pre_layer_capture=bool(getattr(draft_cfg, "add_one_for_pre_layer_capture", True)),
    )
    executor.compile(
        # Include `block_size` so baseline decode can reuse the same fixed token
        # bucket as DFlash verify (keeps outputs comparable on TPU).
        num_tokens_paddings=sorted({1, int(block_size), int(max(1, prefill_bucket))}),
        num_reqs_max_model_len=1,
        max_pages_per_req=int(metadata.max_num_pages_per_req),
        max_num_reqs=1,
        metadata=metadata,
        num_reqs_paddings=[1],
    )
    # Re-run prefill (exclude current token).
    seqbuf.num_computed_tokens[0] = 0
    done = 0
    while done < total_prefill:
        step = int(min(int(prefill_chunk), total_prefill - done))
        seqbuf.num_computed_tokens[0] = int(done)
        scheduled_full_cpu[0] = int(step)
        _ctx_unused, _greedy_unused, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
            num_tokens=int(prefill_bucket),
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            input_ids_buf=input_ids_buf,
            position_ids_buf=position_ids_buf,
            padded_num_reqs=1,
            token_ids_cpu=seqbuf.token_ids,
            num_computed_tokens_cpu=seqbuf.num_computed_tokens,
            position_offset_cpu=pos_off_cpu,
            temperature_cpu=seqbuf.temperature,
            top_p_cpu=seqbuf.top_p,
            top_k_cpu=seqbuf.top_k,
            min_p_cpu=seqbuf.min_p,
            page_table_cpu=page_table_cpu,
            page_table_version=page_table_version,
        )
        done += step
    seqbuf.num_computed_tokens[0] = int(prompt_len - 1)
    seqbuf.num_tokens[0] = int(prompt_len)
    seqbuf.num_tokens_no_spec[0] = int(prompt_len)

    baseline_out: list[int] = []
    t1 = time.time()
    for _ in range(int(max_new_tokens)):
        scheduled_full_cpu[0] = 1
        _ctx_unused, greedy_ids, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
            num_tokens=int(block_size),
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            input_ids_buf=input_ids_buf,
            position_ids_buf=position_ids_buf,
            padded_num_reqs=1,
            token_ids_cpu=seqbuf.token_ids,
            num_computed_tokens_cpu=seqbuf.num_computed_tokens,
            position_offset_cpu=pos_off_cpu,
            temperature_cpu=seqbuf.temperature,
            top_p_cpu=seqbuf.top_p,
            top_k_cpu=seqbuf.top_k,
            min_p_cpu=seqbuf.min_p,
            page_table_cpu=page_table_cpu,
            page_table_version=page_table_version,
        )
        base_len = int(seqbuf.num_computed_tokens[0])
        next_id = int(np.asarray(greedy_ids)[0])
        baseline_out.append(int(next_id))
        seqbuf.token_ids[0, base_len + 1] = np.int32(next_id)
        seqbuf.num_computed_tokens[0] = int(base_len + 1)
        seqbuf.num_tokens[0] = int(base_len + 2)
        seqbuf.num_tokens_no_spec[0] = int(base_len + 2)
    dt_b = max(1e-9, time.time() - t1)
    baseline_res = DFlashBenchResult(
        mode="baseline_greedy_tpu",
        prompt_len=int(prompt_len),
        max_new_tokens=int(max_new_tokens),
        block_size=1,
        accept_rate=None,
        accepted_tokens=None,
        proposed_tokens=None,
        wall_s=float(dt_b),
        output_toks_per_s=float(int(max_new_tokens)) / float(dt_b),
    )
    return DFlashDecodeOutputs(
        dflash=dflash_res,
        dflash_token_ids=dflash_out,
        baseline=baseline_res,
        baseline_token_ids=baseline_out,
    )
