from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


def _maybe_mesh_ctx(model: Any):
    mesh = getattr(model, "mesh", None)
    if mesh is None:
        from contextlib import nullcontext

        return nullcontext()
    return mesh


def _cache_seq_len(cache: Any) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    if hasattr(cache, "get_seq_len"):
        return int(cache.get_seq_len())
    raise RuntimeError("Unsupported cache type (missing get_seq_length/get_seq_len)")


@dataclass(frozen=True)
class DFlashDecodeResult:
    mode: str
    prompt_len: int
    max_new_tokens: int
    block_size: int
    accepted_tokens: int
    proposed_tokens: int
    accept_rate: float
    wall_s: float
    tok_s_total: float
    baseline_wall_s: float | None = None
    baseline_tok_s_total: float | None = None
    speedup_x: float | None = None


def dflash_cached_decode_blockverify(
    *,
    teacher: Any,
    draft: Any,
    rope: Any,
    prompt_ids,
    max_new_tokens: int,
    block_size: int,
    target_layer_ids: list[int],
    add_one_for_pre_layer_capture: bool,
    also_run_baseline: bool = False,
) -> DFlashDecodeResult:
    """DFlash decode with cached teacher + block-parallel verify (fast path).

    Requirements:
    - teacher(...) supports KV-cache via past_key_values, and returns past_key_values.
    - cache supports crop(new_len) OR supports safe tree-cropping by the caller.
      For now we require a native `.crop` (the fast path).
    """
    import jax
    import jax.numpy as jnp

    from .dflash import dflash_accept_len_and_bonus, extract_dflash_context_features_from_hidden_states
    from .dflash_kv_cache import append_draft_ctx_kv, draft_forward_with_ctx_kv, materialize_draft_ctx_kv

    if int(block_size) <= 1:
        raise ValueError("block_size must be > 1")
    if int(prompt_ids.shape[0]) < 2:
        raise ValueError("prompt_ids must have at least 2 tokens (prefix + current)")
    if not target_layer_ids:
        raise ValueError("target_layer_ids must be non-empty")

    # Token split: prefill on prefix (excluding current token) like SGLang.
    prefix_ids = prompt_ids[:-1]
    cur_id = prompt_ids[-1:]
    prefix = jnp.asarray(prefix_ids[None, :], dtype=jnp.int32)
    cur = jnp.asarray(cur_id[None, :], dtype=jnp.int32)  # [1,1]

    emb_fn = teacher.get_embedding()
    lm_head = teacher.get_lm_head()
    kernel = jax.lax.stop_gradient(lm_head.kernel.value)
    bias = jax.lax.stop_gradient(lm_head.bias.value) if getattr(lm_head, "bias", None) is not None else None

    with _maybe_mesh_ctx(teacher):
        out_prefill = teacher(
            input_ids=prefix,
            output_hidden_states=True,
            apply_lm_head=False,
            use_cache=True,
        )
    cache = out_prefill.past_key_values
    if cache is None:
        raise RuntimeError("Teacher did not return past_key_values.")
    if not hasattr(cache, "crop"):
        raise RuntimeError("Teacher cache missing crop(); block-verify requires crop/rollback support.")

    hs_prefill = out_prefill.hidden_states
    if hs_prefill is None:
        raise RuntimeError("Teacher did not return hidden_states on prefill.")

    ctx_feat = extract_dflash_context_features_from_hidden_states(
        hidden_states=hs_prefill,
        target_layer_ids=target_layer_ids,
        add_one_for_pre_layer_capture=bool(add_one_for_pre_layer_capture),
    )
    ctx_hidden = draft.project_context_features(ctx_feat)
    ctx_kv = materialize_draft_ctx_kv(draft=draft, rope=rope, ctx_hidden=ctx_hidden, max_len=int(max_model_len + int(block_size)))

    accepted = 0
    proposed = 0

    t0 = time.time()
    for _ in range(int(max_new_tokens)):
        with _maybe_mesh_ctx(teacher):
            anchor_emb = emb_fn(cur.astype("i4"))[:, 0, :]

        d_hidden = draft_forward_with_ctx_kv(
            draft=draft,
            rope=rope,
            cache=ctx_kv,
            anchor_embedding=anchor_emb.astype(jnp.bfloat16),
            mask_embedding=draft.mask_embedding.value.astype(jnp.bfloat16),
            block_size=int(block_size),
        )
        hs_d = d_hidden[:, 1:, :]
        d_logits = jnp.einsum("bsh,hv->bsv", hs_d.astype(jnp.bfloat16), kernel.astype(jnp.bfloat16))
        if bias is not None:
            d_logits = d_logits + bias[None, None, :]
        draft_tokens = jnp.argmax(d_logits, axis=-1).astype(jnp.int32)  # [1,B-1]

        cand = jnp.concatenate([cur, draft_tokens], axis=1)  # [1,B]
        base_len = _cache_seq_len(cache)

        with _maybe_mesh_ctx(teacher):
            out_v = teacher(
                input_ids=cand,
                past_key_values=cache,
                output_hidden_states=True,
                apply_lm_head=True,
                use_cache=True,
            )
        cache_full = out_v.past_key_values
        if cache_full is None:
            raise RuntimeError("Teacher verify forward missing past_key_values")

        target_predict = jnp.argmax(out_v.logits, axis=-1).astype(jnp.int32)  # [1,B]
        accept_len, bonus = dflash_accept_len_and_bonus(candidates=cand, target_predict=target_predict)
        n_acc = int(accept_len[0])
        keep_in_block = 1 + n_acc

        cache_full.crop(int(base_len + keep_in_block))
        cache = cache_full
        cur = bonus.astype(jnp.int32)[:, None]

        hs_v = out_v.hidden_states
        if hs_v is None:
            raise RuntimeError("Teacher verify forward missing hidden_states")
        seq_dim = int(hs_v[0].shape[1])
        start = 0 if seq_dim == int(block_size) else seq_dim - int(block_size)
        # Slice the hidden states down to committed tokens only.
        hs_commit = tuple(x[:, start : start + keep_in_block, :] for x in hs_v)
        feat_commit = extract_dflash_context_features_from_hidden_states(
            hidden_states=hs_commit,
            target_layer_ids=target_layer_ids,
            add_one_for_pre_layer_capture=bool(add_one_for_pre_layer_capture),
        )
        new_ctx_hidden = draft.project_context_features(feat_commit)
        ctx_kv = append_draft_ctx_kv(draft=draft, rope=rope, cache=ctx_kv, new_ctx_hidden=new_ctx_hidden)

        accepted += n_acc
        proposed += int(block_size) - 1

    dt = max(1e-9, time.time() - t0)
    tok_s = float(int(max_new_tokens)) / dt
    out = DFlashDecodeResult(
        mode="cached_block_verify",
        prompt_len=int(prompt_ids.shape[0]),
        max_new_tokens=int(max_new_tokens),
        block_size=int(block_size),
        accepted_tokens=int(accepted),
        proposed_tokens=int(proposed),
        accept_rate=float(accepted) / float(max(1, proposed)),
        wall_s=float(dt),
        tok_s_total=float(tok_s),
    )

    if not bool(also_run_baseline):
        return out

    # Baseline greedy cached decode for speedup comparison.
    cache_b = cache
    cur_b = cur
    t1 = time.time()
    for _ in range(int(max_new_tokens)):
        with _maybe_mesh_ctx(teacher):
            out_b = teacher(
                input_ids=cur_b,
                past_key_values=cache_b,
                output_hidden_states=False,
                apply_lm_head=True,
                use_cache=True,
            )
        cache_b = out_b.past_key_values
        cur_b = jnp.argmax(out_b.logits[:, -1, :], axis=-1).astype(jnp.int32)[:, None]
    dt_b = max(1e-9, time.time() - t1)
    tok_s_b = float(int(max_new_tokens)) / dt_b
    return DFlashDecodeResult(
        **{
            **out.__dict__,
            "baseline_wall_s": float(dt_b),
            "baseline_tok_s_total": float(tok_s_b),
            "speedup_x": float(tok_s) / float(max(tok_s_b, 1e-9)),
        }
    )
