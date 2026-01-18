from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _as_i32_scalar(x: Any):
    import jax.numpy as jnp

    # `DraftCtxKVCache` is used both for runtime values (ctx_len is a scalar
    # int32 JAX array) and for `in_shardings/out_shardings` pytrees during JIT
    # setup, where leaves can be `NamedSharding` / `PartitionSpec` sentinel-like
    # objects. Those must pass through unchanged.
    try:
        arr = jnp.asarray(x, dtype=jnp.int32)
    except Exception:
        return x
    if getattr(arr, "ndim", 0) != 0:
        raise ValueError(f"ctx_len must be a scalar, got shape={getattr(arr, 'shape', None)}")
    return arr


def _as_i32_0_or_1d(x: Any):
    import jax.numpy as jnp

    try:
        arr = jnp.asarray(x, dtype=jnp.int32)
    except Exception:
        return x
    if getattr(arr, "ndim", 0) not in (0, 1):
        raise ValueError(f"pos_start must be a scalar or [B], got shape={getattr(arr, 'shape', None)}")
    return arr


@dataclass
class DraftCtxKVCache:
    """Per-layer ctx KV cache for the DFlash draft model.

    IMPORTANT (TPU perf): `ctx_len` is stored as a 0-dim int32 JAX array so it
    can be passed through jitted functions without turning into a static Python
    constant (which would trigger recompiles as ctx_len grows).
    """

    k_full: list[Any]
    v_full: list[Any]
    pos_start: Any
    ctx_len: Any
    max_len: int

    def __post_init__(self) -> None:
        self.ctx_len = _as_i32_scalar(self.ctx_len)
        self.pos_start = _as_i32_0_or_1d(self.pos_start)
        self.max_len = int(self.max_len)

    # Make this a JAX pytree so we can jit draft propose/append routines without
    # treating the whole cache as a static Python object.
    def tree_flatten(self):
        children = (tuple(self.k_full), tuple(self.v_full), self.pos_start, self.ctx_len)
        aux = {"max_len": int(self.max_len)}
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        k_full, v_full, pos_start, ctx_len = children
        return cls(
            k_full=list(k_full),
            v_full=list(v_full),
            pos_start=pos_start,
            ctx_len=ctx_len,
            max_len=int(aux["max_len"]),
        )


try:
    import jax

    jax.tree_util.register_pytree_node_class(DraftCtxKVCache)
except Exception:
    pass


def materialize_draft_ctx_kv(
    *,
    draft: Any,
    rope: Any,
    ctx_hidden: Any,
    max_len: int,
    pos_start: Any = 0,
) -> DraftCtxKVCache:
    import jax.numpy as jnp

    ctx_len = int(ctx_hidden.shape[1])
    max_len = int(max_len)
    if max_len < ctx_len:
        raise ValueError(f"max_len={max_len} must be >= ctx_len={ctx_len}")

    k_list = []
    v_list = []
    for layer in draft.layers:
        k_ctx, v_ctx = layer.materialize_ctx_kv(rope=rope, ctx_hidden=ctx_hidden, start_pos=pos_start)  # [B, ctx_len, H, D]
        b, _, h, d = k_ctx.shape
        k_buf = jnp.zeros((int(b), int(max_len), int(h), int(d)), dtype=k_ctx.dtype)
        v_buf = jnp.zeros((int(b), int(max_len), int(h), int(d)), dtype=v_ctx.dtype)
        k_buf = k_buf.at[:, :ctx_len, :, :].set(k_ctx)
        v_buf = v_buf.at[:, :ctx_len, :, :].set(v_ctx)
        k_list.append(k_buf)
        v_list.append(v_buf)
    return DraftCtxKVCache(
        k_full=k_list,
        v_full=v_list,
        pos_start=_as_i32_0_or_1d(pos_start),
        ctx_len=_as_i32_scalar(ctx_len),
        max_len=int(max_len),
    )


def append_draft_ctx_kv(*, draft: Any, rope: Any, cache: DraftCtxKVCache, new_ctx_hidden: Any) -> DraftCtxKVCache:
    import jax
    import jax.numpy as jnp

    if int(new_ctx_hidden.shape[1]) <= 0:
        return cache
    n_new = int(new_ctx_hidden.shape[1])
    start = _as_i32_scalar(cache.ctx_len)
    pos_start = _as_i32_0_or_1d(cache.pos_start)
    end = start + jnp.asarray(int(n_new), dtype=jnp.int32)
    # Avoid Python-side bounds checks in the hot path; if overflow happens, the
    # verify executor will crash anyway due to max_model_len mismatch.
    k_list = []
    v_list = []
    for layer, k_old, v_old in zip(draft.layers, cache.k_full, cache.v_full):
        # Compute only the new KV, then write into the fixed buffers.
        rope_start = pos_start + start
        k_new, v_new = layer.attn._kv_full_for_ctx_hidden(  # type: ignore[attr-defined]
            rope=rope, ctx_hidden=new_ctx_hidden, start_pos=rope_start
        )  # [B, n_new, H, D]
        k_buf = jax.lax.dynamic_update_slice(jnp.asarray(k_old), jnp.asarray(k_new), (0, start, 0, 0))
        v_buf = jax.lax.dynamic_update_slice(jnp.asarray(v_old), jnp.asarray(v_new), (0, start, 0, 0))
        k_list.append(k_buf)
        v_list.append(v_buf)
    return DraftCtxKVCache(
        k_full=k_list,
        v_full=v_list,
        pos_start=pos_start,
        ctx_len=_as_i32_scalar(end),
        max_len=int(cache.max_len),
    )


def append_draft_ctx_kv_windowed(
    *,
    draft: Any,
    rope: Any,
    cache: DraftCtxKVCache,
    new_ctx_hidden: Any,
    ctx_window: int,
) -> DraftCtxKVCache:
    """Append new ctx tokens but keep a fixed-size rolling window.

    This is the TPU-safe mode for long decode:
    - We keep only the last `ctx_window` tokens in the draft KV buffers.
    - We preserve *absolute* RoPE positions by tracking `cache.pos_start`, i.e.
      token 0 in the KV buffers corresponds to absolute position `pos_start`.
    - When we drop `drop` tokens from the left, we increment `pos_start += drop`.
    """
    import jax
    import jax.numpy as jnp

    ctx_window = int(ctx_window)
    if ctx_window <= 0:
        return cache
    if int(new_ctx_hidden.shape[1]) <= 0:
        return cache

    n_new = int(new_ctx_hidden.shape[1])
    start_len = int(jnp.asarray(cache.ctx_len))
    pos_start = _as_i32_0_or_1d(cache.pos_start)

    # We maintain 0 <= ctx_len <= ctx_window.
    start_len = min(start_len, ctx_window)
    total = start_len + n_new
    drop = max(0, total - ctx_window)
    keep = total - drop
    if keep != ctx_window:
        # When total < ctx_window, we just grow the ctx_len.
        pass

    # Maintain absolute positions: after dropping, the window starts at
    # pos_start + drop. New tokens start at pos_start_new + write_pos.
    pos_start_new = pos_start + jnp.asarray(int(drop), dtype=jnp.int32)
    k_list = []
    v_list = []
    for layer, k_old, v_old in zip(draft.layers, cache.k_full, cache.v_full):
        k_buf = jnp.asarray(k_old)
        v_buf = jnp.asarray(v_old)

        if drop > 0:
            # Shift left by drop tokens: [drop:ctx_len) -> [0:ctx_len-drop)
            k_shift = jax.lax.dynamic_slice(k_buf, (0, drop, 0, 0), (k_buf.shape[0], int(ctx_window - drop), k_buf.shape[2], k_buf.shape[3]))
            v_shift = jax.lax.dynamic_slice(v_buf, (0, drop, 0, 0), (v_buf.shape[0], int(ctx_window - drop), v_buf.shape[2], v_buf.shape[3]))
            # Zero-fill the tail (not strictly required for correctness, but avoids
            # stale values influencing masked attention if segment IDs are buggy).
            k_buf = k_buf.at[:, : int(ctx_window - drop), :, :].set(k_shift)
            v_buf = v_buf.at[:, : int(ctx_window - drop), :, :].set(v_shift)

        # Write new KV at the end of the current window region.
        write_pos = int(min(ctx_window, max(0, start_len - drop)))
        rope_start = pos_start_new + jnp.asarray(int(write_pos), dtype=jnp.int32)
        k_new, v_new = layer.attn._kv_full_for_ctx_hidden(  # type: ignore[attr-defined]
            rope=rope, ctx_hidden=new_ctx_hidden, start_pos=rope_start
        )  # [B, n_new, H, D]
        k_buf = jax.lax.dynamic_update_slice(k_buf, jnp.asarray(k_new), (0, write_pos, 0, 0))
        v_buf = jax.lax.dynamic_update_slice(v_buf, jnp.asarray(v_new), (0, write_pos, 0, 0))
        k_list.append(k_buf)
        v_list.append(v_buf)

    new_len = min(ctx_window, keep)
    return DraftCtxKVCache(
        k_full=k_list,
        v_full=v_list,
        pos_start=_as_i32_0_or_1d(pos_start_new),
        ctx_len=_as_i32_scalar(jnp.asarray(new_len, dtype=jnp.int32)),
        max_len=int(cache.max_len),
    )


def append_draft_ctx_kv_windowed_committed(
    *,
    draft: Any,
    rope: Any,
    cache: DraftCtxKVCache,
    new_ctx_hidden_full: Any,
    commit_len: Any,
    ctx_window: int,
) -> DraftCtxKVCache:
    """Windowed append with fixed-shape inputs, committing only a prefix.

    TPU speed requires stable JIT shapes. In DFlash, the number of committed
    tokens per verify block varies (accept_len). If we slice the new context
    features to `keep`, JAX recompiles for many `keep` values.

    This function takes `new_ctx_hidden_full` with a fixed second dimension
    (typically `block_size`) and a scalar `commit_len` in [1, block_size]. It
    only *logically* commits the first `commit_len` tokens by:
    - advancing ctx_len/pos_start using `commit_len` (not full block_size)
    - writing K/V for the full block but zeroing the uncommitted tail so it is
      never accidentally attended.
    """
    import jax
    import jax.numpy as jnp

    ctx_window = int(ctx_window)
    if ctx_window <= 0:
        return cache

    commit_len_i32 = jnp.asarray(commit_len, dtype=jnp.int32)
    if getattr(commit_len_i32, "ndim", 0) != 0:
        commit_len_i32 = jnp.asarray(commit_len_i32.reshape(()), dtype=jnp.int32)
    # IMPORTANT: keep commit_len symbolic (no Python int()) so this function can be jitted.

    block = int(new_ctx_hidden_full.shape[1])
    if block <= 0:
        return cache

    pos_start = _as_i32_0_or_1d(cache.pos_start)
    start_len_i32 = _as_i32_scalar(cache.ctx_len)
    start_len_i32 = jnp.minimum(start_len_i32, jnp.asarray(int(ctx_window), dtype=jnp.int32))

    # Ensure the backing buffers have enough room for a full block update even when
    # committing only a prefix.
    if int(cache.max_len) < int(ctx_window + block):
        raise ValueError(f"cache.max_len={int(cache.max_len)} must be >= ctx_window+block={int(ctx_window + block)}")

    # Rolling window math uses the committed prefix length.
    total_i32 = start_len_i32 + commit_len_i32
    drop_i32 = jnp.maximum(jnp.asarray(0, dtype=jnp.int32), total_i32 - jnp.asarray(int(ctx_window), dtype=jnp.int32))
    keep_i32 = total_i32 - drop_i32
    pos_start_new = pos_start + drop_i32

    # Tail mask: zero out uncommitted KV entries.
    mask = (jnp.arange(block, dtype=jnp.int32) < commit_len_i32)[None, :, None, None]
    # Prefix mask: after shifting the ctx window by drop_i32, only the first
    # (start_len - drop_i32) tokens are valid. The dynamic_slice below always
    # reads a fixed `ctx_window` length, which can include stale data from the
    # extra "+block" tail region. Mask it out deterministically.
    old_keep_i32 = jnp.maximum(jnp.asarray(0, dtype=jnp.int32), start_len_i32 - drop_i32)
    prefix_mask = (jnp.arange(int(ctx_window), dtype=jnp.int32) < old_keep_i32)[None, :, None, None]

    k_list = []
    v_list = []
    for layer, k_old, v_old in zip(draft.layers, cache.k_full, cache.v_full):
        k_buf = jnp.asarray(k_old)
        v_buf = jnp.asarray(v_old)

        # Shift the ctx window by `drop_i32` using a fixed-size dynamic slice.
        # IMPORTANT: We must not leak stale values from the "+block" tail
        # region into the logical ctx window. Always mask the shifted prefix so
        # only the first (start_len - drop_i32) tokens remain valid before we
        # write the newly committed tokens.
        k_shift = jax.lax.dynamic_slice(
            k_buf,
            (0, drop_i32, 0, 0),
            (k_buf.shape[0], int(ctx_window), k_buf.shape[2], k_buf.shape[3]),
        ) * prefix_mask
        v_shift = jax.lax.dynamic_slice(
            v_buf,
            (0, drop_i32, 0, 0),
            (v_buf.shape[0], int(ctx_window), v_buf.shape[2], v_buf.shape[3]),
        ) * prefix_mask
        k_buf = k_buf.at[:, : int(ctx_window), :, :].set(k_shift)
        v_buf = v_buf.at[:, : int(ctx_window), :, :].set(v_shift)

        # Write new KV after the shifted prefix (start_len - drop).
        write_pos_i32 = jnp.maximum(jnp.asarray(0, dtype=jnp.int32), start_len_i32 - drop_i32)
        rope_start = pos_start_new + write_pos_i32
        k_new, v_new = layer.attn._kv_full_for_ctx_hidden(  # type: ignore[attr-defined]
            rope=rope,
            ctx_hidden=new_ctx_hidden_full,
            start_pos=rope_start,
        )  # [B, block, H, D]
        k_new = jnp.asarray(k_new) * mask
        v_new = jnp.asarray(v_new) * mask
        k_buf = jax.lax.dynamic_update_slice(k_buf, k_new, (0, write_pos_i32, 0, 0))
        v_buf = jax.lax.dynamic_update_slice(v_buf, v_new, (0, write_pos_i32, 0, 0))
        k_list.append(k_buf)
        v_list.append(v_buf)

    return DraftCtxKVCache(
        k_full=k_list,
        v_full=v_list,
        pos_start=_as_i32_0_or_1d(pos_start_new),
        ctx_len=_as_i32_scalar(jnp.minimum(jnp.asarray(int(ctx_window), dtype=jnp.int32), keep_i32)),
        max_len=int(cache.max_len),
    )


def draft_forward_with_ctx_kv(
    *,
    draft: Any,
    rope: Any,
    cache: DraftCtxKVCache,
    anchor_embedding: Any,
    mask_embedding: Any,
    block_size: int,
) -> Any:
    import jax.numpy as jnp

    b = int(anchor_embedding.shape[0])
    hidden = int(anchor_embedding.shape[-1])
    mask = jnp.broadcast_to(mask_embedding[None, None, :], (b, int(block_size - 1), hidden))
    noise_hidden = jnp.concatenate([anchor_embedding[:, None, :], mask], axis=1)
    x = noise_hidden
    for layer, k_ctx, v_ctx in zip(draft.layers, cache.k_full, cache.v_full):
        x = layer.forward_with_ctx_kv(
            rope=rope,
            ctx_k_full=k_ctx,
            ctx_v_full=v_ctx,
            ctx_len=_as_i32_scalar(cache.ctx_len),
            ctx_pos_start=_as_i32_0_or_1d(cache.pos_start),
            noise_hidden=x,
        )
    return draft.final_norm(x)
