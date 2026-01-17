from __future__ import annotations

import os
from dataclasses import dataclass
import typing as tp

from flax import nnx


@dataclass(frozen=True)
class DFlashDraftModelConfig:
    hidden_size: int
    num_layers: int
    mlp_ratio: float
    hidden_act: str
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    block_size: int
    num_context_features: int
    # Required for parity between training cache and inference-time verification.
    target_layer_ids: list[int] | None = None
    add_one_for_pre_layer_capture: bool = True
    qk_norm: bool = True
    remat: bool = True

    def get_partition_rules(self):
        # Draft is small; replicate everything by default.
        # eformer checkpointer requires an explicit catch-all rule on multi-axis meshes.
        try:
            from jax.sharding import PartitionSpec as P

            return [(r".*", P())]
        except Exception:
            return [(r".*", None)]


def _repeat_kv(x, n_rep: int):
    import jax.numpy as jnp

    if int(n_rep) == 1:
        return x
    b, s, kvh, d = x.shape
    x = x[:, :, None, :, :]
    x = jnp.broadcast_to(x, (b, s, int(n_rep), kvh, d))
    return x.reshape((b, s, kvh * int(n_rep), d))


def _split_heads(x, n_heads: int, head_dim: int):
    return x.reshape((x.shape[0], x.shape[1], int(n_heads), int(head_dim)))


def _merge_heads(x):
    return x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))


def _apply_rope_separate(*, rope, pos_q, q, pos_k, k):
    q_rot, _ = rope(pos_q, q, q)
    _, k_rot = rope(pos_k, k, k)
    return q_rot, k_rot


def _per_head_rms_norm(x, eps: float):
    import jax
    import jax.numpy as jnp

    x_f = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f), axis=-1, keepdims=True)
    return (x_f * jax.lax.rsqrt(var + float(eps))).astype(x.dtype)


class DFlashDraftModel(nnx.Module):
    """NNX DFlash draft model (no embedding / no LM head).

    Inputs:
      - context_features: bf16 [B, ctx_len, K*hidden]
      - anchor_embedding: bf16 [B, hidden]
      - rope: EasyDeL RoPE object
    Output:
      - hidden: bf16 [B, block_size, hidden]
    """

    def __init__(self, cfg: DFlashDraftModelConfig, *, rngs):
        import jax.numpy as jnp

        self.cfg = cfg
        self.param_dtype = jnp.bfloat16
        self.mesh: tp.Any | None = None
        self.config = cfg

        self.fc = nnx.Linear(
            int(cfg.num_context_features) * int(cfg.hidden_size),
            int(cfg.hidden_size),
            use_bias=True,
            rngs=rngs,
        )
        self.hidden_norm = nnx.RMSNorm(int(cfg.hidden_size), epsilon=float(cfg.rms_norm_eps), rngs=rngs)
        self.mask_embedding = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs.params(), (int(cfg.hidden_size),), dtype=jnp.float32)
        )

        self.layers = []
        for _ in range(int(cfg.num_layers)):
            self.layers.append(_DFlashBlock(cfg, rngs=rngs))
        self.final_norm = nnx.RMSNorm(int(cfg.hidden_size), epsilon=float(cfg.rms_norm_eps), rngs=rngs)

    def project_context_features(self, context_features):
        return self.hidden_norm(self.fc(context_features))

    def __call__(self, *, context_features, anchor_embedding, rope, ctx_pos_start=None):
        import jax.numpy as jnp

        c = self.cfg
        b = int(anchor_embedding.shape[0])

        if ctx_pos_start is None:
            ctx_pos_start_arr = jnp.zeros((b,), dtype=jnp.int32)
        else:
            ctx_pos_start_arr = jnp.asarray(ctx_pos_start, dtype=jnp.int32)
            if getattr(ctx_pos_start_arr, "ndim", 0) == 0:
                ctx_pos_start_arr = jnp.broadcast_to(ctx_pos_start_arr[None], (b,))
            elif getattr(ctx_pos_start_arr, "ndim", 0) == 1:
                if int(ctx_pos_start_arr.shape[0]) != b:
                    raise ValueError(
                        f"ctx_pos_start shape mismatch: got {ctx_pos_start_arr.shape}, expected ({b},)"
                    )
            else:
                raise ValueError(
                    f"ctx_pos_start must be a scalar or [B], got shape={getattr(ctx_pos_start_arr, 'shape', None)}"
                )

        ctx = self.project_context_features(context_features)
        mask = jnp.broadcast_to(
            self.mask_embedding.value.astype(ctx.dtype)[None, None, :],
            (b, int(c.block_size - 1), int(c.hidden_size)),
        )
        noise_hidden = jnp.concatenate([anchor_embedding[:, None, :].astype(ctx.dtype), mask], axis=1)

        x = noise_hidden
        for layer in self.layers:
            if bool(c.remat):
                def _call(rope_, ctx_hidden_, noise_hidden_, ctx_pos_start_):
                    return layer(
                        rope=rope_,
                        ctx_hidden=ctx_hidden_,
                        noise_hidden=noise_hidden_,
                        ctx_pos_start=ctx_pos_start_,
                    )

                x = nnx.remat(_call)(rope, ctx, x, ctx_pos_start_arr)
            else:
                x = layer(rope=rope, ctx_hidden=ctx, noise_hidden=x, ctx_pos_start=ctx_pos_start_arr)
        return self.final_norm(x)

    @property
    def dtype(self):
        return self.param_dtype

    def flops_per_token(
        self,
        sequence_length: int | None = None,
        include_loss: bool = True,
        include_backward: bool = False,
    ) -> float:
        """Return a rough FLOPs/token estimate for trainer logging.

        DFlashDraftModel is an nnx.Module (not an EasyDeLBaseModule), but the
        EasyDeL trainer stack expects `model.flops_per_token(...)` to exist.
        This estimate is used only for reporting; it does not affect training.
        """
        try:
            from easydel.infra.utils import FlopCalcConfig, TaskType, flops_per_token

            c = self.cfg
            hidden = int(c.hidden_size)
            num_heads = int(c.num_attention_heads)
            head_dim = int(c.head_dim)
            kv_heads = int(getattr(c, "num_key_value_heads", num_heads))
            # For DFlash the relevant attention window is the verify block,
            # so default to block_size when no sequence_length is provided.
            seq_len = int(sequence_length) if sequence_length is not None else int(c.block_size)
            intermediate = int(float(getattr(c, "mlp_ratio", 4.0)) * hidden)

            fconf = FlopCalcConfig(
                hidden_dim=hidden,
                intermediate_dim=intermediate,
                num_layers=int(c.num_layers),
                num_heads=num_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                seq_len=seq_len,
                # Draft has no LM head; treat as BASE_MODULE and ignore loss/head FLOPs.
                task=TaskType.BASE_MODULE,
                vocab_size=1,
                include_loss=bool(include_loss) and False,
            )
            flops = float(flops_per_token(fconf))
            if bool(include_backward):
                flops *= 3.0
            return flops
        except Exception:
            return 1.0


class _DFlashAttention(nnx.Module):
    def __init__(self, cfg: DFlashDraftModelConfig, *, rngs):
        self.cfg = cfg
        c = cfg
        self.q_proj = nnx.Linear(int(c.hidden_size), int(c.num_attention_heads) * int(c.head_dim), use_bias=True, rngs=rngs)
        self.k_proj = nnx.Linear(int(c.hidden_size), int(c.num_key_value_heads) * int(c.head_dim), use_bias=True, rngs=rngs)
        self.v_proj = nnx.Linear(int(c.hidden_size), int(c.num_key_value_heads) * int(c.head_dim), use_bias=True, rngs=rngs)
        self.o_proj = nnx.Linear(int(c.num_attention_heads) * int(c.head_dim), int(c.hidden_size), use_bias=True, rngs=rngs)

    def _kv_full_for_ctx_hidden(self, *, rope, ctx_hidden, start_pos):
        """Compute full-head KV for ctx_hidden, with RoPE positions starting at start_pos."""
        import jax.numpy as jnp

        c = self.cfg
        b = int(ctx_hidden.shape[0])
        n = int(ctx_hidden.shape[1])
        k_ctx = _split_heads(self.k_proj(ctx_hidden), int(c.num_key_value_heads), int(c.head_dim))
        v_ctx = _split_heads(self.v_proj(ctx_hidden), int(c.num_key_value_heads), int(c.head_dim))
        start_pos = jnp.asarray(start_pos, dtype=jnp.int32)
        if getattr(start_pos, "ndim", 0) == 0:
            start_pos = jnp.broadcast_to(start_pos[None], (b,))
        elif getattr(start_pos, "ndim", 0) == 1:
            if int(start_pos.shape[0]) != b:
                raise ValueError(f"start_pos shape mismatch: got {start_pos.shape}, expected ({b},)")
        else:
            raise ValueError(f"start_pos must be a scalar or [B], got shape={getattr(start_pos, 'shape', None)}")
        pos_ctx = jnp.arange(n, dtype=jnp.int32)[None, :] + start_pos[:, None]
        _, k_ctx_rope = rope(pos_ctx, k_ctx, k_ctx)
        if bool(c.qk_norm):
            k_ctx_rope = _per_head_rms_norm(k_ctx_rope, eps=float(c.rms_norm_eps))
        rep = int(c.num_attention_heads) // int(c.num_key_value_heads)
        return _repeat_kv(k_ctx_rope, rep), _repeat_kv(v_ctx, rep)

    def __call__(self, *, rope, ctx_hidden, noise_hidden, ctx_pos_start=None):
        """Non-causal attention over (ctx_hidden + noise_hidden) keys/values."""
        import jax
        import jax.numpy as jnp

        c = self.cfg
        b = int(noise_hidden.shape[0])
        ctx_len = int(ctx_hidden.shape[1])
        q_len = int(noise_hidden.shape[1])

        q = _split_heads(self.q_proj(noise_hidden), int(c.num_attention_heads), int(c.head_dim))
        k_ctx = _split_heads(self.k_proj(ctx_hidden), int(c.num_key_value_heads), int(c.head_dim))
        v_ctx = _split_heads(self.v_proj(ctx_hidden), int(c.num_key_value_heads), int(c.head_dim))
        k_noise = _split_heads(self.k_proj(noise_hidden), int(c.num_key_value_heads), int(c.head_dim))
        v_noise = _split_heads(self.v_proj(noise_hidden), int(c.num_key_value_heads), int(c.head_dim))

        # Concatenate KV along sequence.
        k = jnp.concatenate([k_ctx, k_noise], axis=1)  # [B, ctx+q, kvH, D]
        v = jnp.concatenate([v_ctx, v_noise], axis=1)

        # RoPE: keys use positions [pos_start..pos_start+ctx+q-1], queries use
        # [pos_start+ctx..pos_start+ctx+q-1]. For training-parity at long context
        # (e.g. offsets 65k/131k), ctx_pos_start can be non-zero.
        if ctx_pos_start is None:
            pos_start = jnp.zeros((b,), dtype=jnp.int32)
        else:
            pos_start = jnp.asarray(ctx_pos_start, dtype=jnp.int32)
            if getattr(pos_start, "ndim", 0) == 0:
                pos_start = jnp.broadcast_to(pos_start[None], (b,))
            elif getattr(pos_start, "ndim", 0) == 1:
                if int(pos_start.shape[0]) != b:
                    raise ValueError(f"ctx_pos_start shape mismatch: got {pos_start.shape}, expected ({b},)")
            else:
                raise ValueError(
                    f"ctx_pos_start must be a scalar or [B], got shape={getattr(pos_start, 'shape', None)}"
                )

        pos_k = jnp.arange(int(ctx_len + q_len), dtype=jnp.int32)[None, :] + pos_start[:, None]
        pos_q = (jnp.arange(int(q_len), dtype=jnp.int32) + int(ctx_len))[None, :] + pos_start[:, None]
        q_rope, k_rope = _apply_rope_separate(rope=rope, pos_q=pos_q, q=q, pos_k=pos_k, k=k)

        if bool(c.qk_norm):
            q_rope = _per_head_rms_norm(q_rope, eps=float(c.rms_norm_eps))
            k_rope = _per_head_rms_norm(k_rope, eps=float(c.rms_norm_eps))

        rep = int(c.num_attention_heads) // int(c.num_key_value_heads)
        k_full = _repeat_kv(k_rope, rep)
        v_full = _repeat_kv(v, rep)

        # Fast TPU path: pallas flash-attention (encoder-only, non-causal).
        try:
            from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention

            scale = float(int(c.head_dim) ** -0.5)
            q_t = jnp.transpose(q_rope, (0, 2, 1, 3))  # [B,H,Q,D]
            k_t = jnp.transpose(k_full, (0, 2, 1, 3))  # [B,H,K,D]
            v_t = jnp.transpose(v_full, (0, 2, 1, 3))  # [B,H,K,D]
            out_t = flash_attention(q_t, k_t, v_t, causal=False, sm_scale=scale)
            out = jnp.transpose(out_t, (0, 2, 1, 3))
            return self.o_proj(_merge_heads(out))
        except Exception:
            # Fallback: reference attention (slow, correctness-first).
            scale = float(int(c.head_dim) ** -0.5)
            attn = jnp.einsum("bqhd,bkhd->bhqk", q_rope, k_full, precision=jax.lax.Precision.HIGHEST) * scale
            attn = attn - jnp.max(attn, axis=-1, keepdims=True)
            probs = jax.nn.softmax(attn, axis=-1)
            out = jnp.einsum("bhqk,bkhd->bqhd", probs, v_full, precision=jax.lax.Precision.HIGHEST)
            return self.o_proj(_merge_heads(out))

    def materialize_ctx_kv(self, *, rope, ctx_hidden, start_pos=0):
        return self._kv_full_for_ctx_hidden(rope=rope, ctx_hidden=ctx_hidden, start_pos=start_pos)

    def append_ctx_kv(self, *, rope, ctx_k_full, ctx_v_full, new_ctx_hidden, start_pos: int):
        import jax.numpy as jnp

        c = self.cfg
        n_new = int(new_ctx_hidden.shape[1])
        if n_new <= 0:
            return ctx_k_full, ctx_v_full
        k_new_full, v_new_full = self._kv_full_for_ctx_hidden(rope=rope, ctx_hidden=new_ctx_hidden, start_pos=start_pos)
        return jnp.concatenate([ctx_k_full, k_new_full], axis=1), jnp.concatenate([ctx_v_full, v_new_full], axis=1)

    def forward_with_ctx_kv(self, *, rope, ctx_k_full, ctx_v_full, ctx_len, noise_hidden, ctx_pos_start=None):
        import jax
        import jax.numpy as jnp

        c = self.cfg
        b = int(noise_hidden.shape[0])
        ctx_len = jnp.asarray(ctx_len, dtype=jnp.int32)
        if getattr(ctx_len, "ndim", 0) != 0:
            raise ValueError(f"ctx_len must be a scalar, got shape={getattr(ctx_len, 'shape', None)}")
        max_len = int(ctx_k_full.shape[1])
        q_len = int(noise_hidden.shape[1])

        q = _split_heads(self.q_proj(noise_hidden), int(c.num_attention_heads), int(c.head_dim))
        k_noise = _split_heads(self.k_proj(noise_hidden), int(c.num_key_value_heads), int(c.head_dim))
        v_noise = _split_heads(self.v_proj(noise_hidden), int(c.num_key_value_heads), int(c.head_dim))

        # For cache-based drafting, `ctx_k_full`/`ctx_v_full` already include RoPE
        # for ctx tokens. We still need to place the non-causal draft tokens at
        # positions [pos_start+ctx_len .. pos_start+ctx_len+q_len-1]. When
        # ctx_pos_start is omitted, we default to pos_start=0 (normal decode).
        if ctx_pos_start is None:
            pos_start = jnp.zeros((b,), dtype=jnp.int32)
        else:
            pos_start = jnp.asarray(ctx_pos_start, dtype=jnp.int32)
            if getattr(pos_start, "ndim", 0) == 0:
                pos_start = jnp.broadcast_to(pos_start[None], (b,))
            elif getattr(pos_start, "ndim", 0) == 1:
                if int(pos_start.shape[0]) != b:
                    raise ValueError(f"ctx_pos_start shape mismatch: got {pos_start.shape}, expected ({b},)")
            else:
                raise ValueError(
                    f"ctx_pos_start must be a scalar or [B], got shape={getattr(pos_start, 'shape', None)}"
                )
        pos = jnp.arange(q_len, dtype=jnp.int32)[None, :] + ctx_len + pos_start[:, None]
        q_rope, _ = rope(pos, q, q)
        _, k_noise_rope = rope(pos, k_noise, k_noise)

        if bool(c.qk_norm):
            q_rope = _per_head_rms_norm(q_rope, eps=float(c.rms_norm_eps))
            k_noise_rope = _per_head_rms_norm(k_noise_rope, eps=float(c.rms_norm_eps))

        rep = int(c.num_attention_heads) // int(c.num_key_value_heads)
        k_noise_full = _repeat_kv(k_noise_rope, rep)
        v_noise_full = _repeat_kv(v_noise, rep)

        disable_fa = os.environ.get("DFLASH_CTXKV_DISABLE_FLASH_ATTENTION", "0").lower() in ("1", "true", "yes", "y", "on")
        try:
            if bool(disable_fa):
                raise RuntimeError("DFLASH_CTXKV_DISABLE_FLASH_ATTENTION=1")
            from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
            from jax.experimental.pallas.ops.tpu.flash_attention import SegmentIds

            scale = float(int(c.head_dim) ** -0.5)
            # Write noise KV into the fixed ctx buffers at [ctx_len : ctx_len + q_len]
            # using dynamic update to avoid recompiles as ctx_len grows.
            k_buf = jax.lax.dynamic_update_slice(jnp.asarray(ctx_k_full), k_noise_full, (0, ctx_len, 0, 0))
            v_buf = jax.lax.dynamic_update_slice(jnp.asarray(ctx_v_full), v_noise_full, (0, ctx_len, 0, 0))

            kv_seg = (jnp.arange(max_len, dtype=jnp.int32)[None, :] < (ctx_len + int(q_len))).astype(jnp.int32)
            # segment_ids masks out kv positions where id differs; set active=0, pad=1.
            kv_seg = jnp.where(kv_seg == 1, jnp.int32(0), jnp.int32(1))
            q_seg = jnp.zeros((int(noise_hidden.shape[0]), q_len), dtype=jnp.int32)

            q_t = jnp.transpose(q_rope, (0, 2, 1, 3))  # [B,H,Q,D]
            k_t = jnp.transpose(k_buf, (0, 2, 1, 3))  # [B,H,K,D]
            v_t = jnp.transpose(v_buf, (0, 2, 1, 3))  # [B,H,K,D]
            out_t = flash_attention(q_t, k_t, v_t, segment_ids=SegmentIds(q=q_seg, kv=kv_seg), causal=False, sm_scale=scale)
            out = jnp.transpose(out_t, (0, 2, 1, 3))
            return self.o_proj(_merge_heads(out))
        except Exception:
            scale = float(int(c.head_dim) ** -0.5)
            # Fallback path must remain fully JIT-compatible on TPU.
            # Avoid Python-side slicing with dynamic `ctx_len` (causes
            # ConcretizationTypeError). Instead, write noise KV into the fixed
            # buffers and mask out the unused tail tokens.
            k_buf = jax.lax.dynamic_update_slice(jnp.asarray(ctx_k_full), k_noise_full, (0, ctx_len, 0, 0))
            v_buf = jax.lax.dynamic_update_slice(jnp.asarray(ctx_v_full), v_noise_full, (0, ctx_len, 0, 0))

            attn = jnp.einsum("bqhd,bkhd->bhqk", q_rope, k_buf, precision=jax.lax.Precision.HIGHEST) * scale
            # Mask invalid KV positions: only [0 : ctx_len + q_len) are valid.
            kv_valid = (jnp.arange(max_len, dtype=jnp.int32)[None, None, None, :] < (ctx_len + int(q_len)))
            attn = jnp.where(kv_valid, attn, jnp.asarray(-1e9, dtype=attn.dtype))
            attn = attn - jnp.max(attn, axis=-1, keepdims=True)
            probs = jax.nn.softmax(attn, axis=-1)
            out = jnp.einsum("bhqk,bkhd->bqhd", probs, v_buf, precision=jax.lax.Precision.HIGHEST)
            return self.o_proj(_merge_heads(out))


class _DFlashMLP(nnx.Module):
    def __init__(self, cfg: DFlashDraftModelConfig, *, rngs):
        self.cfg = cfg
        c = cfg
        inter = int(round(float(c.hidden_size) * float(c.mlp_ratio)))
        self.gate_proj = nnx.Linear(int(c.hidden_size), inter, use_bias=True, rngs=rngs)
        self.up_proj = nnx.Linear(int(c.hidden_size), inter, use_bias=True, rngs=rngs)
        self.down_proj = nnx.Linear(inter, int(c.hidden_size), use_bias=True, rngs=rngs)

    def __call__(self, x):
        import jax

        act = str(self.cfg.hidden_act).lower()
        if act not in ("silu", "swish"):
            raise ValueError(f"Unsupported hidden_act: {self.cfg.hidden_act!r}")
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _DFlashBlock(nnx.Module):
    def __init__(self, cfg: DFlashDraftModelConfig, *, rngs):
        self.cfg = cfg
        self.in_norm = nnx.RMSNorm(int(cfg.hidden_size), epsilon=float(cfg.rms_norm_eps), rngs=rngs)
        self.attn = _DFlashAttention(cfg, rngs=rngs)
        self.post_norm = nnx.RMSNorm(int(cfg.hidden_size), epsilon=float(cfg.rms_norm_eps), rngs=rngs)
        self.mlp = _DFlashMLP(cfg, rngs=rngs)

    def forward_with_ctx_kv(self, *, rope, ctx_k_full, ctx_v_full, ctx_len, noise_hidden, ctx_pos_start=None):
        x = noise_hidden
        x = x + self.attn.forward_with_ctx_kv(
            rope=rope,
            ctx_k_full=ctx_k_full,
            ctx_v_full=ctx_v_full,
            ctx_len=ctx_len,
            noise_hidden=self.in_norm(x),
            ctx_pos_start=ctx_pos_start,
        )
        x = x + self.mlp(self.post_norm(x))
        return x

    def __call__(self, *, rope, ctx_hidden, noise_hidden, ctx_pos_start=None):
        x = noise_hidden
        x = x + self.attn(
            rope=rope,
            ctx_hidden=ctx_hidden,
            noise_hidden=self.in_norm(x),
            ctx_pos_start=ctx_pos_start,
        )
        x = x + self.mlp(self.post_norm(x))
        return x

    def materialize_ctx_kv(self, *, rope, ctx_hidden, start_pos=0):
        return self.attn.materialize_ctx_kv(rope=rope, ctx_hidden=ctx_hidden, start_pos=start_pos)

    def append_ctx_kv(self, *, rope, ctx_k_full, ctx_v_full, new_ctx_hidden, start_pos: int):
        return self.attn.append_ctx_kv(
            rope=rope,
            ctx_k_full=ctx_k_full,
            ctx_v_full=ctx_v_full,
            new_ctx_hidden=new_ctx_hidden,
            start_pos=start_pos,
        )
