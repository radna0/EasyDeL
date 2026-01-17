from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple


def dflash_accept_len_and_bonus(*, candidates, target_predict) -> Tuple["jax.Array", "jax.Array"]:
    """Compute DFlash accept length + bonus token (SGLang semantics).

    candidates: int32/int64 [bs, B]      (token0=anchor, token1..=draft tokens)
    target_predict: int32/int64 [bs, B]  (argmax target logits at each position)

    Rule:
      accept while candidates[:, 1:] == target_predict[:, :-1] consecutively.
      accept_len excludes current token (index 0).
      bonus token is target_predict[:, accept_len].
    """
    import jax.numpy as jnp

    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got {candidates.shape}")
    if target_predict.shape != candidates.shape:
        raise ValueError(f"target_predict shape mismatch: {target_predict.shape} vs {candidates.shape}")

    matches = candidates[:, 1:] == target_predict[:, :-1]
    accept_len = jnp.sum(jnp.cumprod(matches.astype(jnp.int32), axis=1), axis=1).astype(jnp.int32)
    bonus = jnp.take_along_axis(target_predict, accept_len[:, None], axis=1)[:, 0].astype(jnp.int32)
    return accept_len, bonus


def extract_dflash_context_features_from_hidden_states(
    *,
    hidden_states: Iterable["jax.Array"],
    target_layer_ids: list[int],
    add_one_for_pre_layer_capture: bool = True,
) -> "jax.Array":
    """Concatenate per-layer hidden states into context features [B,S,K*H]."""
    import jax.numpy as jnp

    hs_list = list(hidden_states)
    if not hs_list:
        raise ValueError("hidden_states is empty")
    if not target_layer_ids:
        raise ValueError("target_layer_ids must be non-empty")

    idxs = []
    for x in target_layer_ids:
        i = int(x)
        if add_one_for_pre_layer_capture:
            i += 1
        idxs.append(i)
    if max(idxs) >= len(hs_list):
        raise ValueError(f"requested hidden index {max(idxs)} but only have {len(hs_list)} hidden_states")
    picked = [jnp.asarray(hs_list[i]) for i in idxs]
    return jnp.concatenate(picked, axis=-1)


@dataclass(frozen=True)
class DFlashMetrics:
    accept_len_mean: float
    accept_rate: float


def summarize_accept(accept_len, *, block_size: int) -> DFlashMetrics:
    import jax.numpy as jnp

    m = jnp.mean(accept_len.astype(jnp.float32))
    return DFlashMetrics(
        accept_len_mean=float(m),
        accept_rate=float(m / max(int(block_size), 1)),
    )

