from __future__ import annotations

import numpy as np


def _ref_accept_len_and_bonus(candidates: np.ndarray, target_predict: np.ndarray):
    matches = (candidates[:, 1:] == target_predict[:, :-1]).astype(np.int32)
    accept_len = np.cumprod(matches, axis=1).sum(axis=1).astype(np.int32)
    bonus = target_predict[np.arange(candidates.shape[0]), accept_len].astype(np.int32)
    return accept_len, bonus


def test_dflash_accept_len_and_bonus_parity():
    try:
        import jax.numpy as jnp
    except Exception:
        # CPU test env may not ship with JAX; skip gracefully.
        return

    from easydel.inference import dflash_accept_len_and_bonus

    rng = np.random.default_rng(0)
    for bs in [1, 2, 8]:
        for b in [2, 8, 16]:
            cand = rng.integers(0, 100, size=(bs, b), dtype=np.int32)
            pred = rng.integers(0, 100, size=(bs, b), dtype=np.int32)
            # Force a deterministic matching prefix.
            if b >= 5:
                cand[:, 1] = pred[:, 0]
                cand[:, 2] = pred[:, 1]
                cand[:, 3] = pred[:, 2]
            ref_a, ref_bonus = _ref_accept_len_and_bonus(cand, pred)
            a, bonus = dflash_accept_len_and_bonus(candidates=jnp.asarray(cand), target_predict=jnp.asarray(pred))
            assert np.array_equal(np.asarray(a), ref_a)
            assert np.array_equal(np.asarray(bonus), ref_bonus)

