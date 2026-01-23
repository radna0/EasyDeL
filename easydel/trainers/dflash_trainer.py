from __future__ import annotations

import json
import os
import typing as tp
from dataclasses import dataclass
from pathlib import Path
import shutil
from queue import Queue
from threading import Thread

# Keep caches fast by default on TPU boxes; callers can override.
os.environ.setdefault("HF_HOME", "/dev/shm/hf")
os.environ.setdefault("HF_HUB_CACHE", "/dev/shm/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/dev/shm/hf/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/dev/shm/xdg")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/dev/shm/jax_compilation_cache_dflash")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")
os.environ.setdefault("TMPDIR", "/dev/shm/tmp")

import jax
import numpy as np
from eformer.loggings import get_logger
from flax import nnx
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossMetrics
from easydel.trainers.trainer import Trainer
from easydel.trainers.base_trainer import DEFAULT_ARGS_JSON_NAME
from easydel.trainers.trainer_protocol import (
    TrainerConfigureDataloaderOutput,
    TrainerConfigureFunctionOutput,
    TrainerConfigureModelOutput,
)
from easydel.utils import Registry

from easydel.inference.speculative.dflash_draft_model import DFlashDraftModel, DFlashDraftModelConfig
from .dflash_cache import DFlashTeacherCacheDataset
from .dflash_config import DFlashConfig

logger = get_logger(__name__)

def _env_flag(name: str) -> bool:
    v = os.environ.get(name, "")
    return v.lower() in ("1", "true", "yes", "y", "on")


class _LocalStepCheckpointer:
    """Minimal checkpoint scheduler for multi-host pmap (replicated params).

    We avoid JAX's tensorstore multiprocess serialization entirely and instead
    call back into `Trainer._save_state(...)` (which DFlashTrainer overrides) on
    every host. Each host writes to its *local* filesystem path, so there is no
    cross-host IO coordination requirement.
    """

    def __init__(self, *, save_steps: int):
        self.save_steps = int(save_steps or 0)

    def on_step(
        self,
        *,
        mesh=None,
        pytree=None,
        step: int,
        force: bool = False,
        true_callbacks: list | None = None,
        false_callbacks: list | None = None,
        **_kwargs,
    ) -> None:
        should_save = bool(force) or (self.save_steps > 0 and int(step) > 0 and (int(step) % self.save_steps) == 0)
        callbacks = true_callbacks if should_save else false_callbacks
        if not callbacks:
            return
        dest = f"run-{int(step)}"
        meta = {"step": int(step), "forced": bool(force)}
        for cb in callbacks:
            cb(dest, mesh, meta)


def _set_shm_caches() -> None:
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    # TPU XLA flag availability varies by runtime/libtpu version. A bad flag
    # hard-crashes the process at startup, so keep this opt-in.
    if os.environ.get("EASYDEL_ENABLE_XLA_LATENCY_HIDING", "0").lower() in ("1", "true", "yes", "y", "on"):
        xla_flags = os.environ.get("XLA_FLAGS", "")
        if "--xla_tpu_enable_latency_hiding_scheduler" not in xla_flags:
            os.environ["XLA_FLAGS"] = (xla_flags + " --xla_tpu_enable_latency_hiding_scheduler=true").strip()


def _require_token_present(*, teacher_snapshot_dir: str | None = None, teacher_easydel_dir: str | None = None) -> None:
    """Ensure an HF token is available when we might need to hit the Hub.

    Cache-first DFlash training can run entirely from local snapshot directories
    (only reading `config.json` and `lm_head.weight` from safetensors). In that
    case, requiring a token is unnecessary and blocks offline/airgapped runs.
    """
    if os.environ.get("ALLOW_MISSING_HF_TOKEN", "0").lower() in ("1", "true", "yes", "y", "on"):
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return

    if teacher_snapshot_dir:
        try:
            snap = Path(teacher_snapshot_dir).expanduser().resolve()
            if snap.exists() and (snap / "config.json").is_file():
                return
        except Exception:
            pass

    if teacher_easydel_dir:
        try:
            ckpt = Path(teacher_easydel_dir).expanduser().resolve()
            if ckpt.exists() and (ckpt / "config.json").is_file():
                return
        except Exception:
            pass

    # Fall back to a cached local token (common on TPU boxes where users have
    # previously run `huggingface-cli login`).
    try:
        from huggingface_hub import HfFolder

        cached = HfFolder.get_token()
        if cached:
            os.environ.setdefault("HF_TOKEN", cached)
            return
    except Exception:
        pass

    raise RuntimeError("Missing HF token in env (HF_TOKEN or HUGGINGFACE_HUB_TOKEN).")


def _load_lm_head_weight(snapshot_dir: Path) -> jax.Array:
    """Load `lm_head.weight` as a JAX array [V,H] from safetensors."""
    from safetensors import safe_open

    name_candidates = ("lm_head.weight", "model.lm_head.weight")

    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.exists():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = idx.get("weight_map", {})
        for name in name_candidates:
            shard = weight_map.get(name)
            if shard is None:
                continue
            with safe_open(str(snapshot_dir / shard), framework="flax") as f:
                return f.get_tensor(name)
        raise KeyError(f"Missing {name_candidates} in {index_path.name}")

    single_path = snapshot_dir / "model.safetensors"
    if not single_path.exists():
        raise FileNotFoundError(f"Missing {index_path.name} and {single_path.name} in {snapshot_dir}")
    with safe_open(str(single_path), framework="flax") as f:
        for name in name_candidates:
            if name in f.keys():
                return f.get_tensor(name)
    raise KeyError(f"Missing {name_candidates} in {single_path.name}")


def _load_lm_head_weight_from_easydel(teacher_dir: Path) -> jax.Array:
    """Load lm_head kernel as a JAX array [V,H] from an EasyDeL zarr checkpoint directory."""
    import tensorstore as ts

    candidates = (
        teacher_dir / "model" / "lm_head" / "kernel",
        teacher_dir / "model" / "params" / "lm_head" / "kernel",
    )
    kernel_dir = next((p for p in candidates if (p / ".zarray").exists()), None)
    if kernel_dir is None:
        raise FileNotFoundError(
            f"Could not locate EasyDeL lm_head kernel under {teacher_dir} "
            f"(tried {', '.join(str(p) for p in candidates)})"
        )

    cfg_path = teacher_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    vocab_size = int(cfg["vocab_size"])
    hidden_size = int(cfg["hidden_size"])

    spec = {"driver": "zarr", "kvstore": {"driver": "file", "path": str(kernel_dir)}}
    arr = ts.open(spec, open=True).result()
    np_w = arr.read().result()
    w = jnp.asarray(np_w)
    if tuple(w.shape) == (vocab_size, hidden_size):
        return w
    if tuple(w.shape) == (hidden_size, vocab_size):
        return jnp.swapaxes(w, 0, 1)
    raise ValueError(
        "Unexpected lm_head kernel shape from EasyDeL zarr checkpoint: "
        f"got={tuple(w.shape)} expected={(vocab_size, hidden_size)} or {(hidden_size, vocab_size)} "
        f"(teacher_dir={teacher_dir})"
    )


def _build_rope(*, cfg: dict, dtype):
    from easydel.layers.rotary_embedding import get_rope

    return get_rope(
        head_size=int(cfg["head_dim"]),
        rotary_dim=int(cfg["head_dim"]),
        max_position=int(cfg["max_position_embeddings"]),
        base=int(cfg["rope_theta"]),
        is_neox_style=True,
        rope_scaling=cfg.get("rope_scaling"),
        dtype=dtype,
    )


def _bf16_from_u16(x_u16: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(x_u16.astype(jnp.uint16), jnp.bfloat16)


@dataclass(frozen=True)
class _DFlashBatch:
    context_u16: jax.Array  # [B, ctx, K*H] uint16
    anchor_u16: jax.Array  # [B, H] uint16
    target_ids: jax.Array  # [B, block-1] int32
    ctx_pos_start: jax.Array  # [B] int32 (RoPE start position for ctx token 0)


def _batch_from_dict(batch: dict) -> _DFlashBatch:
    anchor_u16 = jnp.asarray(batch["anchor_embedding_u16"], dtype=jnp.uint16)
    bsz = int(anchor_u16.shape[0])
    ctx_pos = batch.get("ctx_pos_start_i32", None)
    if ctx_pos is None:
        ctx_pos_arr = jnp.zeros((bsz,), dtype=jnp.int32)
    else:
        ctx_pos_arr = jnp.asarray(ctx_pos, dtype=jnp.int32).reshape((bsz,))
    return _DFlashBatch(
        context_u16=jnp.asarray(batch["context_features_u16"], dtype=jnp.uint16),
        anchor_u16=anchor_u16,
        target_ids=jnp.asarray(batch["target_ids"], dtype=jnp.int32),
        ctx_pos_start=ctx_pos_arr,
    )


def _choose_vocab_chunk(*, vocab_size: int, requested: int) -> int:
    if requested <= 0:
        return 0
    if vocab_size % requested == 0:
        return requested
    for d in range(requested, 0, -1):
        if vocab_size % d == 0:
            return d
    return 0


def _parse_run_step(run_dir: Path) -> int | None:
    if not run_dir.is_dir():
        return None
    name = run_dir.name
    if not name.startswith("run-"):
        return None
    try:
        return int(name.split("-", 1)[1])
    except Exception:
        return None


def _is_complete_run(run_dir: Path) -> bool:
    if not (run_dir / "metadata.json").exists():
        return False
    if not (run_dir / "model").is_dir():
        return False
    # DFlash resumes by restoring model weights and re-initializing optimizer
    # state (see `_maybe_resume_state`). Optimizer state may be intentionally
    # skipped to reduce checkpoint IO, so `tx/` is optional.
    return True


def _prune_old_run_dirs(run_root: Path, *, keep: int = 2) -> None:
    """Keep at most `keep` newest run-* dirs under run_root (best-effort)."""
    run_root = Path(str(run_root))
    if keep <= 0 or not run_root.is_dir():
        return
    runs: list[tuple[int, Path]] = []
    for child in run_root.iterdir():
        step = _parse_run_step(child)
        if step is None:
            continue
        if not _is_complete_run(child):
            continue
        runs.append((int(step), child))
    runs.sort(key=lambda x: x[0], reverse=True)
    for _step, path in runs[int(keep) :]:
        try:
            shutil.rmtree(str(path))
        except Exception:
            pass


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(str(tmp), str(path))


def _maybe_gunzip(data: bytes) -> bytes:
    if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
        import gzip

        return gzip.decompress(data)
    return data


def _free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(Path(str(path))))
    return float(usage.free) / float(1024**3)


def _ensure_free_space(run_root: Path, *, min_free_gb: float, keep_runs: int) -> None:
    """Best-effort space guardrail for TPU VM root disks (never raises)."""
    run_root = Path(str(run_root))
    try:
        _prune_old_run_dirs(run_root, keep=int(keep_runs))
    except Exception:
        pass
    try:
        if _free_gb(run_root) >= float(min_free_gb):
            return
    except Exception:
        return
    # Second-chance cleanup: these can balloon quickly on TPU VMs.
    for extra in (Path.home() / "tmp", Path.home() / ".cache", Path.home() / "harmony_logs"):
        try:
            if extra.exists():
                shutil.rmtree(extra, ignore_errors=True)
        except Exception:
            pass
    try:
        _prune_old_run_dirs(run_root, keep=int(keep_runs))
    except Exception:
        pass


def _find_latest_complete_run(run_root: Path) -> tuple[Path, int] | None:
    if not run_root.is_dir():
        return None
    best: tuple[Path, int] | None = None
    for child in run_root.iterdir():
        step = _parse_run_step(child)
        if step is None:
            continue
        if not _is_complete_run(child):
            continue
        if best is None or step > best[1]:
            best = (child, step)
    return best


def _chunked_ce_nll_and_acc(
    *,
    hs: jax.Array,
    labels: jax.Array,
    lm_w: jax.Array,
    vocab_chunk: int,
) -> tuple[jax.Array, jax.Array]:
    """Compute mean CE NLL and top-1 accuracy over vocab, optionally chunked.

    hs: [B, S, H] (bfloat16 preferred)
    labels: [B, S] int32
    lm_w: [V, H] (frozen)

    When chunked, uses a numerically-stable streaming logsumexp; this must rescale
    the running sum when the running max increases.
    """
    import optax

    hs_f = hs.astype(jnp.bfloat16)
    labels_i32 = labels.astype(jnp.int32)
    vocab_size = int(lm_w.shape[0])
    seq_len = int(hs_f.shape[1])

    if int(vocab_chunk) <= 0:
        logits = jnp.einsum("bsh,vh->bsv", hs_f, lm_w, precision=jax.lax.Precision.HIGHEST)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels_i32).mean()
        acc = jnp.mean((jnp.argmax(logits, axis=-1).astype(jnp.int32) == labels_i32).astype(jnp.float32))
        return loss, acc

    bsz = int(hs_f.shape[0])
    running_max = jnp.full((bsz, seq_len), -jnp.inf, dtype=jnp.float32)
    running_sumexp = jnp.zeros((bsz, seq_len), dtype=jnp.float32)
    gold_logits = jnp.full((bsz, seq_len), -jnp.inf, dtype=jnp.float32)

    best_val = jnp.full((bsz, seq_len), -jnp.inf, dtype=jnp.float32)
    best_ids = jnp.zeros((bsz, seq_len), dtype=jnp.int32)

    chunk = int(vocab_chunk)
    # IMPORTANT: avoid a Python `for` loop here. If we unroll 16+ chunks at
    # compile-time, XLA compilation can become huge/unstable on TPU and the
    # process may get killed without a Python traceback. Use `lax.scan` to keep
    # the HLO small and compilation predictable.
    if vocab_size % chunk != 0:
        raise ValueError(f"vocab_size={vocab_size} must be divisible by vocab_chunk={chunk}")
    n_chunks = vocab_size // chunk
    lm_w_chunks = lm_w.reshape((n_chunks, chunk, int(lm_w.shape[1])))
    starts = (jnp.arange(n_chunks, dtype=jnp.int32) * jnp.int32(chunk)).reshape((n_chunks,))

    def _scan_body(carry, xs):
        running_max, running_sumexp, gold_logits, best_val, best_ids = carry
        w, start = xs  # w: [chunk,H], start: scalar int32
        end = start + jnp.int32(chunk)

        chunk_logits = jnp.einsum(
            "bsh,vh->bsv",
            hs_f,
            w,
            precision=jax.lax.Precision.HIGHEST,
        ).astype(jnp.float32)

        chunk_max = jnp.max(chunk_logits, axis=-1)  # [B,S]
        new_max = jnp.maximum(running_max, chunk_max)
        running_sumexp = running_sumexp * jnp.exp(running_max - new_max)
        running_sumexp = running_sumexp + jnp.sum(jnp.exp(chunk_logits - new_max[..., None]), axis=-1)
        running_max = new_max

        in_chunk = (labels_i32 >= start) & (labels_i32 < end)
        idx = jnp.clip(labels_i32 - start, 0, chunk - 1).astype(jnp.int32)
        picked = jnp.take_along_axis(chunk_logits, idx[..., None], axis=-1)[..., 0]
        gold_logits = jnp.where(in_chunk, picked, gold_logits)

        chunk_best_local = jnp.argmax(chunk_logits, axis=-1).astype(jnp.int32)
        chunk_best_val = jnp.take_along_axis(chunk_logits, chunk_best_local[..., None], axis=-1)[..., 0]
        take_chunk = chunk_best_val > best_val
        best_val = jnp.where(take_chunk, chunk_best_val, best_val)
        best_ids = jnp.where(take_chunk, chunk_best_local + start, best_ids)

        return (running_max, running_sumexp, gold_logits, best_val, best_ids), None

    (running_max, running_sumexp, gold_logits, best_val, best_ids), _ = jax.lax.scan(
        _scan_body,
        (running_max, running_sumexp, gold_logits, best_val, best_ids),
        (lm_w_chunks, starts),
    )

    logz = running_max + jnp.log(running_sumexp + 1e-9)
    nll = logz - gold_logits
    loss = jnp.mean(nll)
    acc = jnp.mean((best_ids == labels_i32).astype(jnp.float32))
    return loss, acc


@Registry.register("trainer", "dflash")
class DFlashTrainer(Trainer):
    """Cache-first DFlash draft training (TPU-first).

    - Training never runs the teacher forward; it uses cached teacher features.
    - dp-only pmap path is the stable default on single-host TPU (8 devices).
    """

    arguments: DFlashConfig

    def __init__(
        self,
        arguments: DFlashConfig,
        *,
        processing_class,
        train_dataset: tp.Any | None = None,
        eval_dataset: tp.Any | None = None,
    ):
        if not isinstance(arguments, DFlashConfig):
            raise TypeError("arguments must be a DFlashConfig")

        _set_shm_caches()
        _require_token_present(
            teacher_snapshot_dir=arguments.teacher_snapshot_dir,
            teacher_easydel_dir=getattr(arguments, "teacher_easydel_dir", None),
        )

        if not arguments.cache_dir:
            raise ValueError("DFlashConfig.cache_dir is required")
        teacher_snapshot_dir = arguments.teacher_snapshot_dir
        teacher_easydel_dir = getattr(arguments, "teacher_easydel_dir", None)
        if bool(teacher_snapshot_dir) == bool(teacher_easydel_dir):
            raise ValueError(
                "Provide exactly one of DFlashConfig.teacher_snapshot_dir or DFlashConfig.teacher_easydel_dir"
            )

        self.arguments = arguments
        self.cache = DFlashTeacherCacheDataset(arguments.cache_dir)
        self.teacher_snapshot = Path(teacher_snapshot_dir).resolve() if teacher_snapshot_dir else None
        self.teacher_easydel_dir = Path(teacher_easydel_dir).resolve() if teacher_easydel_dir else None

        meta = self.cache.meta
        if meta.dtype not in ("bf16_u16",):
            raise ValueError(f"Unsupported cache dtype {meta.dtype!r}; expected bf16_u16")

        teacher_root = self.teacher_snapshot if self.teacher_snapshot is not None else self.teacher_easydel_dir
        if teacher_root is None:
            raise RuntimeError("Internal error: missing teacher root")
        teacher_cfg = json.loads((teacher_root / "config.json").read_text(encoding="utf-8"))
        self._rope = _build_rope(cfg=teacher_cfg, dtype=jnp.bfloat16)
        if self.teacher_snapshot is not None:
            lm_w = _load_lm_head_weight(self.teacher_snapshot)
        else:
            lm_w = _load_lm_head_weight_from_easydel(self.teacher_easydel_dir)  # type: ignore[arg-type]
        try:
            expected_vocab = int(teacher_cfg["vocab_size"])
            expected_hidden = int(teacher_cfg["hidden_size"])
            if tuple(lm_w.shape) != (expected_vocab, expected_hidden):
                raise ValueError(
                    f"lm_head.weight shape mismatch: got={tuple(lm_w.shape)} expected={(expected_vocab, expected_hidden)}"
                )
        except Exception as e:
            raise ValueError(f"Invalid lm_head weight loaded from teacher root {teacher_root}: {e}") from e
        self._lm_head_weight = jax.lax.stop_gradient(lm_w)

        dcfg = DFlashDraftModelConfig(
            hidden_size=int(meta.hidden_size),
            num_layers=int(arguments.draft_layers),
            mlp_ratio=float(arguments.mlp_ratio),
            hidden_act=str(arguments.hidden_act),
            num_attention_heads=int(teacher_cfg["num_attention_heads"]),
            num_key_value_heads=int(teacher_cfg["num_key_value_heads"]),
            head_dim=int(teacher_cfg["head_dim"]),
            rms_norm_eps=float(teacher_cfg.get("rms_norm_eps", 1e-5)),
            block_size=int(meta.block_size),
            num_context_features=int(meta.num_context_features),
            target_layer_ids=list(meta.target_layer_ids),
            add_one_for_pre_layer_capture=bool(getattr(meta, "add_one_for_pre_layer_capture", True)),
            qk_norm=bool(arguments.qk_norm),
            remat=bool(getattr(arguments, "remat", True)),
        )

        rngs = nnx.Rngs(0)
        draft_model = DFlashDraftModel(dcfg, rngs=rngs)
        draft_model.mesh = self._make_mesh()

        tx, _scheduler = arguments.get_optimizer_and_scheduler(int(arguments.max_training_steps or 1))
        state = EasyDeLState.create(model=draft_model, tx=tx, init_opt_state=True)

        # --- Custom resume (do NOT rely on EasyDeL's default resume path).
        setattr(arguments, "resume_if_possible", False)
        setattr(arguments, "resume_from_checkpoint", None)
        setattr(arguments, "resume_from", None)

        state = self._maybe_resume_state(state, mesh=draft_model.mesh)

        if train_dataset is None:
            from datasets import Dataset

            # IMPORTANT: EasyDeL's Trainer loop expects the input iterator to
            # sustain `max_training_steps`. With a finite Dataset, StopIteration
            # can terminate training early (well below max_training_steps),
            # especially for large global batch sizes.
            #
            # Build a deterministic repeated index stream long enough for the
            # configured training budget.
            cache_len = int(len(self.cache))
            target_rows = int(getattr(arguments, "max_training_steps", 0) or 0) * int(
                getattr(arguments, "total_batch_size", 1) or 1
            ) * int(getattr(arguments, "gradient_accumulation_steps", 1) or 1)
            if target_rows <= 0:
                target_rows = cache_len
            idx = (np.arange(target_rows, dtype=np.int64) % max(1, cache_len)).astype(np.int64)
            train_dataset = Dataset.from_dict({"idx": idx})

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=state,
            processing_class=processing_class,
            data_collator=_dflash_or_idx_collate,
        )

    def _maybe_resume_state(self, state: EasyDeLState, *, mesh: Mesh) -> EasyDeLState:
        if not bool(getattr(self.arguments, "resume", True)):
            return state

        run_root = Path(self.arguments.save_directory).resolve() / str(self.arguments.model_name)
        explicit = getattr(self.arguments, "resume_path", None)
        chosen: tuple[Path, int] | None
        if explicit:
            run_dir = Path(str(explicit)).expanduser().resolve()
            step = _parse_run_step(run_dir)
            if step is None or not _is_complete_run(run_dir):
                raise ValueError(f"Invalid resume_path={run_dir} (expected complete run-<step>/)")
            chosen = (run_dir, int(step))
        else:
            chosen = _find_latest_complete_run(run_root)

        if chosen is None:
            if bool(getattr(self.arguments, "resume_strict", False)):
                raise FileNotFoundError(f"No complete run-* checkpoint found under {run_root}")
            if jax.process_index() == 0:
                logger.warning("No checkpoint found under %s; starting fresh training.", run_root)
            return state

        run_dir, step = chosen
        if jax.process_index() == 0:
            logger.warning("Resuming DFlash training from %s (step=%d)", run_dir, step)

        # --- Multi-host TPU: avoid JAX multiprocess tensorstore serialization for
        # fully-addressable (replicated) arrays.
        #
        # On multi-host, JAX forbids "multiprocess serialization" for fully
        # addressable arrays because multiple processes could write the same path.
        # Our DFlash draft state is replicated under pmap, so we use a simple
        # per-host checkpoint format (msgpack bytes) instead.
        ckpt_dir = run_dir / "model"
        ckpt_msgpack = ckpt_dir / "graphstate.msgpack"
        ckpt_pkl = ckpt_dir / "graphstate.pkl"
        if ckpt_msgpack.exists():
            from flax import serialization as flax_serialization

            raw = ckpt_msgpack.read_bytes()
            try:
                # Legacy format: msgpack produced by flax_serialization.to_bytes(...).
                graphstate = flax_serialization.from_bytes(state.graphstate, raw)
            except Exception:
                # New format: msgpack of a flax serialization state_dict.
                state_dict = flax_serialization.msgpack_restore(raw)
                graphstate = flax_serialization.from_state_dict(state.graphstate, state_dict)
            opt_state = state.tx.init(graphstate)
            step_arr = jnp.asarray(int(step), dtype=jnp.int32)
            if jax.process_index() == 0:
                logger.warning("Restored replicated checkpoint (graphstate.msgpack); optimizer state re-initialized.")
            return state.replace(graphstate=graphstate, opt_state=opt_state, step=step_arr)
        elif ckpt_pkl.exists():
            import pickle
            from flax import serialization as flax_serialization

            raw = _maybe_gunzip(ckpt_pkl.read_bytes())
            graph_or_state_dict = pickle.loads(raw)
            try:
                graphstate = flax_serialization.from_state_dict(state.graphstate, graph_or_state_dict)
            except Exception:
                graphstate = graph_or_state_dict
            opt_state = state.tx.init(graphstate)
            step_arr = jnp.asarray(int(step), dtype=jnp.int32)
            if jax.process_index() == 0:
                logger.warning(
                    "Restored replicated checkpoint (%s); optimizer state re-initialized.",
                    "graphstate.msgpack" if ckpt_msgpack.exists() else "graphstate.pkl",
                )
            return state.replace(graphstate=graphstate, opt_state=opt_state, step=step_arr)

        self.arguments.ensure_checkpoint_path()
        ckpt = self.arguments.get_streaming_checkpointer()

        with mesh:
            graphstate, _extra_model = ckpt.load_pytree(
                mesh,
                prefix="model",
                path=str(run_dir),
                discover_latest=False,
                discover_raise=True,
                template=state.graphstate,
                strict_shapes=True,
            )
            opt_state = state.tx.init(graphstate)
            step_arr = jnp.asarray(int(step), dtype=jnp.int32)

        if jax.process_index() == 0:
            logger.warning("Optimizer state re-initialized on resume (model weights restored).")

        return state.replace(graphstate=graphstate, opt_state=opt_state, step=step_arr)

    def _save_state(self, state: EasyDeLState, save_directory: str | None = None, *args, **kwargs) -> str:
        # Multi-host TPU: do not use tensorstore multiprocess serialization for
        # fully addressable arrays; save a per-host replicated checkpoint instead.
        if jax.process_count() > 1:
            step = self._get_current_step(state)
            directory_name = self.arguments._get_save_directory_milestone(step=step, create=True)
            directory_name.mkdir(exist_ok=True)
            self.arguments.save_arguments(directory_name / DEFAULT_ARGS_JSON_NAME)
            self._save_readme(directory_name)

            model_dir = directory_name / "model"
            model_dir.mkdir(exist_ok=True)

            # Keep disk usage under control *before* writing a potentially large
            # checkpoint on TPU VM root disks.
            _ensure_free_space(
                directory_name.parent,
                min_free_gb=float(os.environ.get("DFLASH_MIN_FREE_GB", "8.0")),
                keep_runs=int(os.environ.get("DFLASH_KEEP_RUN_DIRS", "2")),
            )

            graphstate_host = jax.device_get(state.graphstate)
            # Prefer Flax msgpack (typically smaller + more stable) over pickle,
            # but fall back to pickle when the graphstate contains objects that
            # msgpack cannot serialize (e.g. flax.nnx State containers).
            from flax import serialization as flax_serialization

            saved_format = "dflash_replicated_pickle_v1"
            try:
                import pickle
                import gzip

                raw = pickle.dumps(graphstate_host, protocol=pickle.HIGHEST_PROTOCOL)
                # Default to gzip on multi-host to reduce IO and disk pressure.
                if str(os.environ.get("DFLASH_CHECKPOINT_GZIP", "1")).lower() in ("1", "true", "yes", "y", "on"):
                    raw = gzip.compress(raw, compresslevel=int(os.environ.get("DFLASH_CHECKPOINT_GZIP_LEVEL", "3")))
                    saved_format = "dflash_replicated_pickle_gzip_v2"
                _atomic_write_bytes(model_dir / "graphstate.pkl", raw)
            except Exception:
                # Legacy fallback: try Flax serialization. This can still fail
                # for some NNX graphstates but is occasionally smaller.
                _atomic_write_bytes(model_dir / "graphstate.msgpack", flax_serialization.to_bytes(graphstate_host))
                saved_format = "dflash_replicated_msgpack_v1"

            # Minimal completion marker for _is_complete_run().
            (directory_name / "metadata.json").write_text(
                json.dumps(
                    {
                        "format": saved_format,
                        "step": int(step),
                        "process_count": int(jax.process_count()),
                        "free_gb": float(_free_gb(directory_name)),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            # Avoid filling the small TPU VM root disk by keeping only a couple
            # of the most recent run-* directories on each host.
            _prune_old_run_dirs(directory_name.parent, keep=int(os.environ.get("DFLASH_KEEP_RUN_DIRS", "2")))
            return str(directory_name)

        return super()._save_state(state=state, save_directory=save_directory, *args, **kwargs)

    def _create_checkpointer(self):
        # BaseTrainer always instantiates a streaming Checkpointer. On multi-host
        # with replicated params, that path will (a) scan metadata in a format we
        # don't produce and (b) eventually hit JAX's multiprocess serialization
        # restriction for fully-addressable arrays.
        if jax.process_count() > 1:
            return _LocalStepCheckpointer(save_steps=int(getattr(self.arguments, "save_steps", 0) or 0))
        return super()._create_checkpointer()

    def configure_model(self) -> TrainerConfigureModelOutput:
        tx, scheduler = self.arguments.get_optimizer_and_scheduler(self.max_training_steps)
        return TrainerConfigureModelOutput(
            model=self.model,
            tx=tx,
            scheduler=scheduler,
            config=getattr(self.model, "config", None),
        )

    def _configure_state(self):
        mesh = self._make_mesh()
        empty = NamedSharding(mesh, P())

        if getattr(self.model_state, "opt_state", None) is None:
            with mesh:
                self.model_state = self.model_state.replace(opt_state=self.tx.init(self.model_state.graphstate))

        graphstate_sh = jax.tree_util.tree_map(lambda _: empty, self.model_state.graphstate)
        graphother_sh = jax.tree_util.tree_map(lambda _: empty, self.model_state.graphother)
        opt_sh = jax.tree_util.tree_map(lambda _: empty, self.model_state.opt_state)
        step_sh = empty

        with mesh:
            graphstate = jax.device_put(self.model_state.graphstate, graphstate_sh)
            graphother = jax.device_put(self.model_state.graphother, graphother_sh)
            opt_state = jax.device_put(self.model_state.opt_state, opt_sh)
            step = jax.device_put(self.model_state.step, step_sh)

        self.model_state = self.model_state.replace(
            graphstate=graphstate,
            graphother=graphother,
            opt_state=opt_state,
            step=step,
        )
        self.state_shardings = self.model_state.replace(
            graphstate=graphstate_sh,
            graphother=graphother_sh,
            opt_state=opt_sh,
            step=step_sh,
        )

    def _make_mesh(self) -> Mesh:
        devices = jax.devices()
        n = int(len(devices))
        if not self.arguments.spmd:
            return Mesh(np.array(devices), axis_names=("dp",))
        dp = int(self.arguments.dp)
        tp_size = int(self.arguments.tp)
        need = dp * tp_size
        if need > n:
            raise ValueError(f"Invalid dp/tp for devices: dp={dp} tp={tp_size} devices={n}")
        dev = np.array(devices[:need]).reshape((dp, tp_size))
        return Mesh(dev, axis_names=("dp", "tp"))

    def configure_dataloaders(self) -> TrainerConfigureDataloaderOutput:
        bs = int(self.training_batch_size)
        if bs <= 0:
            raise ValueError(f"Invalid training_batch_size={bs}")

        # This trainer shards via `jax.pmap`. Under multi-host JAX, pmap expects
        # host-local arrays shaped `[local_device_count, ...]`; JAX will
        # automatically combine them into global arrays across all hosts.
        #
        # Therefore, `training_batch_size` is interpreted as the GLOBAL batch
        # size. We split it evenly across all global devices, and each host
        # yields only its local shard.
        dp_local = int(jax.local_device_count())
        dp_global = dp_local * int(jax.process_count())
        if bs % dp_global != 0:
            raise ValueError(
                f"training_batch_size={bs} must be divisible by global_dp={dp_global} "
                f"(local_dp={dp_local} process_count={int(jax.process_count())})"
            )

        # Each host should fetch a disjoint shard of the GLOBAL batch.
        # Per-host batch = global_batch / process_count.
        bs_host = bs // int(jax.process_count())
        if bs_host <= 0:
            raise ValueError(f"Derived per-host batch is invalid: bs_host={bs_host} (bs={bs})")

        steps = int(self.arguments.max_training_steps or 0)
        if steps <= 0:
            epochs = float(self.arguments.num_train_epochs or 1.0)
            steps = int(max(1, (len(self.cache) * epochs) // bs))

        seed = int(getattr(self.arguments, "seed", 0) or 0)
        shuffle = bool(getattr(self.arguments, "shuffle_train_dataset", True))
        prefetch = int(getattr(self.arguments, "dataloader_prefetch", 128) or 128)
        workers = max(1, int(getattr(self.arguments, "dataloader_workers", 8) or 8))

        class _CachePrefetchLoader:
            def __iter__(self_inner):
                # Each host uses a different seed so that any fallback random
                # sampling doesn't duplicate across hosts.
                rng = np.random.default_rng(seed + int(jax.process_index()))
                n = len(self.cache)
                order = np.arange(n, dtype=np.int64)
                if shuffle:
                    rng.shuffle(order)
                # Each host takes a disjoint slice of the shuffled order.
                pos = int(jax.process_index()) * bs_host

                q: Queue = Queue(maxsize=prefetch)

                def _worker():
                    nonlocal pos
                    while True:
                        # If the cache is smaller than the *global* batch, we
                        # cannot take disjoint host slices without producing an
                        # empty slice on higher process_index hosts. Fall back
                        # to random sampling with replacement (host-seeded) so
                        # training stays correct and never yields a zero batch.
                        if (bs_host * int(jax.process_count())) > n:
                            idx = rng.integers(0, n, size=bs_host, dtype=np.int64)
                            q.put(self.cache.get_batch(idx))
                            continue
                        if bs_host > n:
                            idx = rng.integers(0, n, size=bs_host, dtype=np.int64)
                            q.put(self.cache.get_batch(idx))
                            continue
                        if pos + bs_host > n:
                            if shuffle:
                                rng.shuffle(order)
                            pos = int(jax.process_index()) * bs_host
                        idx = order[pos : pos + bs_host]
                        pos += bs_host * int(jax.process_count())
                        q.put(self.cache.get_batch(idx))

                for _ in range(workers):
                    Thread(target=_worker, daemon=True).start()

                per = bs // dp_global
                while True:
                    batch = q.get()
                    batch = {
                        "context_features_u16": batch["context_features_u16"].reshape(
                            (dp_local, per) + tuple(batch["context_features_u16"].shape[1:])
                        ),
                        "anchor_embedding_u16": batch["anchor_embedding_u16"].reshape(
                            (dp_local, per) + tuple(batch["anchor_embedding_u16"].shape[1:])
                        ),
                        "target_ids": batch["target_ids"].reshape(
                            (dp_local, per) + tuple(batch["target_ids"].shape[1:])
                        ),
                        "ctx_pos_start_i32": batch["ctx_pos_start_i32"].reshape(
                            (dp_local, per) + tuple(batch["ctx_pos_start_i32"].shape[1:])
                        ),
                    }
                    yield batch

        loader = _CachePrefetchLoader()
        return TrainerConfigureDataloaderOutput(
            dataloader_train=loader,
            dataloader_eval=loader,
            max_training_steps=steps,
            max_evaluation_steps=1,
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        meta = self.cache.meta
        rope = self._rope
        lm_w = self._lm_head_weight
        debug_shapes = _env_flag("DFLASH_DEBUG_SHAPES")
        debug_raise_shapes = _env_flag("DFLASH_DEBUG_RAISE_SHAPES")

        hidden_size = int(meta.hidden_size)
        ctx_len = int(meta.ctx_len)
        k = int(meta.num_context_features)
        k_hidden = int(k * hidden_size)
        block_size = int(meta.block_size)
        block = int(block_size - 1)

        vocab_size = int(lm_w.shape[0])
        requested = int(getattr(self.arguments, "vocab_chunk_size", 0) or 0)
        vocab_chunk = _choose_vocab_chunk(vocab_size=vocab_size, requested=requested)

        def dflash_train_step(state: EasyDeLState, batch: dict):
            batch_obj = _batch_from_dict(batch)
            context = _bf16_from_u16(batch_obj.context_u16).reshape((-1, ctx_len, k_hidden))
            anchor = _bf16_from_u16(batch_obj.anchor_u16).reshape((-1, hidden_size))
            labels = batch_obj.target_ids.astype(jnp.int32).reshape((-1, block))
            ctx_pos_start = batch_obj.ctx_pos_start.astype(jnp.int32).reshape((-1,))

            graphdef = state.graphdef
            graphother = state.graphother

            def loss_fn(graphstate: tp.Any) -> jax.Array:
                module = nnx.merge(graphdef, graphstate, graphother)
                out = module(context_features=context, anchor_embedding=anchor, rope=rope, ctx_pos_start=ctx_pos_start)
                hs = out[:, 1:, :]
                if debug_raise_shapes:
                    raise ValueError(
                        "DFLASH_DEBUG_RAISE_SHAPES "
                        f"module={module.__class__.__name__} "
                        f"context={context.shape} anchor={anchor.shape} "
                        f"out={out.shape} hs={hs.shape} labels={labels.shape} lm_w={lm_w.shape}"
                    )
                if debug_shapes:
                    jax.debug.print(
                        "[dflash][train] context={c} anchor={a} out={o} hs={h} labels={l} lm_w={w}",
                        c=context.shape,
                        a=anchor.shape,
                        o=out.shape,
                        h=hs.shape,
                        l=labels.shape,
                        w=lm_w.shape,
                    )
                loss, acc = _chunked_ce_nll_and_acc(hs=hs, labels=labels, lm_w=lm_w, vocab_chunk=vocab_chunk)
                # Stash accuracy in a closed-over variable by returning it via aux.
                # jax.value_and_grad supports aux through `has_aux=True`.
                return loss, acc

            (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.graphstate)
            loss = jax.lax.pmean(loss, "dp")
            grads = jax.lax.pmean(grads, "dp")
            acc = jax.lax.pmean(acc, "dp")
            state2 = state.apply_gradients(grads=grads)
            return state2, LossMetrics(loss=loss, accuracy=acc)

        grad_accum = int(getattr(self.arguments, "gradient_accumulation_steps", 1) or 1)
        if grad_accum <= 1:
            sharded_training_step_function = jax.pmap(
                dflash_train_step,
                axis_name="dp",
                in_axes=(None, 0),
                out_axes=(None, None),
                donate_argnums=(0,),
            )
        else:
            # Gradient accumulation: split batch along batch dimension within each dp shard.
            def dflash_train_step_accum(state: EasyDeLState, batch: dict):
                batch_obj = _batch_from_dict(batch)
                ctx_u16 = batch_obj.context_u16
                anc_u16 = batch_obj.anchor_u16
                tgt_ids = batch_obj.target_ids
                pos = batch_obj.ctx_pos_start

                bsz = int(ctx_u16.shape[0])
                micro = max(1, bsz // grad_accum)
                bsz_use = micro * grad_accum
                ctx_u16 = ctx_u16[:bsz_use].reshape((grad_accum, micro) + ctx_u16.shape[1:])
                anc_u16 = anc_u16[:bsz_use].reshape((grad_accum, micro) + anc_u16.shape[1:])
                tgt_ids = tgt_ids[:bsz_use].reshape((grad_accum, micro) + tgt_ids.shape[1:])
                pos = pos[:bsz_use].reshape((grad_accum, micro) + pos.shape[1:])

                graphdef = state.graphdef
                graphother = state.graphother

                def body(carry, i):
                    loss_sum, grads_sum = carry
                    ctx_mb = ctx_u16[i]
                    anc_mb = anc_u16[i]
                    tgt_mb = tgt_ids[i]
                    pos_mb = pos[i]
                    context = _bf16_from_u16(ctx_mb).reshape((-1, ctx_len, k_hidden))
                    anchor = _bf16_from_u16(anc_mb).reshape((-1, hidden_size))
                    labels = tgt_mb.astype(jnp.int32).reshape((-1, block))
                    ctx_pos_start = pos_mb.astype(jnp.int32).reshape((-1,))

                    def loss_fn(graphstate: tp.Any) -> jax.Array:
                        module = nnx.merge(graphdef, graphstate, graphother)
                        out = module(context_features=context, anchor_embedding=anchor, rope=rope, ctx_pos_start=ctx_pos_start)
                        hs = out[:, 1:, :]
                        loss, acc = _chunked_ce_nll_and_acc(hs=hs, labels=labels, lm_w=lm_w, vocab_chunk=vocab_chunk)
                        return loss, acc

                    (loss_i, acc_i), grads_i = jax.value_and_grad(loss_fn, has_aux=True)(state.graphstate)
                    if grads_sum is None:
                        grads_sum = grads_i
                    else:
                        grads_sum = jax.tree_util.tree_map(lambda a, b: a + b, grads_sum, grads_i)
                    # Track only loss for accumulation; accuracy is recomputed at the end.
                    return (loss_sum + loss_i, grads_sum), None

                init = (jnp.array(0.0, dtype=jnp.float32), None)
                (loss_sum, grads_sum), _ = jax.lax.scan(body, init, jnp.arange(grad_accum, dtype=jnp.int32))
                loss = loss_sum / float(grad_accum)
                grads = jax.tree_util.tree_map(lambda g: g / float(grad_accum), grads_sum)
                loss = jax.lax.pmean(loss, "dp")
                grads = jax.lax.pmean(grads, "dp")
                # Accuracy is non-additive; compute on the full micro-batch once.
                context_full = _bf16_from_u16(ctx_u16.reshape((-1,) + ctx_u16.shape[2:])).reshape((-1, ctx_len, k_hidden))
                anchor_full = _bf16_from_u16(anc_u16.reshape((-1,) + anc_u16.shape[2:])).reshape((-1, hidden_size))
                labels_full = tgt_ids.reshape((-1,) + tgt_ids.shape[2:]).astype(jnp.int32).reshape((-1, block))
                pos_full = pos.reshape((-1,) + pos.shape[2:]).astype(jnp.int32).reshape((-1,))

                module = nnx.merge(graphdef, state.graphstate, graphother)
                out_full = module(context_features=context_full, anchor_embedding=anchor_full, rope=rope, ctx_pos_start=pos_full)
                hs_full = out_full[:, 1:, :]
                _loss_unused, acc = _chunked_ce_nll_and_acc(hs=hs_full, labels=labels_full, lm_w=lm_w, vocab_chunk=vocab_chunk)
                acc = jax.lax.pmean(acc, "dp")
                state2 = state.apply_gradients(grads=grads)
                return state2, LossMetrics(loss=loss, accuracy=acc)

            sharded_training_step_function = jax.pmap(
                dflash_train_step_accum,
                axis_name="dp",
                in_axes=(None, 0),
                out_axes=(None, None),
                donate_argnums=(0,),
            )

        def dflash_eval_step(state: EasyDeLState, batch: dict):
            batch_obj = _batch_from_dict(batch)
            context = _bf16_from_u16(batch_obj.context_u16).reshape((-1, ctx_len, k_hidden))
            anchor = _bf16_from_u16(batch_obj.anchor_u16).reshape((-1, hidden_size))
            labels = batch_obj.target_ids.astype(jnp.int32).reshape((-1, block))
            module = nnx.merge(state.graphdef, state.graphstate, state.graphother)
            ctx_pos_start = batch_obj.ctx_pos_start.astype(jnp.int32).reshape((-1,))
            out = module(context_features=context, anchor_embedding=anchor, rope=rope, ctx_pos_start=ctx_pos_start)
            hs = out[:, 1:, :]
            if debug_raise_shapes:
                raise ValueError(
                    "DFLASH_DEBUG_RAISE_SHAPES "
                    f"module={module.__class__.__name__} "
                    f"context={context.shape} anchor={anchor.shape} "
                    f"out={out.shape} hs={hs.shape} labels={labels.shape} lm_w={lm_w.shape}"
                )
            if debug_shapes:
                jax.debug.print(
                    "[dflash][eval] context={c} anchor={a} out={o} hs={h} labels={l} lm_w={w}",
                    c=context.shape,
                    a=anchor.shape,
                    o=out.shape,
                    h=hs.shape,
                    l=labels.shape,
                    w=lm_w.shape,
                )
            loss, acc = _chunked_ce_nll_and_acc(hs=hs, labels=labels, lm_w=lm_w, vocab_chunk=vocab_chunk)
            loss = jax.lax.pmean(loss, "dp")
            acc = jax.lax.pmean(acc, "dp")
            return LossMetrics(loss=loss, accuracy=acc)

        sharded_evaluation_step_function = jax.pmap(
            dflash_eval_step,
            axis_name="dp",
            in_axes=(None, 0),
            out_axes=None,
        )

        self.arguments.ensure_checkpoint_path()
        if jax.process_count() > 1:
            checkpoint_manager = _LocalStepCheckpointer(save_steps=int(getattr(self.arguments, "save_steps", 0) or 0))
        else:
            checkpoint_manager = self.arguments.get_streaming_checkpointer()

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=self.model.mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def on_step_end(self, state: EasyDeLState, metrics: LossMetrics, step: int):
        try:
            if jax.process_index() == 0:
                every = int(getattr(self.arguments, "report_steps", 0) or 0)
                if every > 0 and (int(step) % every == 0):
                    other = dict(metrics.other_metrics or {})
                    try:
                        exec_s = float(getattr(metrics, "execution_time", 0.0) or 0.0)
                        if exec_s > 0:
                            draft_tokens_per_sample = int(self.cache.meta.block_size) - 1
                            global_batch = int(self.training_batch_size)
                            draft_tokens = float(draft_tokens_per_sample * global_batch)
                            other["draft_tokens_per_step"] = draft_tokens
                            other["draft_tokens_per_s"] = draft_tokens / exec_s
                    except Exception:
                        pass

                    dev = jax.devices()[0]
                    mem = getattr(dev, "memory_stats", None)
                    if callable(mem):
                        st = mem()
                        used = st.get("memory_used", None) or st.get("hbm_memory_used", None)
                        limit = st.get("memory_limit", None) or st.get("hbm_memory_total", None)
                        if used is not None:
                            other["tpu_mem_used_gb"] = float(used) / (1024**3)
                            if limit is not None:
                                other["tpu_mem_limit_gb"] = float(limit) / (1024**3)
                                other["tpu_mem_used_pct"] = 100.0 * float(used) / float(limit)
                    if other:
                        metrics = metrics.replace(other_metrics=other)
        except Exception:
            pass
        return state, metrics


def _dflash_collate(examples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    ctx = np.stack([ex["context_features_u16"] for ex in examples], axis=0).astype(np.uint16, copy=False)
    anc = np.stack([ex["anchor_embedding_u16"] for ex in examples], axis=0).astype(np.uint16, copy=False)
    tgt = np.stack([ex["target_ids"] for ex in examples], axis=0).astype(np.int32, copy=False)
    pos = np.stack(
        [ex.get("ctx_pos_start_i32", np.asarray(0, dtype=np.int32)) for ex in examples],
        axis=0,
    ).astype(np.int32, copy=False)
    return {"context_features_u16": ctx, "anchor_embedding_u16": anc, "target_ids": tgt, "ctx_pos_start_i32": pos}


def _idx_collate(examples: list[dict[str, tp.Any]]) -> dict[str, np.ndarray]:
    idx = np.asarray([int(ex["idx"]) for ex in examples], dtype=np.int64)
    return {"idx": idx}


def _dflash_or_idx_collate(batch: tp.Any) -> dict[str, np.ndarray]:
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, list):
        if batch and isinstance(batch[0], dict) and "context_features_u16" in batch[0]:
            return _dflash_collate(batch)  # type: ignore[arg-type]
        if batch and isinstance(batch[0], dict) and "idx" in batch[0]:
            return _idx_collate(batch)  # type: ignore[arg-type]
    return batch
