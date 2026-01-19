from __future__ import annotations

import json
import os
import typing as tp
from dataclasses import dataclass
from pathlib import Path
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


def _set_shm_caches() -> None:
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_tpu_enable_latency_hiding_scheduler" not in xla_flags:
        os.environ["XLA_FLAGS"] = (xla_flags + " --xla_tpu_enable_latency_hiding_scheduler=true").strip()


def _require_token_present() -> None:
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
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
        _require_token_present()

        if not arguments.cache_dir:
            raise ValueError("DFlashConfig.cache_dir is required")
        if not arguments.teacher_snapshot_dir:
            raise ValueError("DFlashConfig.teacher_snapshot_dir is required")

        self.arguments = arguments
        self.cache = DFlashTeacherCacheDataset(arguments.cache_dir)
        self.teacher_snapshot = Path(arguments.teacher_snapshot_dir).resolve()

        meta = self.cache.meta
        if meta.dtype not in ("bf16_u16",):
            raise ValueError(f"Unsupported cache dtype {meta.dtype!r}; expected bf16_u16")

        teacher_cfg = json.loads((self.teacher_snapshot / "config.json").read_text(encoding="utf-8"))
        self._rope = _build_rope(cfg=teacher_cfg, dtype=jnp.bfloat16)
        self._lm_head_weight = jax.lax.stop_gradient(_load_lm_head_weight(self.teacher_snapshot))

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

            train_dataset = Dataset.from_dict({"idx": np.arange(len(self.cache), dtype=np.int64)})

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

        use_spmd = bool(self.arguments.spmd)
        dp = int(self.arguments.dp) if use_spmd else int(jax.local_device_count())
        if bs % dp != 0:
            raise ValueError(f"training_batch_size={bs} must be divisible by dp={dp}")

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
                rng = np.random.default_rng(seed)
                n = len(self.cache)
                order = np.arange(n, dtype=np.int64)
                if shuffle:
                    rng.shuffle(order)
                pos = 0

                q: Queue = Queue(maxsize=prefetch)

                def _worker():
                    nonlocal pos
                    while True:
                        if bs > n:
                            idx = rng.integers(0, n, size=bs, dtype=np.int64)
                            q.put(self.cache.get_batch(idx))
                            continue
                        if pos + bs > n:
                            if shuffle:
                                rng.shuffle(order)
                            pos = 0
                        idx = order[pos : pos + bs]
                        pos += bs
                        q.put(self.cache.get_batch(idx))

                for _ in range(workers):
                    Thread(target=_worker, daemon=True).start()

                per = bs // dp
                while True:
                    batch = q.get()
                    if not use_spmd:
                        batch = {
                            "context_features_u16": batch["context_features_u16"].reshape(
                                (dp, per) + tuple(batch["context_features_u16"].shape[1:])
                            ),
                            "anchor_embedding_u16": batch["anchor_embedding_u16"].reshape(
                                (dp, per) + tuple(batch["anchor_embedding_u16"].shape[1:])
                            ),
                            "target_ids": batch["target_ids"].reshape((dp, per) + tuple(batch["target_ids"].shape[1:])),
                            "ctx_pos_start_i32": batch["ctx_pos_start_i32"].reshape(
                                (dp, per) + tuple(batch["ctx_pos_start_i32"].shape[1:])
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
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=self.model.mesh,
            checkpoint_manager=self.arguments.get_streaming_checkpointer(),
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
