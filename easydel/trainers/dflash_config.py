from __future__ import annotations

from dataclasses import dataclass, field

from easydel.trainers.training_configurations import TrainingArguments
from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn


@Registry.register("trainer-arguments", "dflash")
@dataclass
class DFlashConfig(TrainingArguments):
    trainer_prefix: str | None = field(
        default="dflashtrainer",
        metadata={"help": "Prefix used for trainer logs, checkpoints, and wandb runs."},
    )

    # Cache-first training: we do NOT run the teacher forward in the loop.
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Path to a DFlash teacher cache directory (meta.json + .npy)."},
    )
    teacher_snapshot_dir: str | None = field(
        default=None,
        metadata={"help": "HF snapshot dir for the teacher model (used only to load lm_head.weight)."},
    )

    # Draft architecture.
    draft_layers: int = field(default=8, metadata={"help": "Number of draft transformer layers."})
    mlp_ratio: float = field(default=4.0, metadata={"help": "Draft MLP expansion ratio."})
    hidden_act: str = field(default="silu", metadata={"help": "Draft MLP activation."})
    qk_norm: bool = field(default=True, metadata={"help": "Enable per-head Q/K RMS normalization in draft attention."})

    # Exact CE loss implementation.
    vocab_chunk_size: int = field(
        default=0,
        metadata={"help": "If >0, uses chunked full-vocab CE (debug). Prefer tp sharding with --spmd."},
    )

    # Data pipeline tuning (cache mmap -> numpy -> device).
    dataloader_prefetch: int = field(
        default=128,
        metadata={"help": "Prefetch queue depth for cache batches (CPU thread -> main thread)."},
    )
    dataloader_workers: int = field(
        default=4,
        metadata={"help": "Number of CPU worker threads to build cache batches."},
    )
    device_prefetch: int = field(
        default=2,
        metadata={"help": "How many batches to prefetch to device (dp/pmap mode only)."},
    )

    # Memory/perf knobs.
    remat: bool = field(
        default=True,
        metadata={"help": "Use activation checkpointing (remat) for draft blocks to reduce HBM and allow larger batch."},
    )

    # SPMD mesh (dp×tp) for exact vocab-parallel CE.
    spmd: bool = field(default=False, metadata={"help": "Enable dp×tp SPMD training step."})
    dp: int = field(default=8, metadata={"help": "Data-parallel size (dp×tp must equal num_devices)."})
    tp: int = field(default=1, metadata={"help": "Vocab-parallel size (dp×tp must equal num_devices)."})

    # Checkpointing / resume.
    save_optimizer_state: bool = field(
        default=False,
        metadata={"help": "Disable optimizer checkpointing (DFlash draft runs can resume from weights only)."},
    )
    resume: bool = field(
        default=True,
        metadata={
            "help": "Resume from the latest complete checkpoint under save_directory/model_name (cache-first runs only)."
        },
    )
    resume_strict: bool = field(
        default=False,
        metadata={"help": "If true, error when resume is requested but no valid checkpoint exists."},
    )
    resume_path: str | None = field(
        default=None,
        metadata={
            "help": "Optional explicit run-* directory to resume from. When set, it overrides auto-discovery."
        },
    )

    __hash__ = hash_fn

