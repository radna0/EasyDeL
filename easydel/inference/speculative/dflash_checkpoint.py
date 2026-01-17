from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import typing as tp


@dataclass(frozen=True)
class DFlashCheckpointRef:
    run_dir: Path
    step: int


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
    return True


def find_latest_complete_run(run_root: str | Path) -> DFlashCheckpointRef | None:
    root = Path(run_root).expanduser().resolve()
    if not root.is_dir():
        return None
    best: DFlashCheckpointRef | None = None
    for child in root.iterdir():
        step = _parse_run_step(child)
        if step is None:
            continue
        if not _is_complete_run(child):
            continue
        if best is None or step > best.step:
            best = DFlashCheckpointRef(run_dir=child, step=step)
    return best


def load_dflash_graphstate_from_run(
    *,
    run_dir: str | Path,
    mesh: tp.Any,
    template_graphstate: tp.Any,
    partition_rules: tp.Any = (),
    strict_shapes: bool = True,
) -> tp.Any:
    """Load a DFlash draft model graphstate from an EasyDeL `run-*/` directory.

    This is intentionally lightweight and does not require TrainerArguments.
    """

    from eformer.serialization.checkpointer import Checkpointer

    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"Missing run_dir: {run_path}")

    ckpt = Checkpointer(base_path=str(run_path), save_interval=None, step_policies=[])
    with mesh:
        graphstate, _extra = ckpt.load_pytree(
            mesh,
            prefix="model",
            path=str(run_path),
            discover_latest=False,
            discover_raise=True,
            partition_rules=partition_rules,
            template=template_graphstate,
            strict_shapes=bool(strict_shapes),
        )
    return graphstate
