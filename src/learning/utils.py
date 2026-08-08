import os
from pathlib import Path

def resolve_checkpoint(experiment_dir):
    """
    Resolves the checkpoint path according to repository conventions.
    Returns best.ckpt if present, else last.ckpt. Raises FileNotFoundError if neither exist.
    """
    out_dir = Path(experiment_dir)
    ckpt_path = out_dir / "best.ckpt"
    if not ckpt_path.exists():
        ckpt_path = out_dir / "last.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint found in {out_dir} (checked best.ckpt and last.ckpt)")
    return ckpt_path
