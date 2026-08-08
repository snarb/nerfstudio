from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from interpolate_lookcloser_checkpoints import checkpoint_step  # noqa: E402


def test_checkpoint_step_loads_trusted_checkpoint_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-with-metadata.ckpt"
    torch.save({"step": 123, "numpy_metadata": np.array([1, 2, 3])}, checkpoint)

    assert checkpoint_step(checkpoint) == 123
