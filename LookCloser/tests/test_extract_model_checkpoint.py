from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_model_checkpoint.py"


def test_extracts_only_evaluation_state(tmp_path: Path) -> None:
    source = tmp_path / "source.ckpt"
    output = tmp_path / "model.ckpt"
    torch.save(
        {
            "step": 12,
            "pipeline": {"weight": torch.ones(3)},
            "optimizers": {"large": torch.ones(20)},
            "schedulers": {"unused": 1},
        },
        source,
    )
    subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(output), "--metric-step", "11"], check=True
    )
    compact = torch.load(output, map_location="cpu", weights_only=False)
    assert sorted(compact) == ["pipeline", "step"]
    assert compact["step"] == 12
    assert output.with_suffix(".ckpt.json").is_file()
