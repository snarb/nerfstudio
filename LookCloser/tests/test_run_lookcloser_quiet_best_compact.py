from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_lookcloser_quiet as runner  # noqa: E402


def test_unchanged_metric_winner_does_not_replace_exact_compact_with_latest_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    run = tmp_path / "run"
    models = run / "nerfstudio_models"
    models.mkdir(parents=True)
    current = models / "step-000091128.ckpt"
    current.write_bytes(b"later-full")
    compact = run / "best_eval_model.ckpt"
    compact.write_bytes(b"exact-best-83534")
    sidecar = compact.with_suffix(".ckpt.json")
    sidecar.write_text(
        json.dumps(
            {
                "metric_step": 83534,
                "selection": "best_psnr34.564_lpips_tiebreak_step_83534",
            }
        ),
        encoding="utf-8",
    )
    fallback = "latest_no_checkpoint_for_best_psnr34.564_lpips_tiebreak_step_83534"
    monkeypatch.setattr(runner, "best_eval_checkpoint", lambda *_: (current, fallback))
    monkeypatch.setattr(runner, "eval_rows", lambda *_: [{"step": "91128"}])

    def unexpected_extract(*_args, **_kwargs):
        raise AssertionError("unchanged winner must retain the existing exact compact checkpoint")

    monkeypatch.setattr(runner.subprocess, "run", unexpected_extract)
    assert runner.preserve_best_eval_model_checkpoint(tmp_path / "metrics.csv", models, run) == fallback
    assert compact.read_bytes() == b"exact-best-83534"


def test_selection_metric_step_uses_intended_winner_not_fallback_source() -> None:
    assert (
        runner.selection_metric_step(
            "latest_no_checkpoint_for_best_psnr34.564_lpips_tiebreak_step_83534"
        )
        == 83534
    )
