from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_lookcloser_quiet.py"
SPEC = importlib.util.spec_from_file_location("run_lookcloser_quiet_surface_opt_in", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_surface_stage_is_disabled_by_default(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "dataset"
    data.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--data", str(data), "--allow-missing-frequency-maps"],
    )

    args = RUNNER.parse_args()

    assert args.surface_light_field_depth_manifest is None
    assert RUNNER.run_surface_light_field({}, args) == {"status": "disabled"}
    assert not any("surface-light-field" in token for token in RUNNER.train_command(args))


def test_surface_stage_runs_only_after_explicit_manifest(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "dataset"
    data.mkdir()
    manifest = data / "mesh_depth_manifest.json"
    manifest.write_text('{"images": []}', encoding="utf-8")
    roi = data / "roi.json"
    roi.write_text('{"boxes_xyxy": [[0, 0, 1, 1]]}', encoding="utf-8")
    render_dir = tmp_path / "renders_best"
    render_dir.mkdir()
    (render_dir / "eval_img_0000.png").touch()
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        command = [str(value) for value in command]
        calls.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        variant = output_dir / "blend2_detail_s4_w1"
        variant.mkdir(parents=True)
        (variant / "metrics.json").write_text(
            json.dumps({"aggregate": {"psnr": 25.0, "ssim": 0.8, "lpips": 0.2, "roi_lpips": 0.16}}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(RUNNER.subprocess, "run", fake_run)
    args = SimpleNamespace(
        surface_light_field_depth_manifest=manifest,
        surface_light_field_neighbors=2,
        surface_light_field_blend_alpha=1.0,
        surface_light_field_camera_distance_power=4.0,
        surface_light_field_depth_log_tolerance=0.01,
        surface_light_field_detail_transfer_sigma=4.0,
        surface_light_field_detail_transfer_strength=1.0,
        surface_light_field_roi_boxes_json=roi,
        data=data,
        eval_mode="filename",
        eval_interval=8,
        orientation_method="up",
        center_method="focus",
        scale_factor=1.0,
        scene_scale=2.0,
    )

    result = RUNNER.run_surface_light_field({"render_dir": str(render_dir)}, args)

    assert result["status"] == "complete"
    assert result["variant"] == "blend2_detail_s4_w1"
    assert result["aggregate"]["roi_lpips"] == 0.16
    assert len(calls) == 1
    assert "--score-metrics" in calls[0]
    assert "--roi-boxes-json" in calls[0]
