"""Regression test for exact intermediate-checkpoint evaluation binding."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest


RUNNER = Path(__file__).resolve().parents[2] / "LookCloser" / "scripts" / "run_lookcloser_quiet.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("lookcloser_quiet_for_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_config_binds_exact_intermediate_step(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    loaded = SimpleNamespace(
        load_dir=tmp_path / "wrong",
        load_step=None,
        load_checkpoint=None,
        pipeline=SimpleNamespace(model=SimpleNamespace(eval_num_rays_per_chunk=2048)),
    )
    monkeypatch.setattr(runner.yaml, "load", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(runner.yaml, "dump", lambda value: repr(value))

    config = tmp_path / "config.yml"
    config.write_text("placeholder", encoding="utf-8")
    checkpoint = tmp_path / "nerfstudio_models" / "step-000045564.ckpt"

    output = runner.eval_config_for_step(config, checkpoint, 16384)

    assert output == tmp_path / "eval_config_step_45564.yml"
    assert loaded.load_dir is None
    assert loaded.load_step == 45_564
    assert loaded.load_checkpoint == checkpoint
    assert loaded.pipeline.model.eval_num_rays_per_chunk == 16384


def test_eval_config_candidate_overrides_are_opt_in(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    loaded = SimpleNamespace(
        load_dir=tmp_path / "wrong",
        load_step=None,
        load_checkpoint=None,
        pipeline=SimpleNamespace(
            model=SimpleNamespace(eval_num_rays_per_chunk=2048),
            datamanager=SimpleNamespace(cache_train_rays=True, cpu_fas_prefetch=True),
        ),
    )
    monkeypatch.setattr(runner.yaml, "load", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(runner.yaml, "dump", lambda value: repr(value))

    config = tmp_path / "config.yml"
    config.write_text("placeholder", encoding="utf-8")
    checkpoint = tmp_path / "nerfstudio_models" / "step-000091128.ckpt"

    output = runner.eval_config_for_step(
        config,
        checkpoint,
        8192,
        cache_train_rays=False,
        filename_tag="candidate_pruned",
    )

    assert output == tmp_path / "eval_config_candidate_pruned_step_91128.yml"
    assert loaded.pipeline.datamanager.cache_train_rays is False
    assert loaded.pipeline.datamanager.cpu_fas_prefetch is False


def _artifact_args(*, gate_only: bool) -> SimpleNamespace:
    return SimpleNamespace(
        artifact_score=True,
        artifact_roi_score=True,
        artifact_render_names="eval_img_0000.png,eval_img_0001.png,eval_img_0002.png",
        artifact_render_name="eval_img_0000.png",
        artifact_detector_preset="significant",
        artifact_roi_crop_names="all",
        artifact_roi_drop_border_components=0,
        artifact_crop_top=0,
        artifact_crop_bottom=0,
        artifact_crop_left=0,
        artifact_crop_right=0,
        artifact_gate_only=gate_only,
    )


def _write_render_placeholders(render_dir: Path) -> None:
    render_dir.mkdir()
    for index in range(3):
        (render_dir / f"eval_img_{index:04d}.png").write_bytes(b"render")


def _clean_artifact_stdout(index: int) -> str:
    return (
        f"[candidate] serious=False artifact_score={index}.000 count=0 largest=0px "
        "serious_artifact_score=0.000\n"
    )


def test_candidate_artifact_views_run_concurrently_but_keep_input_order(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    render_dir = tmp_path / "renders_candidate_step-000091128"
    _write_render_placeholders(render_dir)
    barrier = threading.Barrier(3)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_run(command, **_kwargs):
        nonlocal active, max_active
        index = int(Path(command[2]).stem.rsplit("_", 1)[-1])
        with lock:
            active += 1
            max_active = max(max_active, active)
        barrier.wait(timeout=2)
        time.sleep((2 - index) * 0.01)
        with lock:
            active -= 1
        return subprocess.CompletedProcess(command, 0, stdout=_clean_artifact_stdout(index))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runner,
        "run_roi_artifact_scorer",
        lambda *_args: {"status": "complete", "roi_artifact_score": 0.0},
    )

    result = runner.run_artifact_detector(
        tmp_path,
        {"render_dir": str(render_dir)},
        _artifact_args(gate_only=True),
    )

    assert max_active == 3
    assert [view["render_name"] for view in result["views"]] == [
        "eval_img_0000.png",
        "eval_img_0001.png",
        "eval_img_0002.png",
    ]
    assert [view["artifact_score"] for view in result["views"]] == [0.0, 1.0, 2.0]
    assert result["status"] == "complete"
    assert result["artifact_views_scored"] == 3


def test_candidate_artifact_view_failure_is_fail_closed(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    render_dir = tmp_path / "renders_candidate_step-000091128"
    _write_render_placeholders(render_dir)

    def fake_run(command, **_kwargs):
        index = int(Path(command[2]).stem.rsplit("_", 1)[-1])
        if index == 1:
            raise OSError("detector failed")
        return subprocess.CompletedProcess(command, 0, stdout=_clean_artifact_stdout(index))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runner,
        "run_roi_artifact_scorer",
        lambda *_args: {"status": "complete", "roi_artifact_score": 0.0},
    )

    result = runner.run_artifact_detector(
        tmp_path,
        {"render_dir": str(render_dir)},
        _artifact_args(gate_only=True),
    )

    assert result["status"] == "failed"
    assert result["artifact_views_requested"] == 3
    assert result["artifact_views_scored"] == 2
    assert [view["status"] for view in result["views"]] == ["complete", "error", "complete"]


@pytest.mark.parametrize(
    "render_names",
    [
        "eval_img_0000.png,eval_img_0001.png",
        "eval_img_0000.png,eval_img_0001.png,eval_img_0001.png",
    ],
)
def test_candidate_artifact_gate_requires_exactly_three_unique_views(
    tmp_path: Path,
    render_names: str,
) -> None:
    runner = _load_runner()
    render_dir = tmp_path / "renders_candidate_step-000091128"
    _write_render_placeholders(render_dir)
    args = _artifact_args(gate_only=True)
    args.artifact_render_names = render_names

    with pytest.raises(ValueError, match="exactly three unique"):
        runner.run_artifact_detector(tmp_path, {"render_dir": str(render_dir)}, args)


def test_default_artifact_mode_stays_sequential(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    render_dir = tmp_path / "renders_candidate_step-000091128"
    _write_render_placeholders(render_dir)
    calls = []

    def fake_run(command, **_kwargs):
        index = int(Path(command[2]).stem.rsplit("_", 1)[-1])
        calls.append(index)
        return subprocess.CompletedProcess(command, 0, stdout=_clean_artifact_stdout(index))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runner,
        "run_roi_artifact_scorer",
        lambda *_args: {"status": "complete", "roi_artifact_score": 0.0},
    )

    runner.run_artifact_detector(
        tmp_path,
        {"render_dir": str(render_dir)},
        _artifact_args(gate_only=False),
    )

    assert calls == [0, 1, 2]


@pytest.mark.parametrize("gate_only,write_images", [(False, True), (True, False)])
def test_roi_visualizations_are_omitted_only_for_candidate_gate(
    monkeypatch,
    tmp_path: Path,
    gate_only: bool,
    write_images: bool,
) -> None:
    runner = _load_runner()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.run_roi_artifact_scorer(
        tmp_path / "renders",
        tmp_path / "artifacts",
        _artifact_args(gate_only=gate_only),
    )

    assert ("--write-images" in commands[0]) is write_images


def test_tcnn_network_jit_is_explicit_and_default_off(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", [str(RUNNER)])
    default_args = runner.parse_args()
    default_command = runner.train_command(default_args)
    option = default_command.index("--pipeline.model.tcnn-network-jit")
    assert default_args.tcnn_network_jit is False
    assert default_command[option + 1] == "False"
    scope = default_command.index("--pipeline.model.tcnn-network-jit-scope")
    assert default_command[scope + 1] == "both"

    monkeypatch.setattr(
        sys,
        "argv",
        [str(RUNNER), "--tcnn-network-jit", "--tcnn-network-jit-scope", "color"],
    )
    jit_args = runner.parse_args()
    jit_command = runner.train_command(jit_args)
    option = jit_command.index("--pipeline.model.tcnn-network-jit")
    assert jit_args.tcnn_network_jit is True
    assert jit_command[option + 1] == "True"
    scope = jit_command.index("--pipeline.model.tcnn-network-jit-scope")
    assert jit_command[scope + 1] == "color"


def test_training_ray_cache_is_explicit_and_default_off(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", [str(RUNNER)])
    default_args = runner.parse_args()
    default_command = runner.train_command(default_args)
    option = default_command.index("--pipeline.datamanager.cache-train-rays")
    assert default_args.cache_train_rays is False
    assert default_command[option + 1] == "False"

    monkeypatch.setattr(
        sys,
        "argv",
        [str(RUNNER), "--cache-train-rays", "--cache-train-rays-chunk-size", "65536"],
    )
    cache_args = runner.parse_args()
    cache_command = runner.train_command(cache_args)
    enabled = cache_command.index("--pipeline.datamanager.cache-train-rays")
    chunk = cache_command.index("--pipeline.datamanager.cache-train-rays-chunk-size")
    assert cache_command[enabled + 1] == "True"
    assert cache_command[chunk + 1] == "65536"


def test_cpu_fas_prefetch_is_explicit_default_off_and_fail_closed(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", [str(RUNNER)])
    default_args = runner.parse_args()
    default_command = runner.train_command(default_args)
    option = default_command.index("--pipeline.datamanager.cpu-fas-prefetch")
    assert default_args.cpu_fas_prefetch is False
    assert default_command[option + 1] == "False"

    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--cpu-fas-prefetch"])
    with pytest.raises(ValueError, match="requires --cache-train-rays"):
        runner.train_command(runner.parse_args())

    monkeypatch.setattr(
        sys,
        "argv",
        [str(RUNNER), "--cpu-fas-prefetch", "--cache-train-rays"],
    )
    enabled_args = runner.parse_args()
    enabled_command = runner.train_command(enabled_args)
    option = enabled_command.index("--pipeline.datamanager.cpu-fas-prefetch")
    assert enabled_command[option + 1] == "True"


def test_independent_rng_streams_are_strictly_opt_in_and_receive_campaign_seed(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", [str(RUNNER)])
    default_args = runner.parse_args()
    default_command = runner.train_command(default_args)
    default_params = json.loads(runner.summarize_params(default_args))

    assert default_args.independent_rng_streams is False
    assert "--pipeline.training-seed" not in default_command
    assert "--pipeline.independent-rng-streams" not in default_command
    assert "--pipeline.model.training-seed" not in default_command
    assert "--pipeline.model.independent-rng-streams" not in default_command
    assert "independent_rng_streams" not in default_params

    monkeypatch.setattr(
        sys,
        "argv",
        [str(RUNNER), "--seed", "314", "--independent-rng-streams"],
    )
    enabled_args = runner.parse_args()
    enabled_command = runner.train_command(enabled_args)
    enabled_params = json.loads(runner.summarize_params(enabled_args))

    assert enabled_command[enabled_command.index("--pipeline.training-seed") + 1] == "314"
    assert enabled_command[enabled_command.index("--pipeline.independent-rng-streams") + 1] == "True"
    assert enabled_command[enabled_command.index("--pipeline.model.training-seed") + 1] == "314"
    assert enabled_command[enabled_command.index("--pipeline.model.independent-rng-streams") + 1] == "True"
    assert enabled_params["independent_rng_streams"] is True


def test_fas_consolidated_h2d_is_explicit_and_default_off(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", [str(RUNNER)])
    default_args = runner.parse_args()
    default_command = runner.train_command(default_args)
    option = default_command.index("--pipeline.datamanager.pixel-sampler.fas-consolidate-h2d")
    assert default_args.fas_consolidate_h2d is False
    assert default_command[option + 1] == "False"

    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--fas-consolidate-h2d"])
    enabled_args = runner.parse_args()
    enabled_command = runner.train_command(enabled_args)
    option = enabled_command.index("--pipeline.datamanager.pixel-sampler.fas-consolidate-h2d")
    assert enabled_command[option + 1] == "True"


def test_live_backend_switches_are_explicit_and_default_off(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", [str(RUNNER)])
    default_args = runner.parse_args()
    default_command = runner.train_command(default_args)
    assert default_args.tcnn_network_jit_switch_step is None
    assert default_args.tcnn_network_jit_second_switch_step is None
    assert default_args.tcnn_network_jit_second_switch_scope is None
    assert default_args.fused_adam_switch_step is None
    assert default_args.replay_eval_trajectory is False
    assert "--pipeline.tcnn-network-jit-switch-step" not in default_command
    assert "--pipeline.tcnn-network-jit-second-switch-step" not in default_command
    assert "--pipeline.tcnn-network-jit-second-switch-scope" not in default_command
    assert "--fused-adam-switch-step" not in default_command
    assert "--replay-eval-trajectory" not in default_command

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER),
            "--tcnn-network-jit-switch-step",
            "15189",
            "--tcnn-network-jit-scope",
            "geometry",
            "--tcnn-network-jit-second-switch-step",
            "30377",
            "--tcnn-network-jit-second-switch-scope",
            "color",
            "--fused-adam-switch-step",
            "15189",
        ],
    )
    switched_args = runner.parse_args()
    switched_command = runner.train_command(switched_args)
    jit_option = switched_command.index("--pipeline.tcnn-network-jit-switch-step")
    second_step_option = switched_command.index("--pipeline.tcnn-network-jit-second-switch-step")
    second_scope_option = switched_command.index("--pipeline.tcnn-network-jit-second-switch-scope")
    fused_option = switched_command.index("--fused-adam-switch-step")
    assert switched_command[jit_option + 1] == "15189"
    assert switched_command[second_step_option + 1] == "30377"
    assert switched_command[second_scope_option + 1] == "color"
    assert switched_command[fused_option + 1] == "15189"

    params = json.loads(runner.summarize_params(switched_args))
    assert params["tcnn_network_jit_second_switch_step"] == 30377
    assert params["tcnn_network_jit_second_switch_scope"] == "color"


@pytest.mark.parametrize(
    "options,match",
    [
        (["--tcnn-network-jit", "--tcnn-network-jit-switch-step", "5"], "mutually exclusive"),
        (["--fused-adam", "--fused-adam-switch-step", "5"], "mutually exclusive"),
    ],
)
def test_live_backend_switch_rejects_already_enabled_backend(monkeypatch, options, match) -> None:
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", [str(RUNNER), *options])
    with pytest.raises(ValueError, match=match):
        runner.train_command(runner.parse_args())


@pytest.mark.parametrize(
    "options,match",
    [
        (["--tcnn-network-jit-second-switch-step", "20"], "must be set together"),
        (["--tcnn-network-jit-second-switch-scope", "color"], "must be set together"),
        (
            [
                "--tcnn-network-jit-second-switch-step",
                "20",
                "--tcnn-network-jit-second-switch-scope",
                "color",
            ],
            "requires --tcnn-network-jit-switch-step",
        ),
        (
            [
                "--tcnn-network-jit-switch-step",
                "20",
                "--tcnn-network-jit-second-switch-step",
                "20",
                "--tcnn-network-jit-second-switch-scope",
                "color",
            ],
            "strictly greater",
        ),
        (
            [
                "--tcnn-network-jit-switch-step",
                "20",
                "--tcnn-network-jit-second-switch-step",
                "19",
                "--tcnn-network-jit-second-switch-scope",
                "color",
            ],
            "strictly greater",
        ),
    ],
)
def test_second_live_jit_switch_rejects_invalid_schedules(monkeypatch, options, match) -> None:
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", [str(RUNNER), *options])
    with pytest.raises(ValueError, match=match):
        runner.train_command(runner.parse_args())


def test_eval_trajectory_replay_is_explicit(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--replay-eval-trajectory"])
    args = runner.parse_args()
    command = runner.train_command(args)
    option = command.index("--replay-eval-trajectory")
    assert args.replay_eval_trajectory is True
    assert command[option + 1] == "True"


def test_grad_scaler_controls_are_explicit_and_default_preserving(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", [str(RUNNER)])
    default_args = runner.parse_args()
    default_command = runner.train_command(default_args)
    assert default_args.grad_scaler_init_scale is None
    assert default_args.grad_scaler_growth_interval is None
    assert "--grad-scaler-init-scale" not in default_command
    assert "--grad-scaler-growth-interval" not in default_command

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER),
            "--grad-scaler-init-scale",
            "8192",
            "--grad-scaler-growth-interval",
            "1000000",
        ],
    )
    args = runner.parse_args()
    command = runner.train_command(args)
    scale = command.index("--grad-scaler-init-scale")
    interval = command.index("--grad-scaler-growth-interval")
    assert command[scale + 1] == "8192.0"
    assert command[interval + 1] == "1000000"


def test_occupancy_diagnostics_boolean_flag_is_default_preserving_and_opt_out(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", [str(RUNNER)])
    default_args = runner.parse_args()
    default_command = runner.train_command(default_args)
    default_params = json.loads(runner.summarize_params(default_args))
    assert default_args.occupancy_diagnostics is None
    assert "--pipeline.model.occupancy-diagnostics" not in default_command
    assert "occupancy_diagnostics" not in default_params

    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--occupancy-diagnostics"])
    enabled_args = runner.parse_args()
    enabled_command = runner.train_command(enabled_args)
    enabled_params = json.loads(runner.summarize_params(enabled_args))
    assert enabled_args.occupancy_diagnostics is True
    assert "--pipeline.model.occupancy-diagnostics" not in enabled_command
    assert enabled_params["occupancy_diagnostics"] is True

    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--no-occupancy-diagnostics"])
    disabled_args = runner.parse_args()
    disabled_command = runner.train_command(disabled_args)
    disabled_params = json.loads(runner.summarize_params(disabled_args))
    option = disabled_command.index("--pipeline.model.occupancy-diagnostics")
    assert disabled_args.occupancy_diagnostics is False
    assert disabled_command[option + 1] == "False"
    assert disabled_params["occupancy_diagnostics"] is False


def test_no_argument_runner_resolves_accepted_stage_a_defaults(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", [str(RUNNER)])

    args = runner.parse_args()
    command = runner.train_command(args)
    params = json.loads(runner.summarize_params(args))

    assert args.data == Path("/home/brans/temporal_perframe_stride7_45f/007740")
    assert args.output_dir == Path("/home/brans/lookcloser_leader_repro_runs")
    assert args.seed == 42
    assert args.scene_scale == 1.5
    assert args.scale_factor == 1.0
    assert args.max_num_iterations == 75_941
    assert args.train_num_rays_per_batch == 4096
    assert args.eval_num_rays_per_chunk == 2048
    assert args.max_res == 8192.0
    assert args.adaptive_warmup_steps == 4096
    assert args.occupancy_warmup_steps == 4096
    assert args.occupancy_binary_warmup_steps == 4096
    assert args.stable_occupancy_reduction is True
    assert args.stop_on_no_improve is False
    assert args.prune_checkpoints is False
    assert params["stable_occupancy_reduction"] is True
    stable = command.index("--pipeline.model.stable-occupancy-reduction")
    assert command[stable + 1] == "True"
    assert "--fused-adam" not in command
    assert "--tcnn-network-jit" not in command
    assert "--independent-rng-streams" not in command
    assert "--cpu-fas-prefetch" not in command


@pytest.mark.parametrize(
    "options,match",
    [
        (["--grad-scaler-init-scale", "0"], "finite and positive"),
        (["--grad-scaler-init-scale", "nan"], "finite and positive"),
        (["--grad-scaler-growth-interval", "0"], "must be positive"),
    ],
)
def test_grad_scaler_controls_reject_invalid_values(monkeypatch, options, match) -> None:
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", [str(RUNNER), *options])
    with pytest.raises(ValueError, match=match):
        runner.train_command(runner.parse_args())
