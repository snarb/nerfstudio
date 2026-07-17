from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_static_leader_candidate as evaluator  # noqa: E402
import finalize_static_leader_campaign as finalizer  # noqa: E402
import record_static_leader_candidate as recorder  # noqa: E402
import run_static_leader_e2e as controller  # noqa: E402


def reviewed_staged_speed_argv() -> list[str]:
    return [
        "--tcnn-overlay",
        str(controller.DEFAULT_JIT_TCNN_OVERLAY),
        "--cache-train-rays",
        "--fused-adam-switch-step",
        "15189",
        "--tcnn-network-jit-switch-step",
        "15189",
        "--tcnn-network-jit-scope",
        "color",
        "--tcnn-network-jit-second-switch-step",
        "30377",
        "--tcnn-network-jit-second-switch-scope",
        "geometry",
        "--replay-eval-trajectory",
        "--historical-stage-boundary-rng-reset",
        "--speed-final-step",
        "91128",
    ]


def test_frozen_values_fail_closed_on_missing_or_mismatch() -> None:
    expected = {"source": "abc", "runtime": "2.7.1"}
    controller.require_frozen_values("test", dict(expected), expected)

    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        controller.require_frozen_values("test", {"source": "abc"}, expected)
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        controller.require_frozen_values(
            "test", {"source": "changed", "runtime": "2.7.1"}, expected
        )


def test_controller_protocol_bundle_matches_frozen_fingerprint() -> None:
    fingerprint, sources = controller.controller_protocol_fingerprint()
    assert fingerprint == controller.EXPECTED_CONTROLLER_PROTOCOL_FINGERPRINT
    assert set(sources) == {
        "controller",
        "speed_controller",
        "candidate_evaluator",
        "candidate_recorder",
        "retry_finalizer",
        "dataset_provenance",
        "detail_scorer",
        "detail_reference",
        "checkpoint_fork",
    }


def test_speed_source_fingerprint_binds_named_branch_commit_and_files() -> None:
    sources = {"a.py": "a" * 64, "b.py": "b" * 64}
    fingerprint = controller.committed_speed_source_fingerprint(
        controller.EXPECTED_SPEED_COMMIT,
        controller.EXPECTED_SPEED_BRANCH,
        sources,
    )
    assert len(fingerprint) == 64
    assert fingerprint != controller.committed_speed_source_fingerprint(
        controller.EXPECTED_SPEED_COMMIT, "another_branch", sources
    )
    assert fingerprint != controller.committed_speed_source_fingerprint(
        "0" * 40, controller.EXPECTED_SPEED_BRANCH, sources
    )
    assert fingerprint != controller.committed_speed_source_fingerprint(
        controller.EXPECTED_SPEED_COMMIT,
        controller.EXPECTED_SPEED_BRANCH,
        {**sources, "b.py": "c" * 64},
    )


def test_candidate_recorder_reuses_hash_only_for_unchanged_exact_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "step-000091128.ckpt"
    checkpoint.write_bytes(b"trusted checkpoint")
    recorded_hash = controller.sha256_file(checkpoint)
    manifest = {
        "stage_a": {"checkpoint": str(tmp_path / "other.ckpt")},
        "stage_a_fw03": {
            "checkpoint": str(checkpoint),
            "target_step": 91_128,
            "returncode": 0,
            "checkpoint_sha256": recorded_hash,
            "checkpoint_file_identity": controller.checkpoint_file_identity(checkpoint),
        },
    }

    def unexpected_rehash(_path: Path) -> str:
        raise AssertionError("unchanged exact checkpoint must not be rehashed")

    monkeypatch.setattr(recorder, "sha256_file", unexpected_rehash)
    assert recorder.campaign_checkpoint_sha256(manifest, checkpoint) == recorded_hash

    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(RuntimeError, match="changed after the campaign hash"):
        recorder.campaign_checkpoint_sha256(manifest, checkpoint)


def test_candidate_recorder_falls_back_for_legacy_or_intermediate_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "step-000075940.ckpt"
    checkpoint.write_bytes(b"legacy checkpoint")
    calls: list[Path] = []

    def fallback_hash(path: Path) -> str:
        calls.append(path)
        return "a" * 64

    monkeypatch.setattr(recorder, "sha256_file", fallback_hash)
    manifest = {"stage_a": {"checkpoint": str(tmp_path / "stage-final.ckpt")}}
    assert recorder.campaign_checkpoint_sha256(manifest, checkpoint) == "a" * 64
    assert calls == [checkpoint]


@pytest.mark.parametrize(
    "stage",
    [
        {"checkpoint_sha256": "not-a-sha", "checkpoint_file_identity": {}},
        {"checkpoint_sha256": "a" * 64},
    ],
)
def test_candidate_recorder_rejects_invalid_or_partial_recorded_identity(
    tmp_path: Path, stage: dict
) -> None:
    checkpoint = tmp_path / "step-000091128.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    stage.update(checkpoint=str(checkpoint), target_step=91_128, returncode=0)
    with pytest.raises(RuntimeError, match="invalid checkpoint hash|incomplete checkpoint identity"):
        recorder.campaign_checkpoint_sha256({"stage_a_fw03": stage}, checkpoint)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"returncode": 1}, "did not complete"),
        ({"returncode": 0, "target_step": 75_940}, "target step"),
    ],
)
def test_candidate_recorder_requires_completed_matching_stage(
    tmp_path: Path, update: dict, message: str
) -> None:
    checkpoint = tmp_path / "step-000091128.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    stage = {
        "checkpoint": str(checkpoint),
        "target_step": 91_128,
        "returncode": 0,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_file_identity": controller.checkpoint_file_identity(checkpoint),
        **update,
    }
    with pytest.raises(RuntimeError, match=message):
        recorder.campaign_checkpoint_sha256({"stage_a_fw03": stage}, checkpoint)


def test_candidate_recorder_rejects_ambiguous_stage_match(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-000091128.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    stage = {"checkpoint": str(checkpoint)}
    with pytest.raises(RuntimeError, match="ambiguously matches"):
        recorder.campaign_checkpoint_sha256(
            {"stage_a": dict(stage), "stage_a_fw03": dict(stage)}, checkpoint
        )


def test_default_recipe_remains_exact_accepted_s1() -> None:
    args = controller.parse_args([])
    recipe = controller.resolve_recipe(args)

    assert args.historical_worktree == controller.DEFAULT_ACCEPTED_WORKTREE
    assert args.automatic_finalization is True
    assert controller.accepted_stable_fp16_mode(args)
    assert not recipe.speed_mode
    assert recipe.train_rays_per_batch == 4_096
    assert recipe.adaptive_warmup_steps == 4_096
    assert recipe.checkpoint_interval == 15_188
    assert recipe.save_interval == 15_188
    assert recipe.parent_step == 75_940
    assert recipe.final_step == 106_316
    assert recipe.scheduler_max_steps == 200_000
    assert recipe.fields_lr == 0.01
    assert recipe.fields_lr_final == 0.0001
    assert args.eval_num_rays_per_chunk == 2_048
    assert recipe.cache_train_rays is False
    assert recipe.cpu_fas_prefetch is False
    assert recipe.fused_adam is False
    assert recipe.fused_adam_switch_step is None
    assert recipe.tcnn_network_jit_switch_step is None
    assert recipe.tcnn_network_jit_scope is None
    assert recipe.tcnn_network_jit_second_switch_step is None
    assert recipe.tcnn_network_jit_second_switch_scope is None
    assert recipe.feature_reweighting_switch_step is None
    assert recipe.feature_reweighting_after_switch is None
    assert recipe.replay_eval_trajectory is False
    assert recipe.historical_stage_boundary_rng_reset is False
    assert recipe.hard_candidate_only is False
    assert recipe.wall_milestone_seconds is None
    command = controller.common_runner_args(args, recipe, 42, "test", "timestamp")
    assert "--cache-train-rays" not in command
    assert "--cpu-fas-prefetch" not in command
    assert "--tcnn-network-jit-scope" not in command
    assert "--tcnn-network-jit-second-switch-step" not in command
    assert "--tcnn-network-jit-second-switch-scope" not in command


def test_fused_adam_is_explicit_speed_recipe() -> None:
    args = controller.parse_args(["--fused-adam"])
    recipe = controller.resolve_recipe(args)

    assert recipe.speed_mode is True
    assert controller.speed_mode(args) is True
    assert controller.accepted_stable_fp16_mode(args) is False
    assert recipe.batch_scale == 1
    assert recipe.final_step == controller.ACCEPTED_STEP
    assert recipe.fused_adam is True
    command = controller.common_runner_args(args, recipe, 42, "test", "timestamp")
    assert "--fused-adam" in command


def test_reviewed_staged_speed_recipe_is_exact_and_fully_forwarded() -> None:
    args = controller.parse_args(reviewed_staged_speed_argv())
    recipe = controller.resolve_recipe(args)

    assert recipe.speed_mode is True
    assert controller.speed_mode(args) is True
    assert recipe.cache_train_rays is True
    assert recipe.cpu_fas_prefetch is False
    assert recipe.fused_adam is False
    assert recipe.fused_adam_switch_step == 15_189
    assert recipe.tcnn_network_jit_switch_step == 15_189
    assert recipe.tcnn_network_jit_scope == "color"
    assert recipe.tcnn_network_jit_second_switch_step == 30_377
    assert recipe.tcnn_network_jit_second_switch_scope == "geometry"
    assert recipe.final_step == 91_128
    assert recipe.checkpoint_interval == 15_188
    assert recipe.save_interval == 91_129
    assert recipe.replay_eval_trajectory is True
    assert recipe.historical_stage_boundary_rng_reset is True
    assert recipe.hard_candidate_only is True
    assert recipe.wall_milestone_seconds == 3_600

    command = controller.common_runner_args(args, recipe, 42, "test", "timestamp")
    assert "--cache-train-rays" in command
    assert "--cpu-fas-prefetch" not in command
    assert "--replay-eval-trajectory" in command
    fused_option = command.index("--fused-adam-switch-step")
    jit_option = command.index("--tcnn-network-jit-switch-step")
    jit_scope_option = command.index("--tcnn-network-jit-scope")
    second_jit_option = command.index("--tcnn-network-jit-second-switch-step")
    second_jit_scope_option = command.index("--tcnn-network-jit-second-switch-scope")
    assert command[fused_option + 1] == "15189"
    assert command[jit_option + 1] == "15189"
    assert command[jit_scope_option + 1] == "color"
    assert command[second_jit_option + 1] == "30377"
    assert command[second_jit_scope_option + 1] == "geometry"
    eval_option = command.index("--step-interval")
    save_option = command.index("--save-interval")
    assert command[eval_option + 1] == "15188"
    assert command[save_option + 1] == "91129"


def test_reviewed_staged_speed_recipe_keeps_eval_chunk_caller_selected() -> None:
    default_args = controller.parse_args(reviewed_staged_speed_argv())
    default_recipe = controller.resolve_recipe(default_args)
    default_command = controller.common_runner_args(
        default_args, default_recipe, 42, "test", "timestamp"
    )
    default_chunk = default_command.index("--eval-num-rays-per-chunk")
    assert default_command[default_chunk + 1] == "2048"

    campaign_args = controller.parse_args(
        [*reviewed_staged_speed_argv(), "--eval-num-rays-per-chunk", "8192"]
    )
    campaign_recipe = controller.resolve_recipe(campaign_args)
    campaign_command = controller.common_runner_args(
        campaign_args, campaign_recipe, 42, "test", "timestamp"
    )
    campaign_chunk = campaign_command.index("--eval-num-rays-per-chunk")
    assert campaign_command[campaign_chunk + 1] == "8192"


def test_cpu_fas_prefetch_is_default_off_and_only_extends_exact_staged_recipe() -> None:
    args = controller.parse_args([*reviewed_staged_speed_argv(), "--cpu-fas-prefetch"])
    recipe = controller.resolve_recipe(args)

    assert recipe.speed_mode is True
    assert recipe.cache_train_rays is True
    assert recipe.cpu_fas_prefetch is True
    command = controller.common_runner_args(args, recipe, 42, "test", "timestamp")
    assert "--cache-train-rays" in command
    assert "--cpu-fas-prefetch" in command

    with pytest.raises(ValueError, match="complete reviewed staged speed recipe"):
        controller.resolve_recipe(controller.parse_args(["--cpu-fas-prefetch"]))


def test_staged_speed_switch_pairs_fail_closed() -> None:
    with pytest.raises(ValueError, match="requires both"):
        controller.resolve_recipe(controller.parse_args(["--fused-adam-switch-step", "15189"]))
    with pytest.raises(ValueError, match="jit-switch-step and --tcnn-network-jit-scope"):
        controller.resolve_recipe(
            controller.parse_args(
                [
                    "--tcnn-network-jit-switch-step",
                    "15189",
                ]
            )
        )
    with pytest.raises(ValueError, match="second-switch-step and .*second-switch-scope"):
        controller.resolve_recipe(
            controller.parse_args(
                [
                    "--tcnn-network-jit-second-switch-step",
                    "30377",
                ]
            )
        )


def test_staged_speed_recipe_rejects_incomplete_or_unreviewed_combinations() -> None:
    without_cache = [value for value in reviewed_staged_speed_argv() if value != "--cache-train-rays"]
    with pytest.raises(ValueError, match="complete reviewed staged speed recipe"):
        controller.resolve_recipe(controller.parse_args(without_cache))

    wrong_scope = reviewed_staged_speed_argv()
    wrong_scope[wrong_scope.index("geometry")] = "both"
    with pytest.raises(ValueError, match="complete reviewed staged speed recipe"):
        controller.resolve_recipe(controller.parse_args(wrong_scope))

    with pytest.raises(ValueError, match="complete reviewed staged speed recipe"):
        controller.resolve_recipe(controller.parse_args(["--replay-eval-trajectory"]))

    with pytest.raises(ValueError, match="not part of the reviewed staged speed recipe"):
        controller.resolve_recipe(
            controller.parse_args(
                [
                    *reviewed_staged_speed_argv(),
                    "--feature-reweighting-switch-step",
                    "64813",
                    "--feature-reweighting-after-switch",
                    "0.3",
                ]
            )
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["--cache-train-rays"],
        ["--cpu-fas-prefetch"],
        ["--tcnn-network-jit-scope", "color"],
        ["--tcnn-network-jit-second-switch-step", "30377"],
        ["--tcnn-network-jit-second-switch-scope", "geometry"],
    ],
)
def test_new_staged_options_are_detected_as_speed_mode(argv: list[str]) -> None:
    assert controller.speed_mode(controller.parse_args(argv)) is True


def test_stage_boundary_rng_reset_provenance_allows_only_rng_removal() -> None:
    before = {
        "trainer_step": 75940,
        "adam_steps": [75905],
        "optimizer_lrs": [0.0017],
        "rng_state_present": True,
        "rng_state": {"torch_cpu_sha256": "cpu", "torch_cuda_sha256": ["cuda"]},
    }
    after = dict(before, rng_state_present=False, rng_state=None)
    provenance = {
        "before": before,
        "after": after,
        "source_sha256": "source",
        "output_sha256": "output",
        "actions": {
            "lr_multiplier": 1.0,
            "scheduler_time_scale": 1.0,
            "reset_adam": False,
            "restart_scheduler": False,
            "reset_scaler": False,
            "reset_torch_cpu_rng_seed": None,
            "drop_rng_state": True,
        },
    }
    assert controller.validate_stage_boundary_rng_reset_provenance(
        provenance, source_sha256="source", output_sha256="output"
    ) == (before, after)

    provenance["after"] = dict(after, adam_steps=[])
    with pytest.raises(RuntimeError, match="beyond removing"):
        controller.validate_stage_boundary_rng_reset_provenance(
            provenance, source_sha256="source", output_sha256="output"
        )


def test_unfinalized_and_infrastructure_outcomes_never_exit_zero() -> None:
    args = controller.parse_args(["--no-automatic-finalization"])
    assert args.automatic_finalization is False
    assert controller.controller_exit_code(True, "complete") == 0
    assert (
        controller.controller_exit_code(False, "complete_no_accepted_candidate")
        == controller.QUALITY_FAILURE_EXIT_CODE
    )
    assert (
        controller.controller_exit_code(False, "complete_unfinalized")
        == controller.INCOMPLETE_OR_INFRASTRUCTURE_EXIT_CODE
    )
    assert (
        controller.controller_exit_code(False, "finalization_failed")
        == controller.INCOMPLETE_OR_INFRASTRUCTURE_EXIT_CODE
    )


def test_candidate_summary_distinguishes_quality_failure_from_infrastructure(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "numeric_pass": False,
                "automatic_pass": False,
                "automatic_gate_complete": True,
                "detail_pass": False,
                "detail_gate_complete": True,
                "quality_pass": False,
            }
        ),
        encoding="utf-8",
    )
    summary = controller.load_completed_candidate_summary(summary_path, 0, 42)
    assert summary["automatic_pass"] is False

    with pytest.raises(controller.CandidateFinalizationError, match="exited 7"):
        controller.load_completed_candidate_summary(summary_path, 7, 42)
    with pytest.raises(controller.CandidateFinalizationError, match="wrote no summary"):
        controller.load_completed_candidate_summary(tmp_path / "missing.json", 0, 42)

    summary_path.write_text(
        json.dumps({"numeric_pass": True, "automatic_pass": True}), encoding="utf-8"
    )
    with pytest.raises(controller.CandidateFinalizationError, match="missing"):
        controller.load_completed_candidate_summary(summary_path, 0, 42)

    summary_path.write_text(
        json.dumps(
            {
                "numeric_pass": True,
                "automatic_pass": True,
                "automatic_gate_complete": False,
                "detail_pass": True,
                "detail_gate_complete": True,
                "quality_pass": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(controller.CandidateFinalizationError, match="gate incomplete"):
        controller.load_completed_candidate_summary(summary_path, 0, 42)

    summary_path.write_text(
        json.dumps(
            {
                "numeric_pass": True,
                "automatic_pass": True,
                "automatic_gate_complete": True,
                "detail_pass": False,
                "detail_gate_complete": True,
                "quality_pass": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(controller.CandidateFinalizationError, match="inconsistent quality"):
        controller.load_completed_candidate_summary(summary_path, 0, 42)


def complete_artifact_payload() -> dict:
    return {
        "status": "complete",
        "artifact_views_scored": 3,
        "artifact_views_requested": 3,
        "artifact_count": 0,
        "roi": {
            "status": "complete",
            "returncode": 0,
            "roi_count": 10,
            "roi_serious_count": 0,
        },
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.pop("roi"), "ROI result is missing"),
        (lambda data: data["roi"].update(roi_count=9), "exactly 10 ROI"),
        (lambda data: data["roi"].update(returncode=1), "did not exit successfully"),
        (lambda data: data.update(artifact_views_scored=2), "3 scored artifact views"),
    ],
)
def test_candidate_evaluator_artifact_protocol_is_fail_closed(mutation, message: str) -> None:
    artifact = complete_artifact_payload()
    mutation(artifact)
    assert any(message in error for error in evaluator.artifact_infrastructure_errors(artifact))


def test_candidate_evaluator_accepts_only_complete_three_view_ten_roi_protocol() -> None:
    assert evaluator.artifact_infrastructure_errors(complete_artifact_payload()) == []


def test_candidate_evaluator_default_and_tagged_output_paths(tmp_path: Path) -> None:
    default_paths = evaluator.candidate_output_paths(tmp_path, 91_128, "candidate")
    assert default_paths == {
        "eval_json": tmp_path / "eval_candidate_step-000091128.json",
        "render_dir": tmp_path / "renders_candidate_step-000091128",
        "eval_log": tmp_path / "eval_candidate_step-000091128_stdout.log",
        "detail_dir": tmp_path / "detail_candidate_step-000091128",
        "summary": tmp_path / "candidate_evaluation_step-000091128.json",
    }

    tagged_paths = evaluator.candidate_output_paths(tmp_path, 91_128, "candidate_pruned")
    assert tagged_paths == {
        "eval_json": tmp_path / "eval_candidate_pruned_step-000091128.json",
        "render_dir": tmp_path / "renders_candidate_pruned_step-000091128",
        "eval_log": tmp_path / "eval_candidate_pruned_step-000091128_stdout.log",
        "detail_dir": tmp_path / "detail_candidate_pruned_step-000091128",
        "summary": tmp_path / "candidate_pruned_evaluation_step-000091128.json",
    }


@pytest.mark.parametrize("tag", ["", "../candidate", "candidate.json", "a/b", "x" * 65])
def test_candidate_evaluator_rejects_unsafe_output_tag(tag: str) -> None:
    with pytest.raises(evaluator.argparse.ArgumentTypeError):
        evaluator.output_tag(tag)


def test_detail_scorer_failure_cannot_reuse_stale_or_missing_output(tmp_path: Path) -> None:
    detail_json = tmp_path / "detail.json"
    detail_json.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exited 1"):
        evaluator.require_completed_detail_result(1, detail_json)

    detail_json.unlink()
    with pytest.raises(RuntimeError, match="did not produce"):
        evaluator.require_completed_detail_result(0, detail_json)


def test_detail_quality_fail_is_complete_and_uses_frozen_micro_crops(tmp_path: Path) -> None:
    detail_json = tmp_path / "detail.json"
    rows = [
        {"crop": crop, "pass": crop != "stand_eval0"}
        for crop in sorted(evaluator.FROZEN_DETAIL_CROPS)
    ]
    payload = {"reference_comparison": {"pass": False, "rois": rows}}
    detail_json.write_text(json.dumps(payload), encoding="utf-8")

    completed = evaluator.require_completed_detail_result(2, detail_json)
    assert evaluator.required_detail_gate(completed) is True

    rows[0]["pass"] = False
    detail_json.write_text(json.dumps(payload), encoding="utf-8")
    assert evaluator.required_detail_gate(evaluator.require_completed_detail_result(2, detail_json)) is False


def write_artifact_fixture(artifact_dir: Path, roi_count: int) -> None:
    artifact_dir.mkdir(parents=True)
    for index in range(3):
        (artifact_dir / f"eval_img_{index:04d}_artifact_stdout.log").write_text(
            "[candidate] serious=False artifact_score=0.000 count=0 largest=0px\n",
            encoding="utf-8",
        )
    roi_dir = artifact_dir / "roi_scores"
    roi_dir.mkdir()
    (roi_dir / "roi_artifact_scores.json").write_text(
        json.dumps([{"serious": False} for _ in range(roi_count)]), encoding="utf-8"
    )


def test_retry_finalizer_requires_exact_roi_count(tmp_path: Path) -> None:
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(
        json.dumps(
            {
                "checkpoint": "/tmp/step.ckpt",
                "results": {"psnr": 30.0, "ssim": 0.7, "lpips": 0.2},
            }
        ),
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"
    write_artifact_fixture(artifact_dir, roi_count=10)

    completed = finalizer.finalized_eval(eval_json, tmp_path / "renders", artifact_dir)
    assert completed["artifacts"]["gate_complete"] is True
    assert completed["automatic_pass"] is True

    roi_json = artifact_dir / "roi_scores" / "roi_artifact_scores.json"
    roi_json.write_text(json.dumps([{"serious": False}] * 9), encoding="utf-8")
    incomplete = finalizer.finalized_eval(eval_json, tmp_path / "renders", artifact_dir)
    assert incomplete["artifacts"]["gate_complete"] is False
    assert incomplete["automatic_pass"] is False


def test_retry_finalizer_cannot_bypass_priority_detail_gate(tmp_path: Path) -> None:
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(
        json.dumps(
            {
                "checkpoint": "/tmp/step.ckpt",
                "results": {"psnr": 30.0, "ssim": 0.7, "lpips": 0.2},
            }
        ),
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"
    write_artifact_fixture(artifact_dir, roi_count=10)

    with pytest.raises(RuntimeError, match="micro-detail"):
        finalizer.finalized_candidate(eval_json, tmp_path / "renders", artifact_dir, None)

    render_dir = tmp_path / "renders"
    rows = [{"crop": crop, "pass": True} for crop in sorted(evaluator.FROZEN_DETAIL_CROPS)]
    detail_json = tmp_path / "detail.json"
    detail_json.write_text(
        json.dumps(
            {
                "render_dir": str(render_dir),
                "reference_comparison": {"pass": True, "rois": rows},
            }
        ),
        encoding="utf-8",
    )
    accepted = finalizer.finalized_candidate(
        eval_json, render_dir, artifact_dir, detail_json
    )
    assert accepted["detail_pass"] is True
    assert accepted["quality_pass"] is True

    rows[0]["pass"] = False
    detail_json.write_text(
        json.dumps(
            {
                "render_dir": str(render_dir),
                "reference_comparison": {"pass": False, "rois": rows},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="must pass"):
        finalizer.finalized_candidate(
            eval_json, render_dir, artifact_dir, detail_json
        )


def test_retry_detail_result_is_bound_to_render_and_full_five_crop_protocol(tmp_path: Path) -> None:
    detail_json = tmp_path / "detail.json"
    render_dir = tmp_path / "renders"
    rows = [{"crop": crop, "pass": True} for crop in sorted(evaluator.FROZEN_DETAIL_CROPS)]
    payload = {
        "render_dir": str(render_dir),
        "reference_comparison": {"pass": True, "rois": rows},
    }
    detail_json.write_text(json.dumps(payload), encoding="utf-8")
    evaluator.require_completed_detail_result(0, detail_json, expected_render_dir=render_dir)

    with pytest.raises(RuntimeError, match="different render directory"):
        evaluator.require_completed_detail_result(
            0,
            detail_json,
            expected_render_dir=tmp_path / "other_renders",
        )

    payload["reference_comparison"]["rois"] = rows[:-1]
    detail_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="crop protocol mismatch"):
        evaluator.require_completed_detail_result(0, detail_json, expected_render_dir=render_dir)


def test_retry_protocol_migration_is_explicit_and_never_overrides_mismatch() -> None:
    fingerprint, sources = controller.controller_protocol_fingerprint()

    legacy: dict = {}
    with pytest.raises(RuntimeError, match="allow-legacy-protocol-migration"):
        finalizer.bind_or_require_campaign_protocol(legacy, False)
    finalizer.bind_or_require_campaign_protocol(legacy, True)
    assert legacy["controller_protocol_fingerprint"] == fingerprint
    assert legacy["controller_protocol_source_sha256"] == sources
    assert legacy["protocol_migration"]["from"] is None

    finalizer.bind_or_require_campaign_protocol(
        {"controller_protocol_fingerprint": fingerprint},
        False,
    )
    with pytest.raises(RuntimeError, match="differs from the current frozen protocol"):
        finalizer.bind_or_require_campaign_protocol(
            {"controller_protocol_fingerprint": "stale"},
            True,
        )
