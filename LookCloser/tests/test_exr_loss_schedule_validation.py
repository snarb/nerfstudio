from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_exr_loss_schedule_validation as campaign  # noqa: E402


def args(tmp_path: Path, *extra: str):
    return campaign.parse_args(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--data",
            str(tmp_path / "data"),
            *extra,
        ]
    )


def test_campaign_is_frozen_to_two_new_seeds(tmp_path: Path) -> None:
    parsed = args(tmp_path)
    assert parsed.seeds == [43, 44]
    with pytest.raises(SystemExit):
        args(tmp_path, "--seeds", "42", "43")


def test_scratch_prefixes_never_load_a_checkpoint(tmp_path: Path) -> None:
    parsed = args(tmp_path)
    for seed in parsed.seeds:
        for spec in campaign.prefix_specs(seed):
            command = campaign.train_command(parsed, spec)
            assert "--load-checkpoint" not in command
            assert command[command.index("--seed") + 1] == str(seed)


def test_scratch_pqmse_requires_two_bad_eval_boundaries_before_rejection(tmp_path: Path) -> None:
    parsed = args(tmp_path)
    spec = next(item for item in campaign.prefix_specs(43) if item.recipe == "pqmse")
    command = campaign.train_command(parsed, spec)
    assert command[command.index("--early-reject-after-evals") + 1] == "2"
    assert command[command.index("--early-reject-psnr-below") + 1] == "30.0"


def test_rejected_eval_rows_requires_two_consecutive_material_failures(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text(
        "step,eval_all_psnr,eval_all_ssim,eval_all_lpips\n"
        "100,29,0.79,0.51\n"
        "200,31,0.81,0.49\n",
        encoding="utf-8",
    )
    assert campaign.rejected_eval_rows(path) == []
    with path.open("a", encoding="utf-8") as stream:
        stream.write("300,28,0.78,0.55\n400,27,0.77,0.56\n")
    evidence = campaign.rejected_eval_rows(path)
    assert [row["step"] for row in evidence] == [100.0, 200.0, 300.0, 400.0]


def test_relative_rejection_requires_three_matched_exposure_regressions(tmp_path: Path) -> None:
    header = "step,cumulative_point_samples,eval_all_psnr,eval_all_ssim,eval_all_lpips\n"
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    reference.write_text(
        header
        + "100,1e9,,,\n100,,33,0.88,0.30\n"
        + "200,2e9,,,\n200,,34,0.89,0.25\n"
        + "300,3e9,,,\n300,,35,0.90,0.20\n",
        encoding="utf-8",
    )
    candidate.write_text(
        header
        + "10,1e9,,,\n10,,32.5,0.87,0.31\n"
        + "20,2e9,,,\n20,,33.5,0.88,0.26\n"
        + "30,3e9,,,\n30,,34.5,0.89,0.21\n",
        encoding="utf-8",
    )
    evidence = campaign.comparatively_rejected_eval_rows(candidate, reference)
    assert len(evidence) == 3
    assert evidence[-1]["psnr_delta"] == pytest.approx(-0.5)
    assert evidence[-1]["reference_step"] == 300.0


def test_forks_resume_only_their_declared_same_seed_parent(tmp_path: Path) -> None:
    parsed = args(tmp_path)
    steps = {spec.name: spec.max_step for spec in campaign.prefix_specs(43)}
    for spec in campaign.fork_specs(43, steps):
        assert spec.parent is not None and spec.parent.startswith("s43_")
        parent = tmp_path / spec.parent / "step-000000001.ckpt"
        command = campaign.train_command(parsed, spec, parent)
        assert command[command.index("--load-checkpoint") + 1] == str(parent)
        assert command[command.index("--checkpoint-load-mode") + 1] == "resume"


def test_matrix_contains_required_scratch_and_mature_controls() -> None:
    steps = {spec.name: spec.max_step for spec in campaign.prefix_specs(43)}
    first = campaign.fork_specs(43, steps)
    steps.update({spec.name: spec.max_step for spec in first})
    second = campaign.second_stage_specs(43, steps)
    names = {spec.name for spec in first + second}
    assert {
        "s43_eag_continue",
        "s43_direct_pql1",
        "s43_direct_pqmse",
        "s43_pure_pql1",
        "s43_pure_pqmse",
        "s43_scratch_lpips_continue",
        "s43_scratch_lpips_to_pql1",
        "s43_scratch_lpips_to_pqmse",
        "s43_mature_lpips_to_pql1",
        "s43_mature_lpips_to_pqmse",
    } <= names


def test_rejected_scratch_parents_are_omitted_without_blocking_staged_forks() -> None:
    steps = {
        spec.name: spec.max_step
        for spec in campaign.prefix_specs(43)
        if spec.recipe == "eag"
    }
    names = {spec.name for spec in campaign.fork_specs(43, steps)}
    assert "s43_eag_continue" in names
    assert "s43_mature_lpips_to_b1" in names
    assert "s43_pure_pqmse" not in names
    assert "s43_pure_pql1" not in names
    assert "s43_scratch_lpips_continue" not in names


def test_lpips_recipe_uses_true_64_square_patches(tmp_path: Path) -> None:
    parsed = args(tmp_path)
    spec = next(item for item in campaign.prefix_specs(43) if item.recipe == "lpips")
    command = campaign.train_command(parsed, spec)
    assert command[command.index("--training-patch-size") + 1] == "64"
    assert command[command.index("--train-num-rays-per-batch") + 1] == "16384"
    assert command[command.index("--eag-lpips-weight") + 1] == "0.02"


def test_iteration_caps_can_reach_frozen_exposures() -> None:
    # Measured standard batches render about 1.022e6 points/step; LPIPS batches
    # render about 4.19e6. Keep explicit headroom so a run cannot end before
    # the exposure controller gets a chance to stop it.
    assert campaign.STANDARD_PREFIX_MAX_STEP * 1.0e6 > campaign.EXPOSURE_BASE
    assert campaign.LPIPS_PREFIX_MAX_STEP * 4.0e6 > campaign.EXPOSURE_LPIPS_END
    assert campaign.TAIL_MAX_EXTRA_STEPS * 1.0e6 > (
        campaign.EXPOSURE_FINAL - campaign.EXPOSURE_BASE
    )
    assert campaign.STANDARD_PREFIX_SAVE_INTERVAL == campaign.PREFIX_EVAL_INTERVAL
    assert campaign.LPIPS_PREFIX_SAVE_INTERVAL == campaign.LPIPS_PREFIX_EVAL_INTERVAL


def test_runtime_environment_exposes_venv_tools(tmp_path: Path) -> None:
    parsed = args(tmp_path)
    environment = campaign.runtime_environment(parsed)
    assert environment["PATH"].split(":")[0] in {
        str(parsed.venv / "bin"),
        "/usr/local/cuda-12.6/bin",
    }
    assert str(parsed.venv / "bin") in environment["PATH"].split(":")


def test_exposure_checkpoint_selector_is_fail_closed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    models = run / "nerfstudio_models"
    models.mkdir(parents=True)
    (models / "step-000000100.ckpt").touch()
    (models / "step-000000200.ckpt").touch()
    (run / "metrics_compact.csv").write_text(
        "step,cumulative_point_samples\n100,2.378e11\n200,2.410e11\n",
        encoding="utf-8",
    )
    checkpoint, exposure, error = campaign.choose_exposure_checkpoint(run, campaign.EXPOSURE_BASE)
    assert checkpoint.name == "step-000000100.ckpt"
    assert exposure == 2.378e11
    assert error < campaign.EXPOSURE_REL_TOLERANCE
    with pytest.raises(RuntimeError):
        campaign.choose_exposure_checkpoint(run, 1.0e11)


def strategy(
    name: str,
    psnr: float,
    ssim: float,
    lpips: float,
    seconds: float,
    phases: int,
    patches: bool,
) -> dict:
    return {
        "strategy": name,
        "mean_psnr": psnr,
        "mean_ssim": ssim,
        "mean_lpips": lpips,
        "median_train_seconds": seconds,
        "loss_phases": phases,
        "requires_lpips_patches": patches,
        "peak_vram_mb": 1000,
        "cable_gaps": 0,
        "visual_failure": False,
        "equivalence_bands": campaign.QUALITY_FLOORS,
    }


def test_selector_prefers_fast_simple_arm_inside_quality_equivalence() -> None:
    rows = [
        strategy("staged", 34.35, 0.8990, 0.1980, 5000, 3, True),
        strategy("pure_pqmse", 34.32, 0.8988, 0.1990, 3000, 1, False),
    ]
    selected = campaign.choose_strategy(rows)
    assert selected["quality_winner"] == "staged"
    assert selected["selected"] == "pure_pqmse"


def test_selector_keeps_metric_winner_outside_equivalence() -> None:
    rows = [
        strategy("staged", 34.35, 0.8990, 0.1980, 5000, 3, True),
        strategy("pure_pqmse", 34.20, 0.8970, 0.2100, 3000, 1, False),
    ]
    assert campaign.choose_strategy(rows)["selected"] == "staged"


def test_cable_gap_is_a_hard_veto() -> None:
    bad = strategy("bad", 35.0, 0.91, 0.18, 1000, 1, False)
    bad["cable_gaps"] = 1
    good = strategy("good", 34.0, 0.89, 0.21, 5000, 3, True)
    assert campaign.choose_strategy([bad, good])["selected"] == "good"


def test_paired_variance_bands_have_project_floors() -> None:
    bands = campaign.quality_equivalence_bands(
        {"psnr": [0.01, 0.02], "ssim": [0.0, 0.0], "lpips": [0.001, 0.0015]}
    )
    assert bands == campaign.QUALITY_FLOORS
