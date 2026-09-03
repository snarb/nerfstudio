from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "render_plane_homography.py"
SPEC = importlib.util.spec_from_file_location("render_plane_homography_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_identity_target_plane_projects_to_identity_camera() -> None:
    c2w = torch.eye(4)[:3]
    intrinsics = {"fx": 10.0, "fy": 12.0, "cx": 1.5, "cy": 1.0, "width": 4, "height": 3}
    world, valid, _ = MODULE.target_plane_world_points(
        height=3,
        width=4,
        target_c2w=c2w,
        target_intrinsics=intrinsics,
        plane_point=torch.tensor([0.0, 0.0, -2.0]),
        plane_normal=torch.tensor([0.0, 0.0, 1.0]),
    )
    u, v, depth = MODULE.project_world_to_camera(world, c2w, intrinsics)
    yy, xx = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")

    assert bool(valid.all())
    torch.testing.assert_close(u, xx.float())
    torch.testing.assert_close(v, yy.float())
    torch.testing.assert_close(depth, torch.full((3, 4), 2.0))


def test_homography_nearest_fill_never_averages() -> None:
    first = torch.full((3, 2, 2), 0.2)
    second = torch.full((3, 2, 2), 0.8)
    first_valid = torch.tensor([[True, False], [True, False]])
    second_valid = torch.tensor([[True, True], [False, False]])

    rgb, valid, selected = MODULE.aggregate_homographies(
        [first, second], [first_valid, second_valid], mode="nearest-fill"
    )

    assert selected.tolist() == [[0, 1], [0, -1]]
    assert valid.tolist() == [[True, True], [True, False]]
    torch.testing.assert_close(rgb[:, 0, 0], first[:, 0, 0])
    torch.testing.assert_close(rgb[:, 0, 1], second[:, 0, 1])


def test_homography_mix_averages_only_valid_sources() -> None:
    first = torch.full((3, 1, 2), 0.2)
    second = torch.full((3, 1, 2), 0.8)
    first_valid = torch.tensor([[True, True]])
    second_valid = torch.tensor([[True, False]])

    rgb, valid, _ = MODULE.aggregate_homographies(
        [first, second], [first_valid, second_valid], mode="mix"
    )

    assert bool(valid.all())
    torch.testing.assert_close(rgb[:, 0, 0], torch.full((3,), 0.5))
    torch.testing.assert_close(rgb[:, 0, 1], first[:, 0, 1])
