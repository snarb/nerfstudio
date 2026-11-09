"""Identity consistency when instance numbers/crop origins change between frames."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from marshawn_person_ids import CameraMask, associate_person_ids, intersection


def rectangle(x, y, w=20, h=40, *, camera="AA", frame="RAW_distorted"):
    return CameraMask(np.ones((h, w), dtype=bool), (x, y), (200, 300),
                      camera, frame, "fixed-intrinsics-sha")


def test_permuted_instances_and_native_crop_offsets_preserve_people():
    previous = {"actor_black": rectangle(30, 40), "actor_grey": rectangle(85, 50),
                "worker_0": rectangle(180, 20)}
    current = [rectangle(182, 20), rectangle(87, 51), rectangle(32, 40)]
    report = associate_person_ids(previous, current, ["actor_black", "actor_grey"])
    assert [r["global_id"] for r in report["associations"]] == ["worker_0", "actor_grey", "actor_black"]
    assert [r["safe_to_veto"] for r in report["associations"]] == [True, False, False]
    assert report["missing_target_ids"] == []


def test_unmatched_target_leg_fragment_is_never_background():
    previous = {"actor_black": rectangle(30, 40), "actor_grey": rectangle(85, 50)}
    current = [rectangle(31, 40), rectangle(86, 50), rectangle(32, 65, 8, 12), rectangle(220, 70)]
    report = associate_person_ids(previous, current, previous.keys())
    fragment, stranger = report["associations"][2:]
    assert fragment["global_id"] == "actor_black"
    assert fragment["role"] == "target_fragment"
    assert fragment["safe_to_veto"] is False
    assert stranger["role"] == "unknown" and stranger["global_id"] is None


def test_merged_actor_worker_detection_stays_unknown():
    previous = {"actor": rectangle(30, 40), "worker": rectangle(50, 40)}
    report = associate_person_ids(previous, [rectangle(30, 40, 40, 40)], ["actor"])
    item = report["associations"][0]
    assert item["status"] == "ambiguous_identity"
    assert item["role"] == "unknown" and item["safe_to_veto"] is False


def test_other_person_overlap_requires_pixel_review():
    previous = {"actor": rectangle(30, 40), "worker": rectangle(48, 40)}
    report = associate_person_ids(previous, [rectangle(30, 40), rectangle(48, 40)], ["actor"])
    assert report["associations"][1]["global_id"] == "worker"
    assert report["associations"][1]["status"] == "overlap_requires_pixel_review"
    assert report["associations"][1]["safe_to_veto"] is False


def test_coordinate_conventions_must_match_and_disappearance_is_explicit():
    a = rectangle(30, 40)
    with pytest.raises(ValueError, match="same camera"):
        intersection(a, rectangle(30, 40, camera="AB"))
    with pytest.raises(ValueError, match="same camera"):
        intersection(a, rectangle(30, 40, frame="undistorted"))
    report = associate_person_ids({"actor": a}, [], ["actor"])
    assert report["missing_target_ids"] == ["actor"]
    assert report["associations"] == []


def test_occluding_worker_cannot_veto_disappearing_target():
    previous = {"actor": rectangle(30, 40), "worker": rectangle(50, 40)}
    report = associate_person_ids(previous, [rectangle(48, 40, 24, 40)], ["actor"])
    item = report["associations"][0]
    assert item["global_id"] == "worker"
    assert not item["safe_to_veto"] and not item["bbox_target_eligible"]
    assert report["missing_target_ids"] == ["actor"]


def test_merged_two_actor_detection_can_bound_group_without_invented_single_id():
    previous = {"black": rectangle(30, 40), "grey": rectangle(50, 40)}
    report = associate_person_ids(previous, [rectangle(30, 40, 40, 40)], previous.keys())
    item = report["associations"][0]
    assert item["global_id"] is None and item["role"] == "target_group"
    assert item["bbox_target_eligible"] and not item["safe_to_veto"]
