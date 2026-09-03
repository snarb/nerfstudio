from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_colmap_patch_match_config.py"
SPEC = importlib.util.spec_from_file_location("build_colmap_patch_match_config", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def frame(name: str, x: float) -> dict[str, object]:
    transform = np.eye(4)
    transform[0, 3] = x
    return {"file_path": name, "transform_matrix": transform.tolist()}


def test_explicit_sources_are_nearest_and_exclude_reference() -> None:
    mapping = MODULE.explicit_source_views(
        [frame("images/a.jpg", 0.0), frame("images/b.jpg", 1.0), frame("images/c.jpg", 3.0)],
        source_count=1,
    )

    assert mapping == {
        "images/a.jpg": ["images/b.jpg"],
        "images/b.jpg": ["images/a.jpg"],
        "images/c.jpg": ["images/b.jpg"],
    }


def test_source_count_is_capped_by_available_cameras() -> None:
    mapping = MODULE.explicit_source_views(
        [frame("a.jpg", 0.0), frame("b.jpg", 1.0)],
        source_count=20,
    )

    assert mapping["a.jpg"] == ["b.jpg"]
    assert mapping["b.jpg"] == ["a.jpg"]
