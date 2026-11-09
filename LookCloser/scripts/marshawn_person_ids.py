"""Conservative short-interval person-ID association for fixed-camera masks.

Global actor IDs are seeded once across cameras. Local SAM instance numbers may
change every frame. This associates temporal detections within ONE camera; it
does not infer cross-camera identities from mask overlap or certify mask edges.
Ambiguous masks/new people remain unknown. Partial target detections can never
become automatic background vetoes merely because Hungarian leaves them unmatched.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class CameraMask:
    mask: np.ndarray
    origin_xy: Tuple[int, int]
    full_hw: Tuple[int, int]
    camera_id: str
    coordinate_frame: str
    calibration_fingerprint: str

    def __post_init__(self):
        self.mask = np.asarray(self.mask)
        if self.mask.ndim != 2 or self.mask.dtype != np.bool_:
            raise ValueError("Expected a binary 2D mask; no resizing or threshold guessing")
        if not self.coordinate_frame or not self.calibration_fingerprint:
            raise ValueError("Explicit coordinate frame and calibration fingerprint required")
        x, y = self.origin_xy
        h, w = self.mask.shape
        if x < 0 or y < 0 or x + w > self.full_hw[1] or y + h > self.full_hw[0]:
            raise ValueError("Mask crop lies outside its declared source raster")
        yy, xx = np.nonzero(self.mask)
        self.area = len(xx)
        self.bounds = ((x + int(xx.min()), y + int(yy.min()),
                        x + int(xx.max()) + 1, y + int(yy.max()) + 1)
                       if self.area else (x, y, x, y))


def intersection(a: CameraMask, b: CameraMask) -> int:
    if (a.camera_id, a.full_hw, a.coordinate_frame, a.calibration_fingerprint) != (
            b.camera_id, b.full_hw, b.coordinate_frame, b.calibration_fingerprint):
        raise ValueError("Temporal overlap requires the same camera and source raster")
    x0, y0 = max(a.bounds[0], b.bounds[0]), max(a.bounds[1], b.bounds[1])
    x1, y1 = min(a.bounds[2], b.bounds[2]), min(a.bounds[3], b.bounds[3])
    if x0 >= x1 or y0 >= y1:
        return 0
    ax, ay = a.origin_xy
    bx, by = b.origin_xy
    return int(np.count_nonzero(a.mask[y0 - ay:y1 - ay, x0 - ax:x1 - ax]
                                & b.mask[y0 - by:y1 - by, x0 - bx:x1 - bx]))


def associate_person_ids(previous: Dict[str, CameraMask], current: List[CameraMask],
                         target_ids, *, minimum_iou=.15, ambiguity_margin=.05,
                         duplicate_purity=.9) -> dict:
    """Associate short-interval masks; preserve explicit uncertainty.

    Inputs must already share an undistorted camera raster, or share the RAW
    distorted raster and correct crop origins. Mixed coordinate frames are an
    input error. Appearance/flow can be added later; large motion or full
    occlusion deliberately yields unknown instead of an invented identity.
    """
    if not 0 < minimum_iou <= 1 or not 0 <= ambiguity_margin < 1 or not 0 < duplicate_purity <= 1:
        raise ValueError("Invalid association thresholds")
    ids = list(previous)
    targets = set(target_ids)
    if not targets.issubset(previous):
        raise ValueError("Target identities must have reviewed previous-frame seeds")
    if not previous:
        raise ValueError("At least one previous identity is required")
    reference = next(iter(previous.values()))
    for mask in list(previous.values()) + list(current):
        if (mask.camera_id, mask.full_hw, mask.coordinate_frame, mask.calibration_fingerprint) != (
                reference.camera_id, reference.full_hw, reference.coordinate_frame,
                reference.calibration_fingerprint):
            raise ValueError("Cannot associate different cameras or raster conventions")
    intersections = np.zeros((len(ids), len(current)), dtype=np.float64)
    scores = np.zeros_like(intersections)
    for i, name in enumerate(ids):
        for j, mask in enumerate(current):
            n = intersection(previous[name], mask)
            intersections[i, j] = n
            scores[i, j] = n / max(1, previous[name].area + mask.area - n)
    rows = [dict(instance=j, global_id=None, role="unknown", status="unmatched",
                 safe_to_veto=False, possible_ids=[])
            for j in range(len(current))]
    matched = {}
    if len(current):
        ii, jj = linear_sum_assignment(-scores)
        for i, j in zip(ii, jj):
            best = float(scores[i, j])
            if best < minimum_iou:
                continue
            alternatives = [ids[k] for k in range(len(ids))
                            if scores[k, j] >= minimum_iou and best - scores[k, j] < ambiguity_margin]
            if ids[i] not in alternatives:
                alternatives.append(ids[i])
            row = rows[j]
            row.update(score=best, possible_ids=alternatives)
            if len(alternatives) > 1:
                row.update(role="target_group" if set(alternatives).issubset(targets) else "unknown",
                           status="ambiguous_identity")
                continue
            name = ids[i]
            matched[name] = int(j)
            row.update(global_id=name, role="target" if name in targets else "other_person",
                       status="matched", safe_to_veto=name not in targets)
    # Classify fragments against matched current targets, without unioning masks.
    for j, mask in enumerate(current):
        target_overlap = []
        for name, k in matched.items():
            if name in targets and k != j and mask.area:
                purity = intersection(mask, current[k]) / mask.area
                if purity >= duplicate_purity:
                    target_overlap.append(name)
        if target_overlap:
            rows[j].update(global_id=target_overlap[0] if len(target_overlap) == 1 else None,
                           role="target_fragment", status="duplicate_do_not_veto",
                           possible_ids=target_overlap, safe_to_veto=False)
        # A purported other person overlapping any target has uncertain pixels.
        if rows[j]["safe_to_veto"]:
            if (any(name in targets and intersection(mask, current[k]) > 0
                    for name, k in matched.items())
                    or any(intersection(mask, previous[name]) > 0 for name in targets)):
                rows[j].update(safe_to_veto=False, status="overlap_requires_pixel_review")
        rows[j]["bbox_target_eligible"] = (
            rows[j]["role"] in {"target", "target_group", "target_fragment"}
            and bool(rows[j]["possible_ids"])
            and set(rows[j]["possible_ids"]).issubset(targets))
    return {"associations": rows, "target_ids": sorted(targets),
            "missing_target_ids": sorted(targets - set(matched)),
            "coordinate_scope": "same camera/raster, global crop origins respected",
            "unknown_is_not_background": True,
            "pixel_masks_automatically_unioned": False,
            "temporal_video_quality_validated": False}
