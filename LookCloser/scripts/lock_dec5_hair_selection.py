#!/usr/bin/env python3
"""Lock a measured candidate against a fixed corrected baseline, before test RGB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_dec5_hair_campaign import atomic_json, sha


def admission(base, candidate):
    values = [x for a in (base, candidate) for x in (a['hair']['lpips'], a['face_neck']['lpips'],
              a['holes_fraction'], a['witness_deficit_fraction'], a['silhouette_recall'])]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite acceptance metrics')
    hair_gain = candidate['hair']['lpips'] <= base['hair']['lpips']*.95
    holes_gain = base['holes_fraction'] > 0 and candidate['holes_fraction'] <= base['holes_fraction']*.8
    popping_gain = (base.get('camera_popping_fraction', 0) > 0 and
                    candidate.get('camera_popping_fraction', float('inf')) <= base['camera_popping_fraction']*.8)
    geometry_gain = holes_gain or (base['witness_deficit_fraction'] > 0 and
                         candidate['witness_deficit_fraction'] <= base['witness_deficit_fraction']*.99)
    if base.get('reprojection_error_hair') is not None and candidate.get('reprojection_error_hair') is not None:
        geometry_gain = geometry_gain or (base['reprojection_error_hair'] > 0 and
                            candidate['reprojection_error_hair'] <= base['reprojection_error_hair']*.99)
    face_guard = candidate['face_neck']['lpips'] <= base['face_neck']['lpips']*1.01
    recall_guard = candidate['silhouette_recall'] >= base['silhouette_recall']-.01
    result = dict(hair_lpips_gain_5_percent=bool(hair_gain), holes_gain_20_percent=bool(holes_gain),
                  camera_popping_gain_20_percent=bool(popping_gain),
                  independent_depth_proxy_improved=bool(geometry_gain), face_guard=bool(face_guard),
                  silhouette_recall_guard=bool(recall_guard))
    result['passes'] = bool(geometry_gain and (hair_gain or holes_gain or popping_gain) and face_guard and recall_guard)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidates', type=Path, nargs='*', default=[])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--path-reviews', type=Path,
                   help='Optional JSON mapping candidate metric filenames to frozen-path review metrics.')
    p.add_argument('--review-decisions', type=Path,
                   help='Final visual decisions for every numerically eligible candidate, before test RGB.')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Selection is immutable')
    base = json.loads(args.baseline.read_text())
    reviews = json.loads(args.path_reviews.read_text()) if args.path_reviews else {}
    decisions = json.loads(args.review_decisions.read_text()) if args.review_decisions else {}
    eligible, rows = [], []
    for path in args.candidates:
        candidate = json.loads(path.read_text())
        if candidate['semantic_reference_receipt_sha256'] != base['semantic_reference_receipt_sha256']:
            raise ValueError('Candidate uses a different fixed evaluation domain')
        base_values, values = dict(base['aggregate']), dict(candidate['aggregate'])
        proof = {}
        if str(path) in reviews:
            review_path = Path(reviews[str(path)])
            review = json.loads(review_path.read_text())
            if not review['camera_path_only'] or review['missing_candidate_pixels_excluded']:
                raise ValueError('Path review must preserve the fixed missing-pixel domain')
            if review.get('frame_count', 96) != 96:
                raise ValueError('A short pilot cannot replace the final 96-camera gate')
            base_values['camera_popping_fraction'] = review['aggregate']['baseline']['visibility_toggles_fraction']
            values['camera_popping_fraction'] = review['aggregate']['candidate']['visibility_toggles_fraction']
            proof = dict(path_review=str(review_path), path_review_sha256=sha(review_path))
        gate = admission(base_values, values)
        if args.review_decisions is not None and gate['passes']:
            if str(path) not in decisions:
                raise ValueError('Numerically eligible candidate needs an explicit visual decision')
            decision = decisions[str(path)]
            if not isinstance(decision.get('accepted'), bool) or not decision.get('reason'):
                raise ValueError('Visual decision requires accepted boolean and reason')
            proof['visual_decision'] = decision
            proof['visual_evidence'] = [dict(path=str(p), sha256=sha(p)) for p in map(Path, decision.get('evidence', []))]
            if decision['accepted'] and str(path) not in reviews:
                raise ValueError('Visual promotion requires the full 96-camera review')
            gate['numerical_passes_before_visual_review'] = gate['passes']
            gate['passes'] = bool(gate['passes'] and decision['accepted'])
        rows.append(dict(metrics=str(path), metrics_sha256=sha(path), **proof, **gate))
        if gate['passes']:
            eligible.append((candidate['aggregate']['hair']['lpips'], str(path)))
    chosen = min(eligible)[1] if eligible else str(args.baseline)
    atomic_json(args.output, dict(baseline=str(args.baseline), baseline_sha256=sha(args.baseline),
        candidates=rows, chosen_metrics=chosen, chosen_metrics_sha256=sha(chosen),
        independent_test_rgb_used=False, test_unlocked=False,
        status=('validation and visual review locked before test RGB' if args.review_decisions else
                'validation selection; repeat and visual path review still required')))


if __name__ == '__main__':
    main()
