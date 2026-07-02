#!/usr/bin/env python3
"""Compute LookCloser's significant-artifact score over a temporal run's eval renders.

Nerfstudio ns-eval saves 2-panel `eval_img_NNNN.png` (GT | render). We split each into
GT (left) and candidate (right), run the LookCloser structural-artifact detector with the
'significant' preset, and aggregate `serious_artifact_score` across all eval images.

Target: 0.0 (no significant structural artifacts). Usage:
    python score_artifacts_temporal.py <render_dir> [--preset significant]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_structural_artifacts import detect_defects, PRESETS  # noqa: E402


def load_panels(path: Path, n_panels: int = 2):
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    w = im.shape[1] // n_panels
    return [im[:, i * w:(i + 1) * w, :] for i in range(n_panels)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("render_dir", type=Path)
    ap.add_argument("--preset", default="significant", choices=sorted(PRESETS))
    ap.add_argument("--gt-panel", type=int, default=0)
    ap.add_argument("--cand-panel", type=int, default=1)
    ap.add_argument("--n-panels", type=int, default=2)
    args = ap.parse_args()

    kwargs = dict(PRESETS[args.preset])
    imgs = sorted(args.render_dir.glob("eval_img_*.png"))
    if not imgs:
        print(f"no eval_img_*.png in {args.render_dir}", flush=True)
        return 1

    per = []
    for p in imgs:
        panels = load_panels(p, args.n_panels)
        gt, cand = panels[args.gt_panel], panels[args.cand_panel]
        res = detect_defects(gt, cand, **kwargs)
        per.append({"image": p.name,
                    "serious_artifact_score": res["serious_artifact_score"],
                    "artifact_score": res["artifact_score"],
                    "artifact_count": res["artifact_count"]})

    sas = np.array([r["serious_artifact_score"] for r in per])
    a = np.array([r["artifact_score"] for r in per])
    summary = {
        "preset": args.preset,
        "num_images": len(per),
        "significant_artifacts_score_mean": round(float(sas.mean()), 4),
        "significant_artifacts_score_max": round(float(sas.max()), 4),
        "num_images_with_significant_artifacts": int((sas > 0).sum()),
        "artifact_score_mean": round(float(a.mean()), 4),
        "artifact_score_max": round(float(a.max()), 4),
    }
    out = args.render_dir / "artifact_scores.json"
    out.write_text(json.dumps({"summary": summary, "per_image": per}, indent=2))
    print(json.dumps(summary), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
