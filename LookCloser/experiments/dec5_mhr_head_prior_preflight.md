# DEC5: full-head/neck prior availability and anatomical-domain preflight

## What was tested

2026-09-15. The [canonical face pilot](dec5_canonical_face_prior.md) found a
stable but anatomically incorrect front-face overlap at the requested jaw gap.
Its template stops at the face contour and cannot supply the missing underside.
This preflight checks a different prior's domain and implementation, not its fit
to DEC5 and not a replacement for measured COLMAP geometry.

The official [Momentum Human Rig](https://github.com/facebookresearch/MHR)
contains full-body geometry, including neck and lower-jaw surfaces, with twenty
head identity coefficients among forty-five identity coefficients. We use its
public TorchScript model, not SAM3D image inference or a gated checkpoint.
Both repository and actual release asset carry Apache-2.0 licenses, retained.
Attribution: MHR, Meta Platforms, Inc. and affiliates. Neutral outputs below
are generated from that model; no scene-specific modification has been fitted.

Repository pin: `d96fafa33bbf018647c70c3525e91f53e79d2a14`.
Release: `v1.0.1`, public `assets.zip`, 198,943,157 bytes, SHA-256 checked against
the publisher's GitHub release digest:

```text
e4f4f205cd87c0fa106577ba1de4fc763e4eb197c924461d2ef7e6944e9d6b94
```

Only the 696 MB TorchScript member is extracted for inference; the larger
per-LOD corrective arrays are not expanded. The pinned public
`mhr_face_mask.ply` supplies topology, independently checked against the
TorchScript model's own triangle buffer. Its vertex positions are a different
pose/shape and are not used as the neutral model.

[FLAME 2023 Open](https://flame.is.tue.mpg.de/) was also checked as a candidate.
The official download leads to sign-in; no locally cached FLAME model was found.
No account was created, credentials accessed, or access restriction bypassed.
This is a specific unavailable asset, not a blocker on the active experiment.

## Results

- CPU forward pass works in existing PyTorch 2.7.1+cu128/Python 3.10 environment.
  No environment packages were installed or changed.
- Neutral surface: 18,439 vertices, 36,874 triangles; coordinates in the model's
  centimeter convention. No open or nonmanifold edges in this topology.
- Autograd is finite and nonzero for head identity coefficients. This is a
  genuine backward-pass check, not only successful inference.
- Independent neutral forward replay is exact; imported faces match the model
  triangle buffer exactly. Head-edge median/P99 lengths are 0.4313/2.3533 cm:
  this is a low-frequency anatomical prior, not recovered fine skin detail.
- Main LLM inspected front, side and low-oblique head/neck clay panels. The
  underside of the jaw and neck are continuous, with no obvious topology
  corruption. The generic face is visibly not the captured actor.

![Neutral anatomical-domain inspection](/mnt/data/dec5_mhr_head_prior_preflight/review/anatomy.png)

Root: `/mnt/data/dec5_mhr_head_prior_preflight`. `neutral.ply` is a generic prior,
**not a new DEC5 mesh**. No source RGB, calibration, original mesh, exposure,
renderer, active 6K job or production default was changed. No image-quality
metrics apply to an unfitted neutral model.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python scripts/prepare_mhr_head_prior.py prepare
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python scripts/prepare_mhr_head_prior.py probe
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python scripts/review_mhr_head_prior.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python scripts/audit_mhr_head_prior_preflight.py check
```

The first three commands require an unused output/attempt; current artifacts
are immutable. `check` replays neutral inference and validates the sealed files.

## Insights

This prior covers the anatomical domain missing from the canonical face model
and can be optimized without new inference dependencies. Neither property proves
that its shape fits this woman, that it has metric accuracy at the hole boundary,
or that inferred additions obey measured multiview visibility.

The next bounded test should fit train-only high-confidence geometry/landmarks
with eight train cameras reserved for validation, preserving the original mesh.
Require independent boundary precision and native low-angle review before any
local addition. Do not accept apparent screen coverage by a wrong-depth surface,
and do not substitute this coarse whole head for the detailed COLMAP face.
