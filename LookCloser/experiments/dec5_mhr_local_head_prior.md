# MHR head/neck local-prior fit: 001193

## What was tested

A bounded follow-up to the [canonical-face domain failure](dec5_canonical_face_prior.md): can a full head/neck human prior supply the missing underside while remaining compatible with trusted original COLMAP geometry?

Three CPU-only arms were tested and retained. Named articulation improves the underside fit, but all three fail the frozen validation/local-boundary gate. **No original triangles were changed, no completion mesh was added, and no production or 6K output was modified.** No second time, fourth arm, or temporal run was started.

The public MHR model and anatomical domain were independently checked in the [preflight](dec5_mhr_head_prior_preflight.md). The [pinned official MHR documentation](https://github.com/facebookresearch/MHR/blob/d96fafa33bbf018647c70c3525e91f53e79d2a14/README.md) identifies 45 identity parameters, including 20 head parameters, and differentiable TorchScript inference. This experiment uses that public Apache-2.0 model, not a SAM3D image checkpoint. Asset release `v1.0.1`; model SHA-256 `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc`. Source and release licenses remain with the preflight artifacts. No packages, model weights, source images or calibration were changed; no images were uploaded.

Root: [/mnt/data/dec5_mhr_local_head_prior](/mnt/data/dec5_mhr_local_head_prior/protocol.json).

Initialization and evidence:

- Approximate nose, eye-corner, mouth-corner and chin annotations on the neutral model's front render initialize a model-to-canonical similarity. Only the prior canonical **similarity pose** is transferred to scene coordinates. These annotations are not measured actor 3D landmarks; predicted MediaPipe z and independent landmark triangulations are not used. Chin is initialization-only, not a fixed contour correspondence during fitting.
- Use the same calibrated display-domain cached RGB and seven interior landmark correspondences. The same eight TRAIN cameras remain excluded from fitting and all other-camera depth support. The three dataset held-out cameras never enter this experiment. There are 49 detected fitting cameras and eight detected validation cameras for landmarks; all 62 train cameras contribute geometrically screened anchor candidates.
- Existing official MediaPipe selfie multiclass model identifies combined face/body skin; require quantized confidence at least 230/255, eroded by five native pixels. Sample an eight-pixel grid. A geometric anchor must agree with the original TSDF and the camera's geometric depth within 0.001, plus at least three distinct fitting-camera geometric depths within 0.001, 1.5 pixels reprojection and one degree parallax. Reserved cameras are excluded from those support votes even for validation anchors.
- Initial model-coordinate domain is y=143..176 cm and |x|<12 cm, with model-surface distance at most 0.006 scene units. Deterministically retain at most 100 points per initial upper/lower anatomical candidate group per camera. This produces 12,400 candidates: 10,800 fitting and 1,600 reserved-camera points. Candidate group labels are approximate spatial buckets, not verified anatomy.
- During fitting, recompute nearest model triangle/barycentric correspondence, excluding triangles extending below neutral y=140 cm and correspondences farther than 0.006. Balance upper/lower candidate groups. Combine robust 0.001-scale point-to-plane residuals, weak 0.004-scale point-distance residuals, and 4-pixel interior-landmark residuals. Four correspondence updates, each with at most 25 LBFGS iterations; no validation-driven tuning or target-based selection.

All reported 3D errors use the calibrated normalized COLMAP scene gauge, not assumed millimeters. Model-coordinate centimeter bands are only anatomical bookkeeping and are explicitly distinguished from scene-distance thresholds.

The eight-camera reservation is a fitting/depth-vote partition, not a newly reconstructed held-out COLMAP baseline: the preserved original mesh was built from the existing train set. Its geometry can therefore indirectly contain information from those cameras. This limits claims of fully independent geometric validation, without changing the observed failure or admitting the dataset's held-out views.

Arms:

1. `similarity`: seven global similarity parameters; identity, articulation and expression zero.
2. `head20`: same similarity plus head identity indices 20..39, strong prior standard deviation 0.5 and bounds ±1.5. Body identity, articulation and expression stay zero.
3. `head20_neck6`: separately authorized and protocol-pinned after both controls failed. Initialize from retained `head20`; add only named indices 24..29 (`neck_twist`, `neck_lean`, `neck_bend`, `head_twist`, `head_lean`, `head_bend`) with a 0.15-radian prior and official bounds. Names and matrix effects were verified against the serialized model and [mapping evidence](/mnt/data/dec5_mhr_articulation_mapping/mapping.json). No guessed jaw control, neck-length, body-identity or expression parameter. Pose correctives may move other vertices of the **prior**; original COLMAP remains untouched.

Frozen acceptance remains point-to-plane P90 ≤0.002, interior landmark median/P90 ≤4/8 pixels, plus local proposal distance to original surface ≤0.002 and to original open boundary ≤0.003. The target camera and previously selected 44-pixel jaw component are evaluated only after fitting. No thresholds were widened.

## Results

| Reserved-camera metric | Similarity | Head20 | Head20 + named neck/head6 |
|---|---:|---:|---:|
| Upper candidate group point-to-plane P90, 800 points | 0.001291 | 0.001296 | 0.001515 |
| Lower candidate group point-to-plane median | 0.001875 | 0.001853 | 0.001185 |
| Lower candidate group point-to-plane P90, 800 points | 0.005526 | 0.005546 | 0.003967 |
| Interior landmarks median / P90 pixels, 56 observations | 7.548 / 13.948 | 7.478 / 14.194 | 7.526 / 12.970 |
| Maximum head identity coefficient magnitude | 0 | 0.02085 | 0.04456 |
| CPU fitting seconds | 3.65 | 3.81 | 6.25 |

Those times are fitting only, not asset load, anchor generation, rendering or audit. Skin inference on 62 cached train crops took 8.92 seconds in the isolated existing MediaPipe environment. Inference explicitly used CPU/XNNPACK; MediaPipe initialized an EGL context but did not run a CUDA inference job.

The strong head-shape prior produces little identity change in this bounded configuration. This is not evidence that arbitrary MHR head coefficients cannot represent the actor. Named articulation reduces lower-surface error but slightly worsens upper-face surface error. Its maximum fitted angle is 0.09742 radians; no rotation approaches an official limit. Global similarity and neck/head rotations are partially coupled; no covariance or split-fit uncertainty guarantee is claimed.

### Actual anatomical association, not raw cyan-point counts

The 6,200 raw lower candidate points visibly include lower face, neck and clavicles/chest. They must not all be credited as measured underside support. Final closest-triangle coordinates give:

| Model-defined region | Head20 train associated / validation | Head20 validation P90 | Articulated train associated / validation | Articulated validation P90 |
|---|---:|---:|---:|---:|
| Lower front face, 145≤y<153 and z≥0 cm | 3,827 / 619 | 0.005673 | 4,228 / 600 | 0.003817 |
| Neck band, 135≤y<145 cm | 304 / 53 | 0.005623 | 941 / 157 | 0.004639 |
| Upper face, y≥153 cm | 5,289 / 789 | 0.001284 | 5,265 / 792 | 0.001488 |

Training counts include the 0.006 association cutoff; validation includes all candidates assigned to that anatomical band, without discarding large errors. These three regions are not exhaustive (e.g. posterior lower-head associations remain separate). Median other-camera depth votes are about 38 for upper face and 41–42 for lower face/neck, but votes are correlated reconstruction estimates, not independent ground truth.

Six train anchor overlays were inspected. Yellow samples mostly lie on face skin, with some ear and periocular/lip samples; cyan extends down the neck and onto clavicles. No obvious hand, hair mass, background or reflection anchors were seen in these six overlays. This is not a quantitative all-camera semantic-accuracy claim: the skin model can misclassify eyes/lips and no per-point anatomical ground truth is available. Ears are genuine skin geometry but not specifically underside support.

![Train G_B anchored skin: upper yellow, lower candidate group cyan](/mnt/data/dec5_mhr_local_head_prior/anchor_review/G004_B005_1210FG.png)

### Requested hole: correct anatomical domain, still wrong local placement

| Original jaw-component diagnostic | Similarity | Head20 | Articulated |
|---|---:|---:|---:|
| Original missing rays / prior hits | 44 / 44 | 44 / 44 | 44 / 44 |
| Prior-hit points passing frozen locality | **0 / 44** | **0 / 44** | **0 / 44** |
| Distance to old boundary, median | 0.007129 | 0.007125 | 0.005753 |
| Original visible rim-to-prior distance median / P90 | 0.005997 / 0.006103 | 0.006009 / 0.006111 | 0.004045 / 0.004128 |
| Rim signed original-normal offset median / P90 | +0.005529 / +0.006008 | +0.005548 / +0.006017 | +0.003808 / +0.004019 |
| Rim samples within 0.002 surface limit | 3 / 24 | 3 / 24 | 6 / 24 |

The articulated prior-hit surface distance median is only 0.001092, yet its distance to the actual open boundary is 0.005753: proximity to an existing nearby surface still does not make a valid hole fill. Unlike the face-only canonical model, MHR has underside/neck topology. This test fails placement and local precision, not anatomical-domain availability.

Native clay shows a coherent generic head but materially different identity, jaw shape and neck curvature. The articulated arm retains an angular/pinched neck transition in oblique views. Original COLMAP better captures the actor's face and visible neck. The parent independently inspected the initial alignment, anchor panel and final target clay and agreed that this is not an accepted repair.

![Train E_B: RGB, unchanged original, articulated prior only](/mnt/data/dec5_mhr_local_head_prior/review_head20_neck6/E004_B005_1210I7.png)

![Requested jaw hole: original and articulated prior only; fixed target marked red](/mnt/data/dec5_mhr_local_head_prior/probe_head20_neck6/requested_hole_clay_native.png)

No combined mesh or RGB prediction was created after this rejection. PSNR, SSIM and LPIPS are N/A; clay is a geometric diagnostic, not an RGB reconstruction. No hair quality claim is made.

## Insights

Full head/neck topology is necessary but not sufficient. Here, articulation improves the fit substantially more than strongly regularized head identity alone, but the retained prior remains several scene-distance tolerances away from the requested rim. Image-ray coverage would falsely suggest success.

The independent [head-parameter support probe](dec5_mhr_head_parameter_support.md) found weak neck response to head identity, explaining why neck topology alone did not guarantee fitting freedom. This experiment does not establish failure of full MHR, another regularization strength, body/neck-shape parameters, expression fitting or smooth surface conformance. None was tested. The bounded three-arm experiment ends with the local negative result preserved, without inventing a patch or enlarging tolerances.

Any future conformance experiment must remain a separate hypothesis and retain independent cameras, semantic/locality safeguards and unchanged original triangles. Final nearest-triangle indices, barycentric weights, original observed points/normals, camera identity and validation membership are preserved per arm in `final_associations.npz`; these are inferred model correspondences, not measured anatomical identity.

### Reproduction and validation

Scripts: `study_mhr_local_head_prior.py` (`init`, `semantic`, `anchors`), `fit_mhr_local_head_prior.py`, `fit_mhr_named_articulation.py`, `review_mhr_local_head_prior.py`, `probe_mhr_local_head_prior.py`, `audit_mhr_local_head_prior.py`. The explicit artifact-root constant must be changed consistently for a fresh rerun; production defaults are not involved. Existing fit/review/probe roots reject accidental overwrites.

Use reconstruction Python `/home/brans/repos/nerfstudio/.venv/bin/python`, except `semantic` uses `/home/brans/lookcloser_temp/mediapipe_hand_env/bin/python`. Set `OPENCV_IO_ENABLE_OPENEXR=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2`. Run the initial two controls first; the articulated arm has its own protocol and mapping binding. The review/probe helpers accept `--arms head20_neck6` for that third arm.

Four synthetic tests pass: differentiable Rodrigues agreement, finite nonzero derivative at zero rotation, chin initialization-only policy, and exact crop/native integer-coordinate mapping. Command: `python -m pytest -o addopts='' -q tests/test_mhr_local_head_prior.py`.

The [final audit](/mnt/data/dec5_mhr_local_head_prior/audit.json) verifies 173 explicit file bindings plus all 62 source depth-map hashes, replays all 12,400 other-camera support counts, 75,597 fit residuals and three exact model forwards, and seals the artifact inventory. Original mesh SHA remains `316dc2b6a82c0d99a6e399d15d941a69d491896921bc82e4d88ea307bb358b03`.

Primary evidence: [initial protocol](/mnt/data/dec5_mhr_local_head_prior/protocol.json), [two-control summary](/mnt/data/dec5_mhr_local_head_prior/fit_summary.json), [third-arm result](/mnt/data/dec5_mhr_local_head_prior/head20_neck6/result.json), [actual anatomical/boundary probe](/mnt/data/dec5_mhr_local_head_prior/probe_head20_neck6/result.json). All prior-only meshes, failed controls and diagnostic workspaces remain available.
