# MHR: verified named head/neck articulation mapping

## What was tested

2026-09-15. The bounded scene fit with frozen articulation could not match the
under-jaw neighborhood. Before testing articulation, read the official serialized
model's actual `parameter_names`, `joint_names` and dense parameter-transform
matrix. Do not infer indices from textual parameter order or guess a jaw index.
The model is the hash-verified [MHR preflight asset](dec5_mhr_head_prior_preflight.md).

`probe_mhr_articulation_mapping.py` independently verifies that the transform has
seven rows per joint and one column per serialized name. The explicit release
`compact_v6_1.model` definition is also extracted and retained. Each named
rotation gets a +0.1-radian forward probe with all other inputs zero; no DEC5
data or scene geometry enters the diagnostic.

## Results

| Actual index | Official name | Official min / max | Head RMS motion at +0.1 rad, cm |
|---:|---|---:|---:|
| 24 | neck_twist | −0.8 / 0.8 | 0.709 |
| 25 | neck_lean | −0.5 / 0.5 | 1.842 |
| 26 | neck_bend | −0.6 / 0.5 | 1.873 |
| 27 | head_twist | −0.8 / 0.8 | 0.832 |
| 28 | head_lean | −0.3 / 0.3 | 0.899 |
| 29 | head_bend | −0.4 / 0.4 | 1.052 |

Indices 24–26 directly drive rotation slots of `c_neck`; 27–29 drive `c_head`.
Neck twist also drives two procedural twist joints with coefficients −0.5 and −1.
All six forward probes are finite. The diagnostic head band is neutral `y>=145`
cm; neck band `135<=y<145` has RMS motion 0.0133–0.1027 cm. These are coarse
model-coordinate bands, not anatomical ground-truth masks.

Unlike head-identity perturbations, these pose probes also slightly move some
vertices below `y=135`, with maximum displacement 0.0438–0.1151 cm. Therefore
an articulated prior must not be described as an otherwise byte-frozen body,
even when body identity coefficients stay zero. The actual original COLMAP
mesh remains untouched.

There is a joint named `c_jaw` but **no jaw-named parameter among the first 204
pose parameters**. The presence of a joint does not justify inventing a pose
index. This probe does not test facial expression parameters or claim that MHR
lacks jaw expression controls.

Root: `/mnt/data/dec5_mhr_articulation_mapping`. `mapping.json` binds actual
serialized names, nonzero matrix effects, release limit lines, model/script
hashes and all seven probe surfaces in `pose_probe.npz`. This is numeric rig
verification, not a visual acceptance of an actor fit.

## Insights

A separately pinned six-rotation control is now reproducible without installing
PyMomentum or guessing parameter ordering. The scene-fit agent uses these six
named rotations, official bounds and a strong 0.15-radian prior, preserving its
failed similarity/head20 controls and independent validation cameras.

Global pose and head/neck articulation can be redundant if valid neck anchors
are sparse. Report actual associated neck/underside anchors and their errors;
do not count raw chest candidates as support. No boundary tolerance is widened,
no whole-face replacement is authorized by this diagnostic, and no 6K render
parameter is changed.
