# Lipstick fin: why near-depth protection remains

## What was tested

Read-only follow-up to [measured free-space pruning](dec5_lipstick_fin_measured_free_space.md),
frame `000995`. Hypothesis: a noisy, uncorroborated near observation protects a false
triangle from deletion. Replayed the native-center near gate (`0.0015` normalized
depth tolerance) on the same 86 diagnostic triangles, then corroborated each actual
measured point against the other 61 train views. No mesh, exposure, renderer or
production default was changed. Diagnostic triangle selection is not a deletion rule.

## Results

| Required other views corroborating a near observation | Remaining diagnostic faces protected |
|---:|---:|
| 0 | 38/38 |
| 1 | 38/38 |
| 2 | 38/38 |
| 3 | 38/38 |

The representative fin triangle `48941` is protected at just one vertex by
`I004_C005_1210BA` and `J004_C005_1210I4`. Their measured/query depth differences
are `-0.00137529` and `-0.00144813`; **each actual measured point agrees with 28
other views**. This corroborates the measured points, not the entire triangle or
the exact query vertex. The finite near tolerance still matters.

The auditor replayed 21,328 camera/sample queries and 2,392 near observations.
Near-pixel arithmetic is independent; inter-camera corroboration deliberately
reuses the established helper. Three focused tests pass. An initial audit import
error occurred before publication; the corrected audit exited zero. Both logs
are retained under `/mnt/data/dec5_lipstick_near_protection_audit*.log`.

Main-agent native visual inspection covered both actual train patches:

- [I/C RGB and projected triangle](/mnt/data/dec5_lipstick_fin_near_protection/000995/review/I004_C005_1210BA.png)
- [J/C RGB and projected triangle](/mnt/data/dec5_lipstick_fin_near_protection/000995/review/J004_C005_1210I4.png)

They show a real skin-colored region behind the foreground finger/object, rather
than an image void. The projected triangle is tiny; RGB alone does not establish
its correct depth or prove that its full area is supported. This is a completed
diagnostic review, not an artifact-free render verdict.

Evidence and binding audit:
`/mnt/data/dec5_lipstick_fin_near_protection/000995/{result.json,evidence.npz,audit.json}`.
The numerical auditor's `visual_status=pending` is its pre-review state; the actual
two-image visual assessment is recorded above. No quality metrics were computed.

## Insights

The singleton-noise hypothesis is rejected. Raising the minimum corroboration
to three does not release any remaining diagnostic triangle. Whole-triangle
protection is coarser than the evidence: one near vertex protects its other
vertices and interior. The next bounded test is conforming subdivision followed
by the same measured-depth gates on smaller faces, with subdivision-only RGB as
a control because changing the mesh adjacency graph can change texture labels.
No current production geometry is promoted or replaced by this diagnosis.
