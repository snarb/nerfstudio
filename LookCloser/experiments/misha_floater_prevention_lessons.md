# Misha masked-person floater prevention: measured lessons

## What was tested

Retrospective explanation of the delivered75512 checkpoint and related controlled interventions. This is a masked-person recipe; no background reconstruction. Proposed preventive policy is not a proven universal solution.

## Results

- The world-aligned142x84x128 query/occupancy domain extends real trouser space inX and excludes the traced large phantom inY. At the initial migrated75k state, the old frame11 large phantom disappears while old-region RGB/density are bit-exact. The support has one-cell tolerance and zero impossible opaque TRAIN pixels across66views under the artificial-capacity test. This does not certify true hidden anatomy or learned body completeness.
- Corrected semantics distinguish primary actor, foreground equipment, and genuine empty background. KD secondary-head correction gives a small matched LPIPS benefit, not complete forearm repair. BB omitted-stand-collar correction from zero removes a specific252pixel alpha<.1 novel forearm hole (all corrected alpha>.99); warm repairs were weaker. These findings concern contradictory supervision, not proof all floaters have one cause.
- Independent density-hash-only auxiliary cleanup on related branches reduces AB false-opaque pixels without the broad decoder changes that created holes. It is not evidence that all those branches were merged into delivered75512.
- Delivered75512 uses independent density encoding, density FR0,colour FR1,known-empty optical density weight.1,foreground opacity,corrected visibility,H1source support and finer integration. Density FR0 is a protective architecture choice here, not a separately established sole cause of floater removal.
- Fine8 training lowers person LPIPS.223225(native-training control) to.221212 under identical evaluation. Frozen fine stepping improves forearm LPIPS but not shirt; it does not remove remaining geometry floaters.
- Hard fine clipping can make AB dramatically cleaner while introducing real-shirt holes in AC/BB. Increasing occupancy dilation or full traversal alone does not fix learned bad density. Reject these as complete cures.

## Insights and preventive recipe (analysis; not a universal guarantee)

1. Before SGD, review actor identity and cross-view labels: visible actor, certain empty, occluded/unknown. Equipment/another actor is unknown for hidden-person constraints, never certain empty. Withdraw RGB,alpha,empty penalties consistently there.
2. Build world-space support from calibrated TRAIN evidence, protect measured uncertainty, cover full body, and test zero opaque-ray capacity misses across allTRAIN views. Compare no halo and one physical-cell halo. Preserve warm encoding coordinates when changing traversal bounds.
3. Teach both opaque body and certain empty space from the beginning; inspect actual ray mixture so empty observations are not silently excluded. Keep background-density pressure modest initially; .1 is the tested starting point here, not an invariant coefficient.
4. Keep density independent from appearance/frequency modulation for controlled geometry. If adding stronger local empty-space corrections, start by routing auxiliary gradients to density hash and compare against an unchanged control; do not freeze the density decoder for all ordinary training by assumption.
5. Choose integration step by scene/encoding scale and verify convergence on problematic crops. Do not universally copy1.25e-5 scene units or treat a larger maximum-sample cap as finer actual marching.
6. Review RGB+alpha+depth on allTRAIN cameras and an intermediate orbit early and after each intervention. Acceptance requires reduced false opacity without new body holes, plus person LPIPS. A frequency grid describes detail, not certified occupancy; a clean eval image is not a clean3D guarantee.

The strongest general lesson is correct visibility and spatial feasibility before optimization. Remaining hidden-volume ambiguity means this sequence reduces avoidable failure modes but cannot guarantee absence of every floater on another scene.

## Remaining DD defects reported by the user, 2026-11-01

On `best_DD_HD.png` from checkpoint 75,512, the user reports:

- White haze beside the right arm, identified as the arm closest to the viewer.
- Very small white spots on the same arm's T-shirt sleeve, at the outer edge/in a fold toward the back of the shoulder.

These remain open defects. Their cause has not been established; do not label them as mask contamination, density floaters, or an integration defect without a controlled check. Preserve the current full image as the reference. No new improvement series is authorized until the user gives the next instruction.

The [best Misha recipe](misha_002304_best_recipe.md) records the exact checkpoint, settings, policy hashes, metrics, and reference images. Detailed investigation records remain in `/fsx/lookcloser_misha_maskfix_20261031_pavlo/` on olive-lark and `/cache/ubuntu/lookcloser/misha_maskfix_20261031/` on silver-sable, including `extended_domain_review_v1`, `visible_density512_interval74_gates_v2`, `screen_v11`, and `fine_source_empty_review_v1`. These external artifacts are not bundled into this documentation commit.
