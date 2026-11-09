# Native63 terminal occupancy audit

## What was tested

Actual Core140 (`Batch8kLong_core001`, terminal139999) occupancy/frequency tensors, calibrated CPU ray/contour checks, source temporal identity review, and one33.922sread-only GPU traversal comparison. Checkpoint SHA`a2a76583…`; scientific masks, grids, model, RNG and metrics remain unchanged.

## Results

29599actual core/active cells satisfy R0 and fixed108699-cell TRAIN63admission;0lie outside it. Largest of249six-connected components contains26710cells. AllXYZslices and8camera overlays were visually inspected. Frequency at occupancy centres is level0=9750(32.9403%), level15=19849(67.0597%); this is cell-centre coverage, not termination-weight coverage.

The original113contour misses were an8camera subset. Full715529rays across66views have510baseline misses, allgrey on the same black mesh/cap item in7TRAIN views; heldout0. Receipt-bound later IA/MAframes show boots moving away while that item remains at the same mat seam. ROOT independently confirmed the static-prop identity and additional5camera crops. Existing supervision/metrics were not corrected.

TRAIN-only scratch V2removes2719cells and preserves every baseline-supported nominal contour:0new misses across682725TRAIN/32804heldout rays. The bounded GPU comparison passes full state restoration and native baseline parity, using1.744GiBreported peak process memory. Agent and ROOT visual review show mostly real mat/floor contribution removed. These8tiny support-loss-selected patches demonstrate local foreground cutout, not physical empty space or an actor LPIPS gain. ROOT keeps the original grid and rejects promotion from this diagnostic.

[Full report, source hashes, temporal evidence, CPU/GPU proofs and native contacts](/fsx/tmp/marshawn_native63_24h_20261108/research/final_grid_audit/Core140/REPORT.md).

## Insights

Conservative traversal support is not true3Dgeometry. OrdinaryR1cannot repair admission misses because the same hard admission also gates density. A future TRAIN-only static-prop exclusion would require a separate annotation proof and controlled cold-trial factor; it is only a proposal, not implemented here.
