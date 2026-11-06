# Independent occupancy and depth audit during frame1660 tuning

## What was tested

Frozen fields from the Full-HD tuning campaign, with the accepted floor-frame
box. This audit performs no optimizer updates and changes no production grid or
source. Temporary grid substitutions exist only in a separate read-only model;
the complete model-state digest is restored and checked afterward.

The source is the actual GPU-exported core/active mask, never a CPU reconstruction
from EMA. CPU-only checks cover every cell, every XYZ slice and exact per-cascade
ray intervals. A GPU replay compares native, core, exactly one halo, all cells,
and occupancy-independent 8192/16384-point integration. The 28 selected rays are
the two original training observations of 14 triangulated points on the white
lettering on the black actor's upper back. They are a deliberately problematic
subset, not a scene-wide metric sample.

Tool: `scripts/audit_marshawn_tuning_checkpoint.py`. Five tests pass, including
camera convention, unknown-space semantics, actual-core authority and separated
layer detection. Production source remains frozen and unchanged.

## Results

At control step2000, actual core has583610 cells but all2097152 cells remain
admitted by the intended binary warmup. This cannot establish halo quality.

At control step6000, core/active has819926 cells;1277226 cells are skipped and
have unknown physical geometry. Of9022 train-only triangulated witnesses,
725 lie inside the common box and18 fall in skipped cells. Four are background
witnesses;14 are actor shirt-lettering points, confirmed in original training
views AA/AB/BA/AC/AD. Exactly one halo admits all18 missed point cells. This is
point-coverage evidence, not proof that all those cells are solid.

The6000 field was additionally queried at all2097152 cell centers plus eight
quarter-cell positions:18874368 density queries. The largest sampled density
inside a skipped cell was20.890625. Finite subcell probes cannot certify
continuous geometry or true free space.

All28 rays converged under the declared8192/16384 checks: opacity difference
at most0.005 and foreground-RGB maximum-channel difference at most1/255. At6000,
native traversal skips mean17.41% of the same field's termination mass
(maximum35.00%). One halo changes RGB by more than1/255 on27/28 rays; mean
maximum-channel change is0.01281, maximum0.04775. Mean RGB disagreement to the
uniform reference changes0.03149→0.02749. The uniform reference is the learned
field, not ground-truth geometry.

Critically, triangulated shirt depth shows that most of the restored mass is
foreground fog, not the real shirt surface. The current early LPIPS leaders
were therefore audited at the matched15188 checkpoint as well:

| Frozen configuration | Mean termination before shirt band | In shirt band | Separated-layer candidates | Converged rays |
|---|---:|---:|---:|---:|
| Full/default/r0,6000 |99.42%|0.111%|7/28|28/28|
| Full/body80/r0,15188 |84.80%|12.93%|26/28|28/28|
| Actor/body80/r1,15188 |74.60%|0.662%|15/28|28/28|

The shirt band is±0.02 prepared-world units around the train-triangulated
surface, not a claimed metric uncertainty bound. Remaining mass can terminate
after the band or remain transmitted. In particular, actor-only does not simply
move all early termination onto the shirt.

Independent prepared OpenGL camera rays match actual model rays to maximum
origin error2.78e-8 and direction error7.86e-8. Witness-to-ray perpendicular
error is at most0.000498, consistent with rounding the source observation to a
pixel center. A coordinate-frame mismatch does not explain the large depth gap.
The render-only loader explicitly sets FR1 through75940 and FR0.3 afterward.

[Matched depth comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/matched_shirt_depth_comparison.png),
[source lettering witnesses](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/full_baseline_r0_006000/missed_upper_back_witnesses.png),
[6000 detailed traces](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/full_baseline_r0_006000/density_trace_comparison.png).

## Insights and next steps

- One-cell dilation repairs sampled witness admission but does not establish
  correct geometry. It can also reveal additional fog in the learned field.
- The matched15188 results justify the conditional stronger-distortion
  experiment. Compactness alone may collapse density onto a wrong front
  surface; repeat the independent shirt-depth diagnostic after that experiment.
- A low masked LPIPS does not certify correct depth or a perfect occupancy grid.
  Do not promote arbitrary learned empty cells into known physical free space.
- These are early-checkpoint results, not conclusions about the mandatory full
  leader trajectory or the final winner. Broader healthy controls, additional
  body regions and final-checkpoint inspection remain necessary.
- The separated-layer heuristic is explicit:96 depth bins, Gaussian sigma1-bin
  smoothing, peak separation8 bins and at least10% termination mass in each
  peak's±3-bin neighborhood. It is a supplementary diagnostic trigger, not an
  independent geometry certificate.

An apparent missing HC grey front shoe was separately investigated in native
HC/DC/KA RGB and temporal frames1660/1720/1780/1840/1900. The excluded dark object
stays at the same floor position after the actor moves away; it is a separate
floor bag/weight. No mask correction is warranted for that object.
[Temporal evidence](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/queries/HC_foot_temporal.png),
[native multiview evidence](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/queries/grey_foot_multiview_native.png).

Raw arrays, receipts, grid atlases, coordinate checks and immutable audit copies:
`/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy`.
GPU replays ran sequentially on otherwise idle silver-sable GPU7; it was released
afterward. No training process was interrupted.
