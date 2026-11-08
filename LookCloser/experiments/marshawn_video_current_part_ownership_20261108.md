# Marshawn: current-image anatomy and actor ownership

## What was tested

The all-camera contour admission check rejected source frame 1705 before
training. We traced the failing rays back to the original native photographs,
the current neural segmentation components, and the immutable pre-semantic
masks. An independent CPU audit also checked ray/box intersection and every
admitted cell along the failing rays.

## Results

Only GC/black failed: 48 base-grid, 44 padded-grid, and four mapped-grid
contour misses out of 152922 tested contours. No failing mapped ray was outside
the fixed AABB. The four rays crossed cells with independent clear-camera
evidence; changing the box or adding occupancy would conceal a source error.

The current `hand` and `hands` predictions agree on a seated crew member's
hand. Their agreement identifies anatomy, not one of the two target actors.
Prior complete-actor optical-flow support incorrectly admitted this newly
disoccluded hand. The same false component first appears at 1695 and persists
at 1700 and 1705. Current-photograph review of the preceding GC frames found
no corresponding false hand through 1690; apparent overlap at 1690 belongs
to foreground actor hair.

| Source frame | Wrong current-NN hand pixels retracted in GC |
| --- | ---: |
| 1695 | 4334 |
| 1700 | 4380 |
| 1705 | 4344 |

Original published sources and model artifacts remain preserved. The old
1695 model is withheld from delivery despite completing 140000 iterations;
the old 1700 model was stopped gracefully and retained its checkpoints.
Both require independent cold reruns after corrected-source qualification.
No 1705 model had started. Seven preceding completed models remain eligible
subject to final sequence review.

Corrections, native before/after contacts, and fresh source QA are under
`/fsx/tmp/marshawn_video_source_gcrole_g1_20261108/operations/`.
Root personally inspected `GC_corrected_native_visual/ALL3_GC_native_correction.jpg`;
the exact component removals preserve visible actor and braid boundaries.
`ROOT_finite_GC_current_photo_visual_review.json` binds this observation to
the correction and contact hashes. It does not waive fresh grid checks.

Cold replacements use the separate model campaign
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_gcrole_g1_20261108`.
The versioned fleet uses three physical hosts and ten GPU slots; old fleet
CPU processes remain held because their in-memory assignments are stale.
The new dispatcher must bind each frame's source and model generation and
require current source, sparse-grid, and all-camera dense-grid receipts.

Independent native-photograph checks confirmed that detached MD1695 boots,
FB1705 top-border hands, and GD1705 boot pixels are actual actor parts.
Consequently, rejecting every component lacking current body-mask overlap
would remove true detail. Further ambiguous components and a general role
qualification rule are still under investigation.

Corrected1695 passed143535 dense contour rays in all three admission forms;
corrected1705 passed152874, both with zero misses. The latter removes the
48 false hand-boundary rays responsible for its old failure. Both frames
have now started genuine cold140000-step models; actual audits show step0,
zero occupancy/EMA/Adam, no resume and66 TRAIN cameras. Seven preceding
models remain complete. These are operational counts, not a quality claim.

Native review of NC1700's8390-pixel dark fragment could not establish actor
or crew ownership. Its unsupported foreground assertion was withdrawn while
preserving physical UNKNOWN and the existing confidence falloff:95 covered
HD pixels retain positive W64, with total mass4.78188. A separate198-pixel
HE1700 fragment also lacks proven target identity; its proposed UNKNOWN
treatment retains positive W64 on all37 covered HD pixels (mass7.600524).
That input revision is being rebuilt before any1700 model consumes it.
Original source markers, arrays and CPU admission controls are archived;
no1700 model or assignment exists for the retired revision. The three changed
cameras require63-camera invariance and all66 RGB equality checks.

An independent audit reproduced all2097152 cells and the fresh26-neighbor
pad1 array exactly, with zero mismatches. It found35860 admitted cells with
no W64-positive projection; actual source footprints explain their UNKNOWN
protection. This is conservative sampling allowance, not proof that these
cells contain objects. No justified safe carve was established. Evidence:
`operations/independent_grid_review_g1_1695_20261108/report.md` under the
corrected source campaign.

For repeatable manual inspection, `scripts/export_marshawn_admission_slices.py`
verifies the manifest-bound grid payload and exports every orthogonal slice.
Gray denotes base allowance, blue the extra pad1 allowance, and black cells
excluded from admission. The helper changes no source arrays and makes no
physical occupancy claim.

## Insights

Anatomy confidence and optical flow cannot independently establish actor
identity after disocclusion. Preserve current-image evidence, test both
confirmed crew negatives and detached actor positives, and qualify any
replacement ownership rule before using it on later sources. Recompute
affected guidance from corrected masks; do not relabel stale inference as
a new run or reuse learned model state after changing its inputs.

### First continuous-source frame1710: retain a real trouser fragment

The released role filter kept107/110 current anatomy-pair proposals and left
three proposals UNKNOWN. ROOT inspected the actual small contacts, then the
[full JE photograph with scope](/fsx/tmp/marshawn_video_source_gcrole_g1_20261108/operations/production_role_review_v139/frame001710/JE_full_source_ROLEUNKNOWN_context.jpg)
and its wider native crop. The12226-pixel pants proposal is the black fighter's
right trouser leg, behind the crouching grey fighter and above the foreground
worker. The tight crop alone did not establish that identity.

The baseline64-native-pixel falloff had retained partial RGB supervision:
829/863 covered validHDpixels were positive, with coverage-weighted mean
W64=.450386. A finite, ROOT-photo-qualified correction used exactly the existing
current NN intersection: pants/dark-pants scores.773438/.789063, IoU.996989,
independent current CREW overlap0. It added those12226 pixels to actor/H and
uncertain ownership, preserving physical UNKNOWN, grey labels and all prior
actors. No new hand-drawn mask, prior bitmap foreground, threshold change or
additional inference was used. The simulated meanW64 becomes.982653; this is
weight accounting, not a demonstrated rendered-quality gain.

The actual application passed exact expected-array comparisons. All other65
native archives, all66 RGB images and the original NN qualification remained
unchanged. The separate JE74-pixel and DE126-pixel warnings remain UNKNOWN;
JE74 has zero overlap with the approved trouser scope. Historical role/ordered
receipts were archived, and current receipts record the explicit finite
positive extension. [Application receipt](/fsx/tmp/marshawn_video_source_gcrole_g1_20261108/operations/1710_JE12226_current_positive/application_complete.json)
SHA256 `a787525cd251ab6e1dcfd4e4e107afa89507b87c1ac00314851f811e67752745`.
New JE native SHA256
`f460835c4393e0133033f3ca9fb1ff060557e0a89b2a63faabadb969818731b7`.

This happened before SOURCE publication or any1710 field training. A detached
transaction resumed the exact source controller/watchdog after its PASS;
fresh hair admission, all66 guards, DATA/source QA and the dispatcher's fresh
grid/contour gates still precede the cold model. V139 code and the immutable
continuous-generation authorization remain unchanged.

Missing prior same-family evidence does not establish nonactor identity.
The conservative UNKNOWN path avoided a hard RGB cutoff in this case; full
camera context provided the missing positive role evidence. Future warning
reviews must retain this distinction.

### Frame1710 DE mixed instance: current anatomy overrides an old crew ID

The subsequent all66 guard passed65 cameras and stopped on DE instance1.
Its historical stable crew ID did not describe the current prediction: the
native connected components contain parts of both fighters. ROOT reviewed
[the full source and wide native component outlines](/fsx/tmp/marshawn_video_source_gcrole_g1_20261108/operations/1710_DE1_readonly/DE_two_actors_wide_native_outline.jpg)
and all five component contacts. Component1 is the black fighter's forearm
across the grey fighter's upper back; component3 is his right trouser leg,
continuing to the already qualified component5 boot; component4 is the grey
fighter's knee fold. Tight crops initially made the dark regions ambiguous;
the full two-person anatomy establishes their role.

The exact current-NN components1/3/4 contain94310 pixels, of which2322 were
already actor/H and91988 were missing. Independent current CREW overlap is
zero even when testing the whole qualified CREW instances, not only their
smaller cores. The finite application added only that91988-pixel residual
to actor/H and uncertain ownership. Geometry UNKNOWN, all prior actors,
the existing25390-pixel boot component, all other65 native archives including
the JE correction, and all66 RGB images remained unchanged. The tiny separate
component2 remains68 pixels of UNKNOWN/veto with no actor/H/U assignment.
The DE126 role warning lies entirely inside the proven actor knee component
and is explicitly superseded; original NN decisions remain archived unchanged.

[Actual application receipt](/fsx/tmp/marshawn_video_source_gcrole_g1_20261108/operations/1710_DE_components_current_positive/application_complete.json)
SHA256 `60b892f2651985aa9c7354d5d35a4de1b09817c469a74093f2399906ebf21770`;
new DE native SHA256
`1a3f36a33891bb0d3e249e77fef86db88d947f80ae3bd2b603ef3a2a5141aab5`.
This restores121 deep RAW-PRE pixels through a current-photo-proven actor
positive; no general PRE guard waiver was introduced. Original code/release,
the continuous source authorization and the140k model recipe remain unchanged.
The detached transaction resumed the source parents after its PASS. Fresh
hair admission, all66 guards, frozen DATA/source QA and admission/contour
checks remain required before the first1710 cold model.

Coordinate lesson: `people.bbox_coarse` uses512×270, whereas component-check
boxes use1024×540. Native diagnostics must use the actual bitmap or the
correct declared scale. Bounding boxes and an old instance ID alone cannot
establish actor or crew ownership in an occluding fight.
