# Marshawn 3-second video preparation

Training sources are frames1660–2015 inclusive, step5:72 independently prepared frames at24fps from120fps. All66 physical cameras are TRAIN; DC/HC/KA are explicitly in-sample monitors. A later60fps extension uses step2 without renaming models or changing source frame IDs.

## Source and colour contract

The original RAW TIFF root supplies frames≤1900; the rebuilt1901–2020 root supplies the tail. Local TIFF ID=globalframe−1252. The inventory found all4752 requested camera/frame files. Qualified TIFFs stop at2020; there is no qualified two-second GT continuation after the requested three seconds.

Training PNGs use the frozen first-frame Log3G10/RWG→linearRec709 conversion, AREA prefilter before exactly one calibrated lens remap, gain4, frozen shared k4 radial field and fixed cameraEVs, Reinhard+sRGB. Former eval cameraEVs stay frozen. There is no per-frame exposure fit, PQ interpretation, or second rectification. A separate native distorted-raster image is exported for segmentation only; its simpler global frozen gain is never used as a training target.

## Frame-local masks and confidence

SAM3 checkpoint9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e is loaded from the existing local overlay. Every new source receives fresh person detections. Recent native labels identify the two fighters; they are never copied as final masks. Native1008×1008 interactive boundary tiles recover add-only detail within64 native pixels. Whole-image detection is internally resized; native-output dimensions alone are not described as native encoder accuracy.

Distant tiny disconnected detector islands are excluded from foreground and retained as UNKNOWN. The largest connected body, pieces within128 native pixels, sizeable disconnected body parts, and previous identity overlap are protected. This avoids training on a detector speck as though it were an actor. A concrete failure was167 spurious native pixels on the far foreground floor in2020HC, at x2368–2411,y4293–4298, unrelated to either fighter.

W32/W64 are recomputed from each frame's native RAW-raster masks and projected with the exact baseline AREA+single-remap footprint. Baseline body/core/band construction is preserved. Verified other-person pixels veto confidence; ambiguous identity is UNKNOWN. Missing/ambiguous detections never fabricate a foreground mask or certify absence. Data publication pauses unless a source-bound review establishes a known absence. Visibility is recorded per actor as present/absent/unknown; absent requires reviewed evidence and a null crop.

## Independent admission and model starts

Each frame recomputes its128³ data-derived admission from calibrated mask projections, requiring two complete clear camera footprints before rejecting a cell; UNKNOWN cannot provide clear evidence. Pad1 is one fresh, noncumulative26-neighbour dilation. Whole-cell mapping matches the selected recipe. This gate is distinct from learned occupancy, which starts cold with the model.

At1660 only, reviewed image-projection label INPUTS are portable under data/admission_inputs. Neither the old gate nor a learned grid is copied. The three former eval sources have W64 positive inputs and UNKNOWN clear visibility pending qualification. Every manifest, source image, semantic archive, transform, box and support source is hash bound. Missing portable inputs fail closed.

The first-frame prototype appends explicit UNKNOWN/veto arrays for hybrid source eligibility while preserving every original semantic array bitwise. All66 original baseline NPZ hashes and arrays were checked unchanged; the immutable winning baseline was not modified.

## Actual checks

Frame1665 HC/DC/KA native segmentation and HD preparation completed. Identity IoU against the first-frame labels was0.69–0.82 with clearly separated competing detections. The diagnostic three-view gate checked81280 DC,67898 HC and57676 KA actor pixels: zero AABB misses and zero base/pad1 ray misses. The independent compiled ray traversal crosscheck passed306 synthetic/boundary rays. This diagnostic is explicitly ineligible as a production ALL66 gate; ray intersection is necessary evidence, not proof of physical surface containment.

Tail1960/2020 uses independently detected, manually identified masks in HC/DC/KA/FC/MC/NA. The bald helper in a black leather jacket is crew; the grey fighter lies on the mat. Mat exclusion is therefore physically incompatible with retaining the actor. The six-view reconstruction found grey three-view SIFT anchors2/1 and black5/2 at1960/2020. The fitted floor's maximum residual is0.00464 prepared units; a−0.02 lower halfspace bounds known actor geometry, with additional box margin. Final box and source visuals are recorded in the campaign's reviewed immutable revision, not inferred from the obsolete three-view proxy.

## Durable execution

marshawn_video_prep_producer.py prepares frames chronologically on olive-lark and dispatches SHA-verified inputs to consumers. Sources→nativeSAM3→HDsemantics→verified remote staging are separate resumable stages. The shared GPU lease is inherited by the SAM3 child so an orphan still holds its lease. The producer verifies its release, scripts, source inventory and common box before work. Frequency fitting and model training remain separate, frame-local consumers. Unknown visibility or source mismatches block publication and are recorded in preparation_error.json.
