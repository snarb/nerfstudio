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

## Native identity regressions found during production preparation

The first all-camera visual gate caught a real error in MC1665: the selected grey-fighter instance included the bald crew member's forehead and leather shoulder. The original all-person classifier treated a person overlapping the nominal actors as ambiguous, which did not veto the contamination. Matching the correct whole actor is insufficient to establish correct ownership of every pixel.

A new current-image native 1008-pixel interactive ROI, with crew-positive and fighter-negative prompts, provided clean crew candidates 0 and 2. Their source-reviewed union removed the crew contamination while preserving the grey coat/face and black actor's arm. The global crew proposal was rejected because it also contained the real grey actor's leg. All 65 other cameras' semantic channels stayed bitwise unchanged, RGB/frequency inputs were preserved, and the original winning frame1660 data was untouched. Before/after proof and hashes are in the campaign's `operations/crew1665` and `frames/001665/crew_inputs` directories.

MC1670 demonstrates a different branch: current global crew instance18 was already excluded by the canonical veto (64,019/64,019 pixels, zero actor overlap). A new native head segmentation confirmed identity. Its body candidate incorrectly included the grey coat and was rejected; no actor or veto pixel was changed. A correct existing mask must not be replaced by a poorer new candidate merely to force a tracker update.

JE1670 exposed the opposite failure: selecting only the highest previous-IoU fighter instance had omitted both visible black trouser-leg components. Fresh current person10 supplied components3 and4, separated from the torso by the foreground grey fighter. Full-source visual review established both as the black fighter's legs. Add-only recovery supplied 66,177 and38,685 previously unlabeled native black pixels respectively; all existing grey/black pixels, crew veto, and independent UNKNOWN were preserved. No hidden occlusion gap was filled.

The proposed autonomous fragment rule requires at least95% previous and current agreement on the same fighter's upper/head anchor, an already established actor identity, a lower-body component and direct same-fighter or explicit current other-fighter occlusion connection. Independent CREW/UNKNOWN conflicts take priority. In the initial positive regression it proposed the two real JE legs; the MC1665 bald-crew counterexample was rejected (only51.19% previous grey-head agreement). The initial source-context proof is mandatory evidence; head agreement alone does not establish arbitrary new body ownership.

## Operational release boundaries

The production scientific code tree is immutable. The corrected central producer and the later identity/fragment guards live under the campaign's `operations` directory with separate releases. The original producer was stopped after the MC finding. It must not be resumed unchanged: future preparation must pass the all-person guard, then publish `native_semantic_ready.json` only. The visibility dispatcher separately finishes all66 frame-local frequency maps, seals the inventory, builds the visibility-safe admission gate and publishes training dispatch.

Future frames1675–2015 have one exclusive durable RAW/native source decoder; the producer waits for its hash-bound `sources/<frame>/source_ready.json` and does not decode the same files concurrently. Per-camera LOCAL/SG segmentation assists preserve canonical detector/refinement math and save all current person proposals for identity review. Segmentation identity references may cross frames; final foreground pixels, frequency maps, admission projections, NeRF weights and learned occupancy do not.

A separate admission issue was identified before releasing the next model: the old winning source contract marked mat/bag occlusion UNKNOWN and disabled clear-depth votes for CE/JE/NE. Fresh person masks alone omit these occluders and can incorrectly carve an actor behind the thick mat. Current-image mat/bag qualification and the same conservative camera-depth restrictions are therefore required before admission/dispatch. This layer changes admission visibility only, preserving RGB, fighter labels, soft W64 and hybrid source eligibility. It is not a physical-perfect occupancy certificate.

### Released native preparation canary (1670)

ROOT visually approved the final JE/MC/HC source, actor-label and W64 contact, full-source JE leg context, and native before/after proofs. The operational V3 canary then exited successfully and published only `frames/001670/native_semantic_ready.json` (SHA `e0f8e52ac5ef3e2c92c943fba3c14c3df626fb34fed8cd7a1251ad25d86ea61b`). It did not publish a field dispatch marker.

The canary checks all66 native archive/metadata/current-person hashes, current guard bindings, frozen DATA mask-source bindings, RGB bytes and semantic bytes before publication. Initial prepared SHA is `127864f70918b5b49650151db07798cc78800fe803c799792f85257a43dbeb92`; initial ready SHA is `bd5373b129ab7f0d41b43bd9e45275135df8e5980f76dcd8947bcd51c40b4812`. The latter is expected to receive a separately recorded frequency-inventory seal before admission. The stronger guard protects all previously verified other people as crew, rather than only the manually named MC helper. The two qualified current component archives have explicit SHA bindings; older proof equality was also checked against the exact connected components of the hashed current NN output.

A proven actor-positive fragment may coexist with its own old proposal's UNKNOWN bitmap, which remains conservative geometry evidence. This does not waive independent CREW or unknown-person conflicts. The frozen hybrid source eligibility currently rejects UNKNOWN pixels; such proven actor fragments can consequently use learned rendering instead of pixel transfer. Do not clear geometry UNKNOWN globally to change that behavior.

The full chronological V3 watchdog was launched after the LOCAL/SG observer canaries passed. Its operational release SHA is `226119430a0db3923926d1652c7d228e7e9017ff9eaef9fb24de057e33afc4d6`; it binds all operational helpers, the fragment policy, the native-assist release and all of its helper SHAs, the exclusive source ownership/configuration, and the original frozen preparation inputs. Initial watchdog/producer PIDs were2248938/2248940. The first future frame1675 started a real native worker on LOCAL GPU0 (PID2248986, verified physical UUID/global lease, no initial OOM evidence) with14 LOCAL and52 silver-sable views across initially idle GPUs1/5/6/7. Sources are not decoded again. The producer publishes frame-local native semantics only; the separately released visibility dispatcher owns frequency completion, admission and field dispatch. An unresolved independent person/crew conflict remains a blocking correctness gate rather than being silently accepted.

An exact24-file copy of the released V3 operational code, native-assist helpers/releases, source ownership/configuration, policy and actual-source regression evidence is archived at `operations/releases/v3_226119430a0db392` under the campaign root, with per-file SHA manifest. This preserves the operational version independently of any later supervised fix; the scientific code remains in the original immutable code snapshot.

### V4 scheduling correction

The first future frame revealed an operational throughput bug: the native assist treated 60 seconds of total frame age as a timeout for every camera not yet dispatched. A healthy52-view, four-GPU queue therefore completed8 remote views and assigned44 remaining views to LOCAL fallback. Those44 were not completed when the transport summary was written; the outer helper correctly continued running them before semantic publication. This was not a segmentation or CUDA failure.

The versioned V4 native controller instead times a genuine interval with no own active/dispatching/unknown/orphaned or uncollected work/progress. Normal multiwave queues remain queued; externally unavailable idle capacity can eventually fall back without duplicating an already claimed launch. The CPU regression passed52-view waves beyond60 seconds, real idle fallback, and unknown/orphaned/claimed-job protection.

Only the exact CPU outer watchdog/producer were replaced, with PID/start/boot identity receipts. The existing1675 native helper and its LOCAL fallback workers survived and were adopted; no current matte was regenerated or overwritten. Scientific files and field jobs were untouched. The new producer release SHA is `e1064fff6acaa63c7e457545dc96225f829e97b6f9760ebdba631796ed8e9e21`, and native V4 release SHA is `4f1b2bd1d134134e6dba3b9198c1a9a7e51e0730e196c80de4c2cce42bf9f873`. Initial V4 watchdog/producer PIDs were2251915/2251917. The unchanged CPU programs retain V3 filenames and take the V4 release path.

Exact qualified operational helper copies are now in `scripts/marshawn_video_ops`, with per-file source SHA metadata. The V4 entrypoint/controller and CPU regression are in the parent scripts directory. The initial35-view1675 diagnostic passed without pixel changes, and explicitly did not publish an all66 pass marker; full-frame publication remains gated on the complete current inputs.
