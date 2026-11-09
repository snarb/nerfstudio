# Two-actor AABB teacher sanity

What was tested: immutable full-scene R1 selected20k teacher, fd56e541…88fa2, with only native ray-near/far clipped to the exact reviewed subject.loose box. The original full-scene field normalization, learned weights, radius1 mask and EMA remain unchanged. Entry/exit preserve the.052+n×.003 marching phase. No NN training, semantic masking or source compositing occurs.

Results: all3eval1920×1012 renders retain the central actor pair and remove background people/rig outside the box. Floor, mat and detached black prop inside the box remain, so this is not clean semantic isolation. Live model state hash9d5791d6…17366 matches before/after; EMA/grid/checkpoint match. Nine paired native rays agree in retained sample starts within2.146e−6world. Raw premultiplied RGB+floatalpha are kept; straight RGBA uses the verified black forward-BG contract and C/alpha, with reconstruction error≤2.981e−8. Alpha is teacher volume opacity, not confidence or ground truth. No quality metrics are computed.

Insights: an AABB alone cannot remove scene surfaces that share the actors' volume. This full-scene teacher sanity does not substitute for the new actor-only/soft-confidence training experiment. LOOSE is the only rendered variant; MEDIUM remains unauthorized pending root review.

[Report and RGBA contract](/fsx/tmp/marshawn_001660_hd/two_actor_isolation/teacher_R1_20k_LOOSE_clip_SANITY/README.md), [DC](/fsx/tmp/marshawn_001660_hd/two_actor_isolation/teacher_R1_20k_LOOSE_clip_SANITY/DC_premult_black.png), [HC](/fsx/tmp/marshawn_001660_hd/two_actor_isolation/teacher_R1_20k_LOOSE_clip_SANITY/HC_premult_black.png), [KA comparison](/fsx/tmp/marshawn_001660_hd/two_actor_isolation/teacher_R1_20k_LOOSE_clip_SANITY/KA_local_comparison.jpg).
