# Marshawn: actor-metric review and checkpoint selection

The user prioritizes fixed-mask actor LPIPS and visible artifact removal.
`summarize_marshawn_actor_candidates.py` therefore reports each candidate's
raw minimum archived V1 LPIPS, with PSNR breaking exact ties. It verifies the
same six actor/view crops, mask pixel counts, metric domain and equal-weight
aggregation before comparison. It reads metrics only and changes no training,
historical selector, source image, render, checkpoint or stopping tracker.

The historical worker selector treats LPIPS differences within0.001 as a tie
and prefers PSNR. This can hide meaningful progress at the current score scale.
For cold distortion0.1, its selected91128 checkpoint has LPIPS.02228724 and
PSNR31.652626, whereas raw-min/terminal126316 has LPIPS.02135285 and PSNR31.65173:
4.19% better LPIPS for less than0.001dB PSNR difference. Both checkpoints remain
retained and labelled; this review does not retroactively relabel the runner's
selection. The same distinction applies to every other candidate.

The first source-bound11-candidate report is
[the provisional table](/fsx/tmp/marshawn_001660_hd/tuning_20261106/actor_candidate_review_20261107_0123/report.md)
and [complete inputs and per-view/actor measurements](/fsx/tmp/marshawn_001660_hd/tuning_20261106/actor_candidate_review_20261107_0123/report.json).
At that observation, half-step91128 has LPIPS.02001403, quarter-step30376
.02033603 and focused-frequency45564 .02112244. Several models are still
running and their ages differ. These are an image-metric shortlist, not a
claim of completed optimization or clean geometry.

Final review must also inspect the source/novel-view hair and shoes, exterior
opacity, actual schema2 core/admission/active snapshots, and reserved source-ray
termination depth and thickness. A better masked LPIPS alone cannot promote a
candidate with false density, missing actor geometry or a contaminated exterior.
No loss numbers are included in this report.
