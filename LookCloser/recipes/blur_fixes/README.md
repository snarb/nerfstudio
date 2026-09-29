# Fresh blur validation recipes

Standard LookCloser defaults are unchanged. Canonical scaling is scoped to the
lipstick recipes: its fight identity run missed the SSIM gate by .00103. Use the
original fight settings or the separately validated `fight_exp_sh.json` there.
The preferred full-frame lipstick recipe is `lipstick_room_exp_sh.json`; the
canonical version is a simpler alternative with more background artifacts.

These requests run with `scripts/run_blur_experiment.py` on the cleaned field
implementation. Seed 42 is fixed. Each run starts from random weights; no model
or optimizer checkpoint is loaded. See
[the experiment report](../../experiments/blur_ablation_fresh.md) for selection,
metrics, cropped images and dataset provenance.

| Request | Scope | Validation status |
|---|---|---|
| `lipstick_actor.json` | Historical masked actor, small AABB | Completed 24k pair |
| `lipstick_room_canonical.json` | All RGB, tight room AABB, canonical softplus | Completed 24k control; selected 16k |
| `lipstick_room_exp_sh.json` | All RGB, tight room AABB, safe exponential and corrected SH | Completed 24k room pair and 30376 fight transfer |
| `fight_exp_sh.json` | Original 66/3 fight, safe exponential and corrected SH | Completed 30376 transfer; all limits pass |

The bounds in the lipstick requests belong to the frozen DEC5 coordinate system.
They are not general defaults for another scene. The actor recipe supervises
valid mask pixels only; its background is unsupervised. The room requests
supervise the complete RGB images.

`canonical_aabb` means density multiplied by 3/max(AABB side lengths), preserving
the original fight scene's span-three reference. Exponential casts logits to
FP32 before its bias and activation; this does not make the entire network FP32.
The research runner explicitly defaults to legacy unnormalized density, so
requests remain independent of future interactive method-preset changes.

## Run

Use the installed environment's Python:
`/home/brans/repos/nerfstudio/.venv/bin/python`.
The JSON files record the prepared local datasets, frozen ROIs and new output
folders. Review those paths and choose an unused output folder before a rerun.
Existing histories cannot be overwritten.

For process/GPU/budget supervision, put a JSON list containing the absolute
request filename in the **parent directory of its output folder**, then run:

```bash
/home/brans/repos/nerfstudio/.venv/bin/python scripts/supervise_blur_campaign.py /path/to/output-parent/manifest.json
```

The supervisor resolves runtime code from this repository and logs every 30 seconds.
Keep it supervised through evaluation and the final visual review. It caps
training at 22 active GPU-hours across sibling runs, reserving 2 hours for evaluation.
Selected native images and crops are saved in `eval_<step>/`; `selection.json`
and `best.pt` identify the highest all-eval PSNR, with LPIPS tie-break within .07 dB.

Historical requests requiring removed ablation controls need branch
`lookcloser-blur-ablation-archive` (`7a6ffd5f`), which preserves the removed
experimental controls. Equivalent reference-three checkpoints load on the
cleaned implementation; checkpoints requiring different removed math fail
explicitly rather than silently changing their rendering.
