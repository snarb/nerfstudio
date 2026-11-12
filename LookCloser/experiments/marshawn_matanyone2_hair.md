# Marshawn: MatAnyone 2 hair experiment, 2026-11-12

## What was tested

**Decision: do not replace the delivered SAM matte with whole-frame MatAnyone 2.** Soft prompting improved an early hair gap and the outer halo, but incorrectly removed dark clothing below the actor's left shoulder. The experiment is complete; no training, model checkpoint, final render, or training environment was changed.

Official sources: [MatAnyone 2 repository](https://github.com/pq-yang/MatAnyone2), [CVPR 2026 project](https://pq-yang.github.io/projects/MatAnyone2/), and [official v1.0.0 weights](https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth). This is MatAnyone **2**, not the CVPR 2025 predecessor.

- Repository commit: `0079197acd6d16a741f71558809c06c586c579e0`.
- Checkpoint SHA256: `5e9821e4087231427376b437c85bb6e072b41e582314f06fd524f75bc4af5914`.
- Machine: `silver-sable`, GPU7, NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB.
- Isolated environment: Python 3.11.18, torch 2.11.0+cu128, torchvision 0.26.0+cu128. Minimal inference dependencies; GUI/training dependencies were not installed. Exact versions are in the remote `requirements_inference_frozen.txt`.
- Experiment root: `/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/operations/matanyone2_hair_20261112` (called `E` below).
- Inputs: `../flying_strict_20261112/gt_sources/{001251..001267}_rgb.png`, native BC crop 2120×1450, original unmasked RGB. The original 8K-derived SAM soft alpha at frame1251 supplied the prompt. Both actors are included in one foreground object.
- Source rate: 120 fps. Ascending frames1251–1267 only; 1268–1269 excluded because the scene ends at1267. Ten repeated first-frame warmup inputs, followed by17 real frames. The official first-frame prediction reset is retained through the first real output; subsequent frames propagate temporal memory. No resizing, float16 autocast, default model config.

Input-range audit: the official `inference_matanyone2.py` passes a float first-frame mask in **0–255** to `processor.step`; `InferenceCore.step` divides it by255 in its matting branch. This runner uses the same0–255 convention. RGB is divided by255 exactly once. The body hole is therefore not explained by a0–1/0–255 integration mismatch. The prompt difference image is calculated in signed int16 before conversion to uint8 to avoid subtraction wraparound.

Three prompt variants used identical weights and RGB:

| Variant | First-frame prompt |
| --- | --- |
| `native_official_close10` | Official dilation10 then erosion10; these helpers turn the soft mask into a binary prompt |
| `native_soft_prompt` | Original SAM 0–255 soft values, no morphology or thresholding |
| `native_corrected_close10` | Official binary prompt, then21 pixels manually set to0 in a small gray core of the early gap |

The correction polygon in native crop coordinates is `[(1317,546),(1320,548),(1323,553),(1319,551)]`. It deliberately does not trace the entire diagonal strip, which may include real hair. `native_corrected_close10/` retains `original_sam_prompt.png`, `unedited_close10_prompt.png`, `first_prompt.png`, `prompt_diff.png`, and `correction.json`. No source image or source mask was edited.

## Results

| Variant | Mean inference loop/frame | Complete run incl. loading/warmup/outputs | Peak allocated GPU | Peak reserved GPU | Observation |
| --- | ---: | ---: | ---: | ---: | --- |
| Official close10 | 0.112 s | 15.05 s | 9.34 GiB | 21.63 GiB | Cleaner outer halo, but early internal hair gap stays filled |
| Soft prompt | 0.111 s | 12.92 s | 9.34 GiB | 21.63 GiB | Opens early gap; removes real dark clothing elsewhere |
| Corrected close10 | 0.110 s | 12.95 s | 9.34 GiB | 21.63 GiB | Small prompt correction is not retained as the desired gap; visually close to uncorrected binary run |

Timing includes per-frame PNG read, host-to-device transfer, prediction, alpha transfer, and synchronization, but excludes per-frame image/video output. The first run also downloaded ResNet initialization weights. This is approximately9 processed frames/s, **not** realtime120fps. There was no OOM. All inference workers finished; GPU7 was at0 MiB/0% after the experiment.

SAM is a comparison baseline, not verified alpha ground truth. PSNR, SSIM, and LPIPS would measure agreement with that imperfect baseline rather than matte correctness, so no quality scores are claimed. The clip spans only0.142 s of source time; it cannot establish general freedom from flicker.

Review evidence:

- [All four variants, hair/shoulder temporal contact](../lookcloser_debug_outputs/matanyone2_hair_20261112/hair_temporal_all_variants.jpg).
- [Early dread-gap close-up](../lookcloser_debug_outputs/matanyone2_hair_20261112/dread_gap_variants_1251.png).
- [Both actors: original RGB / SAM / soft MatAnyone 2](../lookcloser_debug_outputs/matanyone2_hair_20261112/both_people_source_sam_soft.jpg).
- [Body-hole rejection evidence, frame1267: source / SAM / soft](../lookcloser_debug_outputs/matanyone2_hair_20261112/body_hole_1267.png). ROI `(880,590,1180,1050)` shows clothing replaced by black, not simply background removed.
- Review video:17 source frames at24fps, explicitly labeled5× slow, looped4×; full video is on Drive and Silver (`E/review/`).
- [Shared MatAnyone2 comparison video](https://drive.google.com/file/d/1vxdpxnjRqP6l1TfPBMaLpjOqN54n9Nls/view).
- [Delivered main flight, without MatAnyone2](https://drive.google.com/file/d/1kA9tJue9g9nbU0xCFHEtBmth5ZIkXvdq/view).

All alpha outputs are 8-bit grayscale PNGs, native2120×1450: `E/<variant>/alpha/{001251..001267}.png`. Black composites and original SAM composites are under `black/` and `baseline_black/`; per-frame measurements and provenance are in `receipt.json`. Each variant also includes an original-rate120fps comparison and a12fps review. The cross-variant24fps videos and contacts are in `E/review/`.

## Insights

1. **Prompt preprocessing changes the result materially.** Default morphology closes narrow gaps that the soft prompt can open. A tiny21-pixel manual correction after closing was insufficient; enlarging it without reliable foreground/background evidence would risk deleting hair.
2. **Whole-frame soft MatAnyone2 fails the clothing gate.** Both people remain recognizable in full thumbnails, but this is insufficient validation: a substantial wedge of dark sleeve/chest is removed from the foreground. The original native RGB confirms clothing in that region. The delivered temporally stabilized SAM tail remains unchanged.
3. The soft result is useful as a separate hair-region research comparison only. A future head-region combination would still need close-up validation and temporal checks. No such combination was applied here. Thin dark strands and motion blur remain ambiguous; the test does not prove that every newly black pixel is correct.
4. No isolated bat was visible in the evaluated native crop, so its preservation is not established. This is not an endorsement for other views or earlier action frames.
5. Matting cannot restore raised hair already clipped by the NeRF AABB. This native-photo test does not solve missing NeRF geometry and was not applied to the neural frames.

## Reproduction

The standalone [inference script](../scripts/marshawn_sequence/matanyone2_hair_test.py) is copied to `E/matanyone2_hair_test.py`. `E/make_review.py` creates the cross-variant contacts/video. The original official checkout remains unchanged.

```bash
E=/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/operations/matanyone2_hair_20261112
cd "$E"
export UV_CACHE_DIR="$E/uv_cache" TMPDIR="$E/tmp"
./uv-x86_64-unknown-linux-gnu/uv venv venv
./uv-x86_64-unknown-linux-gnu/uv pip install --python venv/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
./uv-x86_64-unknown-linux-gnu/uv pip install --python venv/bin/python numpy pillow opencv-python scipy tqdm einops hydra-core imageio imageio-ffmpeg huggingface_hub==0.36.2 safetensors kornia requests
./uv-x86_64-unknown-linux-gnu/uv pip install --python venv/bin/python --no-deps -e MatAnyone2

CUDA_VISIBLE_DEVICES=7 TORCH_HOME="$E/torch_cache" HF_HOME="$E/hf_cache" \
  venv/bin/python -u matanyone2_hair_test.py \
  --source ../flying_strict_20261112/gt_sources \
  --checkpoint MatAnyone2/pretrained_models/matanyone2.pth \
  --out native_soft_prompt --prompt soft > inference_soft.log 2>&1
```

For the other variants change `--prompt` and `--out` to `official_close10`/`native_official_close10` or `corrected_close10`/`native_corrected_close10`. For exact dependency reproduction use the frozen environment file. Initial system `python3 -m venv` failed because ensurepip was absent; isolated uv creation worked. The first installation cache hit a space error under `/home/ubuntu/.cache/uv`; assigning experiment-local `UV_CACHE_DIR` and `TMPDIR` resolved it. Neither issue invalidated inference. The existing nerfstudio/NeRF environments were untouched.

## User-selected delivery

The user subsequently preferred the MatAnyone2 soft-prompt appearance and explicitly
requested the complete flight with that variant. Published84-frame3.5sFullHD24fps
video: https://drive.google.com/file/d/1FUtvZFMSrzmXwKKuWqW-vdu5JTPtbgxQ/view .
The70strict NeRF frames are retained; all14BC photographic tail frames now use
the native MatAnyone2 soft alpha, without manual corrections or SAM temporal median.
Silver output: `operations/flying_matanyone2_20261112/`.
SHA256: `c5f8fe52b9f757ffa7d7538b621ffb69c409c9a93b3e19e721037743d8db9b14`.
All84encoded frames decoded successfully. The dark region interpretation remains
visually ambiguous without a trusted semantic annotation; earlier clothing rejection
was a conservative interpretation, not a ground-truth segmentation measurement.
