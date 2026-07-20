# Deterministic temporal EXR-to-leader-JPEG conversion

## What was tested

`scripts/convert_temporal_exr_to_leader_jpeg.py` freezes the accepted 007740 EXR color pipeline,
camera-to-train/eval filename mapping, exact Python package versions and FFmpeg encoder version. It
has a non-destructive `stage` command and a separately gated `apply` command. Staging and backups
must be children of the script-local `scripts/temp` directory; `apply` refuses frame 007740 and all
older frames, first runs as a dry-run, and requires an explicit overwrite confirmation plus
`--execute` before changing a dataset.

The committed file is an exact copy of the tested dev3 script. Its SHA-256 is
`4b055c239cf2f787cd05a579b13b42c84e59cf688098bc7ffaa77d33f69266e1`.

## Results

| Check | Result |
|---|---:|
| Canonical 007740 EXR rerun versus protected leader JPEGs | 69/69 byte-exact |
| Independent clean 007747 rerun versus installed 007747 JPEGs | 69/69 byte-exact |
| 007747 manifest hash mismatches | 0/69 |
| Final dimensions | 1920×1080 |
| Final chroma sampling | 4:2:2 |
| Quantization-table SHA-256 | `a412dffd7346a1fb47fd63bd5563df629b103fea55100fa4fc616c03ed6e4d15` |
| Encoder comment | `Lavc59.18.100` |

The independent 007747 check used a previously absent directory,
`/home/ubuntu/repos/red-to-exr/temp/reverify_007747_for_repo_commit`, reconverted all 69 EXRs, and
compared complete JPEG-file SHA-256 values to the installed dataset. This proves deterministic
reproduction of the installed 007747 outputs. It does not mean 007747 image bytes equal 007740;
the scene content is a different temporal frame.

## Exact recipe on dev3

FSX is available only on `dev3`. From a LookCloser checkout there, create the pinned environment:

```bash
cd /home/ubuntu/repos/nerfstudio/LookCloser
python3.12 -m venv scripts/.venv-leader-jpeg
scripts/.venv-leader-jpeg/bin/pip install \
  -r scripts/requirements-leader-jpeg.txt
```

Before converting future frames, create a fresh 007740 reference proof. This reads the protected
dataset but does not modify it:

```bash
scripts/.venv-leader-jpeg/bin/python \
  scripts/convert_temporal_exr_to_leader_jpeg.py stage \
  --frame 7740 \
  --staging-dir "$PWD/scripts/temp/reference_proof_007740_exact" \
  --workers 4
```

Stage a frame or a stride-7 range into a new temp directory:

```bash
scripts/.venv-leader-jpeg/bin/python \
  scripts/convert_temporal_exr_to_leader_jpeg.py stage \
  --start-frame 7747 --end-frame 8048 --stride 7 \
  --staging-dir "$PWD/scripts/temp/temporal_007747_008048_leader_jpeg" \
  --workers 4
```

Run `apply` without `--execute` first and review its frame/image counts and paths. Use a new backup
directory, then repeat with `--execute` only after that review:

```bash
scripts/.venv-leader-jpeg/bin/python \
  scripts/convert_temporal_exr_to_leader_jpeg.py apply \
  --staging-dir "$PWD/scripts/temp/temporal_007747_008048_leader_jpeg" \
  --reference-proof "$PWD/scripts/temp/reference_proof_007740_exact/conversion_manifest.json" \
  --backup-dir "$PWD/scripts/temp/backup_before_leader_jpeg_007747_008048" \
  --confirm-overwrite-after-007740
```

The executable apply is the same command with `--execute`. Never use
`--allow-version-mismatch` for canonical output: it is only a diagnostic escape hatch. Preserve the
conversion and backup manifests until the dataset has passed byte-hash and visual audits.

## Insights

- Reproducibility depends on the complete pipeline, not only the color formula: OpenEXR decoding,
  package versions, camera mapping, intermediate JPEG, and the exact FFmpeg/libavcodec build are
  all checked.
- JPEG replacement creates a new dataset revision. Frequency maps generated from earlier JPEG
  bytes are stale and must be regenerated and re-audited before training.
- The safe workflow is stage → hash/profile/visual audit → apply dry-run → verified backup → atomic
  apply → post-apply hash audit. Keep the canonical reference frame read-only throughout.
