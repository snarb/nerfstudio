"""Package a completed native reference gate without changing image/alpha bytes."""
import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_only(value):
    if isinstance(value, dict):
        return {key: metric_only(item) for key, item in value.items()
                if "loss" not in key.lower()}
    if isinstance(value, list):
        return [metric_only(item) for item in value]
    return value


def main(args):
    production = args.production
    state_bytes = (production / "state.json").read_bytes()
    state = json.loads(state_bytes)
    selected = next(row for row in state["evaluations"] if row["step"] == args.step)
    evaluation_path = Path(selected["evaluation"])
    evaluation = json.loads(evaluation_path.read_text())
    assert sha(evaluation_path) == selected["evaluation_sha256"]
    assert sha(selected["checkpoint"]) == selected["checkpoint_sha256"]
    source = evaluation_path.parent
    output = production / "downloads" / ("native_reference_%09d_snapshot" % args.step)
    output.mkdir(parents=True, exist_ok=False)
    original_files = []

    def copy(path, relative):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        assert sha(path) == sha(destination)
        original_files.append({"source": str(path), "bundle_path": str(relative),
                               "sha256": sha(path)})

    for camera in ["DC", "HC", "KA"]:
        folder = source / "renders" / camera
        for filename in [camera + "_inside.png", "receipt.json", "raw_partition.npz"]:
            copy(folder / filename, Path("renders") / camera / filename)
    for path in sorted((source / "detail_review").iterdir()):
        if path.is_file():
            copy(path, Path("detail_review") / path.name)
    copy(source / "actor_comparison.jpg", Path("actor_comparison.jpg"))
    copy(source / "visual_review.json", Path("visual_review.json"))
    box = production.parent / "native_semantic_grid_20000/code/marshawn_1660_1900_floor.json"
    assert sha(box) == evaluation["box_sha256"]
    copy(box, Path("accepted_box.json"))
    (output / "metrics.json").write_text(json.dumps(metric_only(evaluation), indent=2) + "\n")
    (output / "README.txt").write_text(
        "Native geometry-supervised reference: immutable %d-step snapshot\n\n" % args.step +
        "This is NOT a cleaned actor asset. Floor, mat, stationary bag and other\n"
        "nonactor fragments remain inside the accepted box. Fine details remain\n"
        "softer than source photographs. No artifact-free claim is made.\n\n"
        "renders/{DC,HC,KA}/*_inside.png: original 1920x1012 accepted-box renders,\n"
        "copied byte-for-byte. Black exterior is the original render background.\n"
        "The with_original_alpha ZIP includes raw_partition.npz: original float\n"
        "RGB/opacity partitions, unchanged. The smaller review ZIP omits these\n"
        "arrays; their original paths and hashes remain in the manifest.\n"
        "detail_review: existing source | native20k | previous gate | this gate\n"
        "comparisons of hair, hands/contact and feet. No new crops or edits.\n\n"
        "metrics.json contains common FHD PSNR, SSIM and masked AlexNet LPIPS.\n"
        "V1 is the frozen primary protocol; V2 is a separate source-label audit.\n"
        "The fixed actor crops use identical neutral RGB0.5 exterior.\n"
        "The model uses original TRAIN SfM/MVS geometry supervision and nativeR1.\n"
        "No semantic gate is applied to these renders. Original field coordinates,\n"
        "checkpoint, pixels, alpha, masks and source datasets were not modified.\n\n"
        "This records the selected snapshot when packaged; later continuation\n"
        "may select a different step. Model weights are referenced, not bundled.\n"
    )
    manifest = dict(kind="IMMUTABLE_NATIVE_REFERENCE_SNAPSHOT_NOT_CLEANED_ACTOR_ASSET",
                    created_unix=time.time(), step=args.step,
                    selected_when_packaged=state["selection"]["step"] == args.step,
                    checkpoint=selected["checkpoint"], checkpoint_sha256=selected["checkpoint_sha256"],
                    source_evaluation=str(evaluation_path), source_evaluation_sha256=sha(evaluation_path),
                    state_at_packaging_sha256=hashlib.sha256(state_bytes).hexdigest(),
                    helper_sha256=sha(__file__), original_files=original_files,
                    pixels_alpha_masks_modified=False, model_weights_included=False,
                    limitations=["Persistent floor/mat/bag and other nonactor fragments.",
                                 "Geometry-supervised auxiliary reference, not a matched architecture-only control."])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "SHA256SUMS").write_text("".join(sha(path) + "  " + str(path.relative_to(output)) + "\n"
                                                for path in files))
    archives = []
    for suffix, include_raw in [("review", False), ("with_original_alpha", True)]:
        archive = output.with_name(output.name + "_" + suffix + ".zip")
        included = [path for path in files if include_raw or path.name != "raw_partition.npz"]
        checksums = "".join(sha(path) + "  " + str(path.relative_to(output)) + "\n"
                            for path in included)
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zipped:
            for path in included:
                zipped.write(path, output.name + "/" + str(path.relative_to(output)),
                             compress_type=zipfile.ZIP_STORED if path.suffix == ".npz" else None)
            zipped.writestr(output.name + "/SHA256SUMS", checksums)
        with zipfile.ZipFile(archive) as zipped:
            assert zipped.testzip() is None
            for line in checksums.splitlines():
                expected, relative = line.split("  ", 1)
                digest = hashlib.sha256()
                with zipped.open(output.name + "/" + relative) as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                assert digest.hexdigest() == expected
        archives.append(dict(path=str(archive), bytes=archive.stat().st_size,
                             sha256=sha(archive), includes_original_float_alpha=include_raw))
    receipt = dict(step=args.step, directory=str(output), archives=archives,
                   all_copies_byte_identical=True, ZIP_CRC_verified=True,
                   lightweight_archive_note="Review ZIP omits raw_partition.npz; full provenance lists their hashes.")
    output.with_suffix(".packaging.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    main(parser.parse_args())
