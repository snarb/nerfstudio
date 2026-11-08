"""Compare private packed inputs against immutable, actually executed V5 tasks."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compare(baseline, candidate, configuration, snapshot_manifest):
    baseline, candidate = Path(baseline), Path(candidate)
    config = json.loads(Path(configuration).read_text())
    snapshot = json.loads(Path(snapshot_manifest).read_text())
    execution = json.loads((candidate / "packing_execution.json").read_text())
    records = json.loads((candidate / "packing_records.json").read_text())
    cameras = [row["camera"] for row in config["records"]]
    if [row["camera"] for row in records] != cameras or cameras != snapshot["cameras"]:
        raise AssertionError("Camera order changed")
    source_files_unchanged = all(sha(row["path"]) == row["sha256"] for row in snapshot["files"])
    if not source_files_unchanged:
        raise AssertionError("Immutable source snapshot binding changed")
    names = ("task.json", "reference.npz", "source_original_receipt.json",
             "source_portable_receipt.json", "union_resolver.py", "seed_policy_v5.json")
    results = []
    for old_record, new_record in zip(config["records"], records):
        camera = old_record["camera"]
        for key in ("original_native", "original_native_sha256", "camera", "host", "task_sha256"):
            if old_record[key] != new_record[key]:
                raise AssertionError("Factory record changed: %s/%s" % (camera, key))
        if list(new_record["variants"]) != ["silver", "olive"]:
            raise AssertionError("Variant order changed")
        for host, folder in (("silver", "tasks"), ("olive", "tasks_local")):
            old, new = baseline / folder / camera, candidate / folder / camera
            equality = {name: sha(old / name) == sha(new / name) for name in names}
            links_equal = (old / "raw_native.png").is_symlink() and (new / "raw_native.png").is_symlink()
            links_equal = links_equal and (old / "raw_native.png").readlink() == (new / "raw_native.png").readlink()
            old_spec, new_spec = (json.loads((p / "task.json").read_text()) for p in (old, new))
            if not all(equality.values()) or not links_equal or old_spec != new_spec:
                raise AssertionError("Task input changed: %s/%s %s" % (camera, host, equality))
            if sha(new / "task.json") != new_record["variants"][host]["task_sha256"]:
                raise AssertionError("Task variant digest differs")
            if new_record["variants"][host]["task"] != str(new / "task.json"):
                raise AssertionError("Private variant locator differs")
            results.append(dict(camera=camera, host=host, files_byte_equal=equality,
                                reference_blob_byte_equal=equality["reference.npz"],
                                source_png_symlink_target_equal=links_equal,
                                complete_task_metadata_equal=True))
    return dict(kind="NATIVE_PACK_CPU8_EXECUTED_INPUT_PARITY_V1", passed=True,
                frame=config["frame"], cameras=len(cameras), variants=len(results),
                camera_and_host_order_equal=True, all_input_bytes_equal=True,
                source_snapshot_unchanged=source_files_unchanged,
                ignored_fields=["parent record task output locator only"],
                baseline_configuration_sha256=sha(configuration),
                snapshot_manifest_sha256=sha(snapshot_manifest),
                execution_sha256=sha(candidate / "packing_execution.json"),
                workers=execution["workers"],
                measured_seconds=execution["finished_at"] - execution["started_at"],
                no_GPU_or_source_publication=True, records=results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--configuration", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = compare(a.baseline, a.candidate, a.configuration, a.snapshot_manifest)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}))
