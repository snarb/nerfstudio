"""Compare actual all66 asynchronous collection with original V62 outputs."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(value, source, target):
    if isinstance(value, str):
        return value.replace(str(source), str(target))
    if isinstance(value, list):
        return [normalize(v, source, target) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v, source, target) for k, v in value.items()}
    return value


def compare(original_config, candidate_config, collected):
    original = json.loads(Path(original_config).read_text())
    candidate = json.loads(Path(candidate_config).read_text())
    baseline = Path(original["isolated_campaign"])
    actual = Path(candidate["isolated_campaign"])
    source_frame = baseline / "frames" / ("%06d" % original["frame"])
    actual_frame = actual / "frames" / ("%06d" % candidate["frame"])
    records, intervals = [], []
    for row in original["records"]:
        camera = row["camera"]
        unit = Path(candidate["root"]) / "collectors" / camera
        proof = json.loads((unit / "complete.json").read_text())
        old_app_path = source_frame / "union_resolution_inputs" / camera / "application.json"
        new_app_path = actual_frame / "union_resolution_inputs" / camera / "application.json"
        old_app = json.loads(old_app_path.read_text())
        new_app = json.loads(new_app_path.read_text())
        if normalize(new_app, actual, baseline) != old_app:
            raise AssertionError("Application/source metadata differs: " + camera)
        checked = {}
        for relative, digest in proof["files_sha256"].items():
            old, new = baseline / relative, actual / relative
            if sha(new) != digest:
                raise AssertionError("Output receipt digest differs: " + relative)
            if new.suffix != ".json":
                same = sha(old) == digest
            else:
                old_value = json.loads(old.read_text())
                new_value = json.loads(new.read_text())
                if relative.endswith("/native_masks/" + camera + ".json"):
                    if old_value["union_resolution"]["receipt_sha256"] != sha(old_app_path):
                        raise AssertionError("Baseline application binding changed")
                    if new_value["union_resolution"]["receipt_sha256"] != sha(new_app_path):
                        raise AssertionError("Candidate application binding changed")
                    new_value["union_resolution"]["receipt_sha256"] = old_value["union_resolution"]["receipt_sha256"]
                same = normalize(new_value, actual, baseline) == old_value
            if not same:
                raise AssertionError("Collected artifact changed: " + relative)
            checked[relative] = True
        if not proof["proof"]["union_conserved"] or not proof["proof"]["canonical_post_clear_replay_exact"]:
            raise AssertionError("Original source/mask validation did not pass")
        intervals.append((proof["started_at"], 1))
        intervals.append((proof["finished_at"], -1))
        records.append(dict(camera=camera, archives_byte_equal=True,
                            all_normalized_metadata_equal=True, source_bindings_equal=True,
                            checked_files=len(checked), original_gather_proof=proof["proof"]))
    depth = peak = 0
    for _, delta in sorted(intervals):
        depth += delta
        peak = max(peak, depth)
    if peak > 2:
        raise AssertionError("More than two collectors executed concurrently")
    completed = json.loads(Path(collected).read_text())
    ticks = [row["seconds"] for row in completed["ticks"]]
    return dict(passed=True, cameras=len(records), frame=original["frame"], records=records,
                all_archives_byte_equal=True, all_normalized_metadata_and_order_equal=True,
                normalized_only=["isolated output root locator", "SHA of application JSON containing that locator"],
                original_configuration_sha256=sha(original_config), configuration_sha256=sha(candidate_config),
                elapsed_seconds=completed["seconds"], maximum_concurrent_collectors=peak,
                parent_poll_loop_seconds=dict(median=statistics.median(ticks), maximum=max(ticks)),
                no_NN_or_GPU_jobs_or_source_publication=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-config", type=Path, required=True)
    p.add_argument("--candidate-config", type=Path, required=True)
    p.add_argument("--collected", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = compare(a.original_config, a.candidate_config, a.collected)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))
