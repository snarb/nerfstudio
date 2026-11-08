"""Read-only artifact adoption after exact GPU process/supervisor death."""
from pathlib import Path
import argparse
import json
from policy import geometry_check, manifest_row, read, sha


def verify(directory, manifest, path, index):
    """A reachable verifier distinguishes incomplete artifacts from SSH doubt."""
    result = dict(kind='SOURCE_BOUND_NATIVE_GEOMETRY_VERIFICATION_V2', index=int(index),
                  verifier_sha256=sha(__file__), policy_sha256=sha(Path(__file__).with_name('policy.py')),
                  manifest_sha256=sha(manifest), path_sha256=sha(path))
    try:
        _, row = manifest_row(manifest, index, check_local_inputs=False)
        result.update(geometry_check(directory, row, read(path), sha(path)))
    except (OSError, ValueError, KeyError, TypeError) as error:
        # The source-bound helper actually ran and inspected this artifact.
        # No valid completion can be adopted; preserve partial files for review.
        result.update(verified=False, review_required=True,
                      failure_kind='missing_incomplete_or_invalid_native_artifact',
                      error_type=type(error).__name__, error=str(error))
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--path', type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.directory, a.manifest, a.path, a.index)))
