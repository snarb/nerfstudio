"""Read-only artifact adoption after exact GPU process/supervisor death."""
from pathlib import Path
import argparse
import json
from policy import geometry_check, manifest_row, read, sha

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--path', type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    a = p.parse_args()
    _, row = manifest_row(a.manifest, a.index, check_local_inputs=False)
    print(json.dumps(geometry_check(a.directory, row, read(a.path), sha(a.path))))
