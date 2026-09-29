"""Remove only unlinked intermediate checkpoints from brans' recorded DEC5 runs."""
import argparse
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--journal', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/home/brans/lookcloser_artifacts/dec5_core_retraining')
    uid = pwd.getpwnam('brans').pw_uid
    if os.getuid() != uid:
        raise RuntimeError('This cleanup is authorized only for brans')
    candidates = []
    for path in root.rglob('step_*.pt'):
        stat = path.lstat()
        if (path.is_symlink() or stat.st_uid != uid or stat.st_nlink != 1
                or not re.fullmatch(r'step_\d+\.pt', path.name)):
            continue
        # A retained best model and a request identifying this exact directory
        # are required. Never delete a best model's hardlinked inode.
        request = path.parent / 'request.json'
        if not request.is_file() or not (path.parent / 'best.pt').is_file():
            continue
        if str(path.parent) not in request.read_text():
            continue
        candidates.append((path, stat.st_ino, stat.st_size))
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    removed = 0
    with args.journal.open('a') as journal:
        for path, inode, size in sorted(candidates, key=lambda r: str(r[0])):
            if shutil.disk_usage(root).free >= 100 * 2**30:
                break
            stat = path.lstat()
            if stat.st_ino != inode or stat.st_uid != uid or stat.st_nlink != 1 or path.is_symlink():
                raise RuntimeError(f'Checkpoint changed: {path}')
            record = dict(path=str(path), uid=uid, inode=inode, bytes=size,
                          action='unlink' if args.execute else 'dry_run', time=time.time())
            journal.write(json.dumps(record) + '\n'); journal.flush(); os.fsync(journal.fileno())
            if args.execute:
                path.unlink()
            removed += size
    print(json.dumps(dict(execute=args.execute, candidates=len(candidates),
                          processed_GiB=removed/2**30, free_GiB=shutil.disk_usage(root).free/2**30)))


if __name__ == '__main__':
    main()
