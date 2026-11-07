"""Silver-only CPU source closure, immutable checkpoint hardlink, no CUDA work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main(root):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path[:0] = [str(root / 'code/LookCloser/scripts'), str(root / 'code')]
    import torch
    from train_marshawn_semantic_late_distortion import tree_digest
    read = lambda p: json.loads(Path(p).read_text())
    preparation = read(root / 'preparation.json')
    source = Path(preparation['source_checkpoint_remote'])
    expected = preparation['source_checkpoint_sha256']
    assert sha(source) == expected
    target = root / 'anchor/checkpoint.ckpt'
    if not target.exists():
        os.link(source, target)
    assert sha(target) == expected
    checkpoint = torch.load(target, map_location='cpu', weights_only=False)
    anchor = read(root / 'anchor_audit.json')
    hashes = {key: tree_digest(checkpoint[key]) for key in anchor['fullstate_hashes']}
    assert hashes == anchor['fullstate_hashes'] and checkpoint['step'] == 126316
    for path, digest in anchor['source_hashes'].items():
        assert sha(path) == digest, path
    inventory = read(root / 'source_manifest.json')
    for path, digest in inventory['files_sha256'].items():
        assert sha(root / 'code' / path) == digest, path
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total',
                                   '--format=csv,noheader,nounits'], text=True)
    rows = {row[0].strip(): dict(uuid=row[1].strip(), memory_used_MiB=int(row[2]),
            memory_total_MiB=int(row[3])) for row in (line.split(',') for line in gpu.splitlines())}
    uuids = {key: rows[key]['uuid'] for key in ('3', '4')}
    assert not torch.cuda.is_initialized()
    receipt = dict(passed=True, CPU_only=True, CUDA_initialized=False, checkpoint_sha256=expected,
        source=str(source), target=str(target), source_size_bytes=source.stat().st_size,
        hardlinked=source.stat().st_ino == target.stat().st_ino,
        source_seven_state_hashes=hashes, original1495_source_exact=True,
        total_scientific_and_helper_files=len(inventory['files_sha256']),
        anchor_evidence_all_actual_remote_hashes_exact=True, GPU_snapshot=rows,
        approved_UUIDs=uuids, physical_GPU_lease_acquired=False, GPU_started=False,
        source_checkpoint_hash_after_audit=sha(source))
    assert receipt['source_checkpoint_hash_after_audit'] == expected
    (root / 'GPU_UUIDs.json').write_text(json.dumps(uuids, indent=2) + '\n')
    (root / 'checkpoint_staging_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (root / 'remote_staging_verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(passed=True, step=126316, seven_states_exact=True, approved_UUIDs=uuids,
                         original1495_exact=True, CUDA_initialized=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
