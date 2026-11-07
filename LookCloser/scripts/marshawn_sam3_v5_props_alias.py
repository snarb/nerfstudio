"""Same-instant V5 native input aliases for the unchanged MAT/BAG reader."""
from pathlib import Path
import argparse, hashlib, json, os, shutil


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(campaign, frame):
    V = Path(campaign); source = V / 'operations/sam3_assist/v5' / ('frame%06d' % frame)
    target = V / 'operations/sam3_assist' / ('candidates%06d' % frame); records = {}
    for task in sorted((source / 'tasks').glob('*/task.json')):
        spec = json.loads(task.read_text()); camera = spec['camera']
        if spec['frame'] != frame:
            raise ValueError('V5 source frame differs')
        out = target / 'tasks' / camera; out.mkdir(parents=True, exist_ok=True)
        for name, expected in [('raw_native.png', spec['raw_native_image_sha256']),
                               ('source_original_receipt.json', spec['original_source_receipt_sha256'])]:
            src = task.parent / name; dst = out / name
            # The reviewed1675 canary staged LOCAL-view RGB only at the
            # existing legacy prefix. Reuse that exact input in either
            # direction; future factories stage every native PNG directly.
            if not src.exists() and dst.exists() and sha(dst) == expected:
                os.link(dst, src)
            if sha(src) != expected:
                raise ValueError('V5 native source changed: ' + camera)
            if not dst.exists():
                os.link(src, dst)
            if sha(dst) != expected:
                raise ValueError('Legacy MAT input differs; refuse overwrite: ' + camera)
        dst = out / 'candidate_task.json'
        if not dst.exists():
            shutil.copy2(task, dst)
        existing = json.loads(dst.read_text())
        for key in ['frame', 'camera', 'raw_native_image_sha256', 'original_source_receipt_sha256']:
            if existing[key] != spec[key]:
                raise ValueError('Existing MAT source spec differs: ' + camera)
        records[camera] = dict(candidate_task_sha256=sha(dst), original_source_receipt_sha256=spec['original_source_receipt_sha256'],
                               raw_native_image_sha256=spec['raw_native_image_sha256'], v5_task_sha256=sha(task),
                               candidate_task_byte_identical_to_v5=sha(dst) == sha(task))
    if len(records) != 66:
        raise ValueError('ALL66 V5 native inputs required')
    result = dict(frame=frame, records=records, same_frame_only=True, hardlinked_native_images=True,
                  existing_legacy_specs_preserved=True, helper_sha256=sha(__file__), no_inference=True)
    output = source / 'props_alias_ready.json'; temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n'); temporary.replace(output)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--campaign', type=Path, required=True); p.add_argument('--frame', type=int, required=True)
    a = p.parse_args(); print(json.dumps(run(a.campaign, a.frame)))
