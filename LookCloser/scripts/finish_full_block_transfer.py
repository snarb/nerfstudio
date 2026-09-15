"""Native diagnostic crops and checksum seal; no rendering or mesh changes."""
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from joint_temporal_texture import read, sha, atomic_json
from review_full_block_transfer import ROOT, VIDEO
from review_jaw_repair_transfer import verified_image, panel


def crops(frame):
    root = ROOT/frame
    boxes = {'moving': (380, 1400, 680, 1870),
             'H004_C005_1210SZ': (230, 1080, 530, 1550),
             'K004_B005_1210DS': (40, 1120, 340, 1590)}
    for view, box in boxes.items():
        images, labels = [], []
        out = root/'review'/view
        if view != 'moving':
            images.append(np.asarray(Image.open(out/'train_gt.png')))
            labels.append('real train GT')
        for arm in ('production', 'fuse-original', 'fuse-full-block'):
            folder = VIDEO if view == 'moving' and arm == 'production' else root/'rgb'/view/arm
            rgb, _ = verified_image(folder, frame)
            images.append(rgb); labels.append(arm)
            Image.fromarray(rgb).crop(box).save(out/(arm+'_lipstick_native.png'))
        panel(out/'lipstick_native.png', images, labels, box)
    atomic_json(root/'review/crop_request.json', dict(frame=frame, boxes=boxes,
        native_scale=True, diagnostic_only=True, script_sha256=sha(__file__)))


def seal(frame):
    root = ROOT/frame
    result = read(root/'review/result.json')
    visual = read(root/'visual_review.json')
    assert set(visual['views']) == {r['view'] for r in result['records']}
    assert visual['production_promoted'] is False
    hashes = {}
    received = read(root/'received.json')
    assert received['request_sha256'] == sha(root/'request.json')
    for name, digest in {**received['retained_hashes'], **received['depth_hashes']}.items():
        assert sha(root/name) == digest
    for record in result['records']:
        view = record['view']
        assert visual['views'][view]['status'] not in ('pending', 'uncertain')
        for name, digest in record['image_hashes'].items():
            assert sha(root/'review'/view/name) == digest
        for arm, digest in record['request_hashes'].items():
            folder = VIDEO if view == 'moving' and arm == 'production' else root/'rgb'/view/arm
            assert sha(folder/'request.json') == digest
            verified_image(folder, frame)
            q = read(folder/'request.json')
            for name, expected in q['script_hashes'].items():
                p = Path(__file__).with_name(name)
                assert sha(p) == expected
                hashes[str(p.resolve())] = expected
            for name in ('request.json', f'frames/{frame}/frame.png',
                         f'frames/{frame}/complete.json', f'frames/{frame}/target_depth.npz'):
                p = folder/name; hashes[str(p)] = sha(p)
    for p, digest in result['source_receipt']['source_rgb_hashes'].items():
        assert sha(p) == digest; hashes[p] = digest
    atomic_json(root/'review_completion.json', dict(frame=frame,
        status=visual['status'], visual_review_sha256=sha(root/'visual_review.json'),
        workers_terminal=True, production_promoted=False, remote_scratch_deleted=False))
    for p in root.rglob('*'):
        if p.is_file() and p.name != 'artifact_manifest.json': hashes[str(p)] = sha(p)
    hashes[str(Path(__file__).resolve())] = sha(__file__)
    atomic_json(root/'artifact_manifest.json', dict(hashes=hashes))
    for p, digest in read(root/'artifact_manifest.json')['hashes'].items(): assert sha(p) == digest
    print('Sealed and rechecked', len(hashes), 'SHA-256 bindings', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('crops', 'seal'))
    p.add_argument('--frame', required=True, choices=('000995', '000997'))
    a = p.parse_args(); globals()[a.action](a.frame)
