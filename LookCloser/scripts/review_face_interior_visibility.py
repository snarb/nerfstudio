"""Audit face-interior recovery and save native before/after/source witnesses.

Review-only, never changes meshes, predictions, masks or frozen experiment code.
"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw
import torch
from study_multiview_face_prior import read, save, sha
from native_texture_footprint import sample_native, snap_centers, relevant_tap
from study_face_interior_visibility import proposals

NOSE = Path('/mnt/data/dec5_nose_source_visibility_001123')


def support_at(masks, uv):
    q = snap_centers(torch.tensor(uv[:, None], device='cpu'))
    tensor = torch.tensor(masks[:, None].astype(np.float32))
    support = torch.ones(uv.shape[:2], dtype=torch.bool)
    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        values = sample_native(tensor, q.floor() + q.new_tensor([dx, dy]))[:, 0, 0]
        support &= (values > .5) | ~relevant_tap(q, dx, dy)
    return support.numpy()


def panel(images, labels, path):
    out = Image.new('RGB', (sum(im.width for im in images), max(im.height for im in images) + 25))
    draw = ImageDraw.Draw(out)
    x = 0
    for im, label in zip(images, labels):
        out.paste(im, (x, 25)); draw.text((x + 3, 4), label, fill='white'); x += im.width
    out.save(path)


def main(root):
    review = root / 'review'; review.mkdir(exist_ok=False)
    result = read(root/'result.json'); request = read(root/'request.json')
    assert result['request_sha256'] == sha(root/'request.json')
    bindings = {}
    for name, h in result['hashes'].items():
        assert sha(root/name) == h; bindings[str(root/name)] = h
    for p, h in request['input_hashes'].items():
        assert sha(p) == h; bindings[p] = h
    base = Path(result['baseline']); complete = read(base/'complete.json')
    assert sha(base/'complete.json') == result['baseline_complete_sha256']
    for name, h in complete['hashes'].items():
        assert sha(base/name) == h; bindings[str(base/name)] = h
    a = np.array(Image.open(base/'prediction_native.png')); b = np.array(Image.open(root/'prediction_native.png'))
    sa = np.array(Image.open(base/'source_ids.png')); sb = np.array(Image.open(root/'source_ids.png'))
    e = np.load(root/'evidence.npz'); changed = np.flatnonzero((sa != sb).ravel())
    np.testing.assert_array_equal(changed, np.sort(e['pixels']))
    np.testing.assert_array_equal(sa.ravel()[e['pixels']], e['old_sources'])
    np.testing.assert_array_equal(sb.ravel()[e['pixels']], e['new_sources'])
    np.testing.assert_array_equal(a[sa == sb], b[sa == sb])
    assert (e['old_valid_face_votes'] >= 3).all()
    assert (e['new_quality'] > e['old_quality']).all()
    assert np.isfinite(e['direct_t']).all() and (abs(e['direct_t']-1) <= 1e-5).all()
    assert not ((a.max(2)>0) & (b.max(2)==0)).any()
    A, B = Image.fromarray(np.rot90(a)), Image.fromarray(np.rot90(b))
    panel([A.resize((540,960)), B.resize((540,960))], ['Baseline overview', 'Candidate overview'], review/'overview.png')
    panel([A.crop((780,450,1080,850)), B.crop((780,450,1080,850))], ['Baseline face 1:1', 'Candidate face 1:1'], review/'face.png')
    panel([A.crop((875,595,960,765)), B.crop((875,595,960,765))], ['Baseline', 'Candidate'], review/'nose.png')
    Y,X = np.nonzero(np.rot90(sa != sb)); boxes=[]
    # Every changed pixel is covered by a native 128px tile with 12px context.
    for tx,ty in sorted(set(zip((X//128).tolist(),(Y//128).tolist()))):
        box=[max(0,128*tx-12), max(0,128*ty-12), min(1080,128*(tx+1)+12), min(1920,128*(ty+1)+12)]
        name=f'changed_{tx}_{ty}.png'; boxes.append(dict(path=name,box=box))
        panel([A.crop(box),B.crop(box)], ['Baseline 1:1','Candidate 1:1'],review/name)
    ne=np.load(NOSE/'evidence.npz'); masks=np.load(root/'face_masks.npz')['masks']
    assert sha(root/'face_masks.npz')==request['face_masks_sha256']
    support=support_at(masks,ne['uv']); cand,votes,quality=proposals(ne['raster_valid'],support,ne['quality'],ne['source_ids'])
    j=np.arange(len(ne['source_ids'])); x,y=ne['native_xy'].T
    names=read(NOSE/'result.json')['source_cameras']; ci=names.index('H004_C005_1210SZ')
    stages=dict(samples=len(j), h_exact_visible=int(ne['direct_visible'][ci].sum()),
        h_original_raster_valid=int(ne['raster_valid'][ci].sum()), h_face_support=int(support[ci].sum()),
        chosen_face_support=int(support[ne['source_ids'],j].sum()), three_old_face_witnesses=int((votes>=3).sum()),
        h_new_proposals=int(cand[ci].sum()), changed_source_at_nose=int((sa[y,x]!=sb[y,x]).sum()))
    np.savez_compressed(review/'nose_gates.npz',support=support,votes=votes,proposals=cand)
    for p in [root/'request.json', root/'result.json', root/'face_masks.npz', NOSE/'evidence.npz', NOSE/'result.json', Path(__file__)]:
        bindings[str(p)]=sha(p)
    save(review/'audit.json',dict(nose_stages=stages,changed_pixels=len(changed),
        changed_tiles=boxes,rest_bit_identical=True,new_black=0,input_hashes=bindings,
        geometry_changed=False,visual_status='pending',
        images={str(p):sha(p) for p in review.glob('*.png')}))
    print(stages,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    torch.set_num_threads(2)
    with torch.inference_mode(): main(a.root)
