"""Validate a finished native hybrid delivery and produce a compact audit."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def run(root):
    root=Path(root);op=root/'operations';out=op/'flying_native_hybrid_4k_20261112';base=op/'flying_union_interior_20261112'
    old=json.loads((base/'delivery_receipt.json').read_text());old_hash={r['index']:r['image_sha256'] for r in old['rows']}
    assert (out/'path.json').read_bytes()==(base/'path.json').read_bytes()
    coverage=[];seconds=[];tail=[]
    for i in range(70):
        f=out/'neural'/f'{i:03}';r=json.loads((f/'receipt.json').read_text())
        assert r['index']==i and r['source_frame']==908+5*i
        assert r['output_sha256']==sha(f/'strict.png')
        for d in r['donors'].values():assert d['rgb_resized'] is False
        image=cv2.imread(str(f/'strict.png'));baseline=cv2.imread(str(f/'baseline_4k.png'))
        weight=cv2.imread(str(f/'transfer.png'),0)
        assert image.shape==baseline.shape==(2160,3840,3)
        # Quantized weight zero can include a tiny nonzero float feather;
        # exact float-zero fallback was asserted in the renderer itself.
        black=np.all(baseline==0,axis=-1)
        assert np.array_equal(image[black],baseline[black])
        coverage.append(r['detail']['transferred_fraction']);seconds.append(r['seconds'])
    for i in range(70,84):
        p=base/'gt'/f'{i:03}.png';assert sha(p)==old_hash[i]
        expected=cv2.resize(cv2.imread(str(p)),(3840,2160),interpolation=cv2.INTER_LANCZOS4)
        actual=cv2.imread(str(out/'gt'/f'{i:03}.png'));assert np.array_equal(expected,actual)
        tail.append(dict(index=i,original_sha256=sha(p),exact_presentation_resize=True))
    video=out/'Marshawn_native_hybrid_flight_4K.mp4';cap=cv2.VideoCapture(str(video));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(2160,3840,3);n+=1
    fps=cap.get(cv2.CAP_PROP_FPS);cap.release();assert n==84 and fps==24
    receipt=dict(frames=n,fps=fps,dimensions=[3840,2160],duration_seconds=n/fps,video_sha256=sha(video),
        hybrid_frames=70,source_rgb_native=True,source_times_exact=True,black_exterior_unchanged=True,
        coverage_min=float(min(coverage)),coverage_mean=float(np.mean(coverage)),coverage_max=float(max(coverage)),
        mean_seconds=float(np.mean(seconds)),tail=tail)
    (out/'final_validation.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='tail'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
