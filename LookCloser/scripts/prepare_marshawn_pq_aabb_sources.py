"""Read-only PQ source previews and calibrated cameras for AABB debugging.

Preview colour uses PQ-space area reduction before decoding, so is not a
photometric training master. Raster is contained in a 1920x1080 canvas without
stretching or padding. Intrinsics use actual independent raster scale factors.
"""
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image


def prepare(a):
    cv2.setNumThreads(1)
    def one(task):
        f, name = task
        roots = list(a.root.glob('*/root/Marshawn_2_1/working/fullres_pq16'))
        p = next(r/f'{f:06d}'/'images'/f'{name}_{f:06d}.png' for r in roots
                 if (r/f'{f:06d}'/'images'/f'{name}_{f:06d}.png').exists())
        text = p.parent.parent/'sparse/text'
        cameras = {int(t[0]): t for l in (text/'cameras.txt').read_text().splitlines()
                   if l and not l.startswith('#') for t in [l.split()]}
        row = next(l.split() for l in (text/'images.txt').read_text().splitlines()
                   if l and not l.startswith('#') and len(l.split()) >= 10
                   and Path(l.split()[9]).stem == f'{name}_{f:06d}')
        cam = cameras[int(row[8])]
        assert cam[1] == 'PINHOLE'
        w, h = map(int, cam[2:4]); fx, fy, cx, cy = map(float, cam[4:8])
        x = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        assert x.shape == (h,w,3) and x.dtype == np.uint16
        scale = min(1920/w,1080/h); W,H = round(w*scale), round(h*scale)
        x = cv2.resize(x,(W,H),interpolation=cv2.INTER_AREA)[...,::-1].astype(np.float32)/65535
        q = x**(1/(2523/4096*128))
        L = (np.maximum(q-3424/4096,0)/(2413/4096*32-2392/4096*32*q))**(1/(2610/16384))*10000/52.13360318135631
        def xyz(x,y): return np.array([x/y,1,(1-x-y)/y])
        def matrix(pr):
            P=np.stack([xyz(*v) for v in pr],1)
            return P*np.linalg.solve(P,xyz(.3127,.3290))
        M=np.linalg.inv(matrix([(.64,.33),(.3,.6),(.15,.06)]))@matrix([(.713,.293),(.165,.830),(.128,.044)])
        x=np.maximum(L@M.T,0)*1.6; y=np.clip(x*(1+x/36)/(1+x),0,1)
        s=np.where(y<=.0031308,y*12.92,1.055*np.maximum(y,1e-8)**(1/2.4)-.055)
        out=a.output/str(f)/name; out.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.uint8(np.clip(s*255+.5,0,255))).save(out/'source.png')
        q=np.array(row[1:5],float); R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        pose=np.eye(4);pose[:3,:3]=R.T@np.diag([1,-1,-1]);pose[:3,3]=-R.T@np.array(row[5:8],float)
        camera=dict(physical_camera=name,w=W,h=H,fl_x=fx*W/w,fl_y=fy*H/h,
                    cx=cx*W/w,cy=cy*H/h,transform_matrix=pose.tolist())
        mask=a.masks/f'{f:06d}'/f'{name}_{f:06d}.png'
        if mask.exists():
            m=cv2.imread(str(mask),0);assert m.shape==(h,w)
            Image.fromarray(cv2.resize(m,(W,H),interpolation=cv2.INTER_AREA)).save(out/'production_mask.png')
        receipt=dict(frame=f,camera=name,camera_row=camera,image=str(out/'source.png'),
                     source=str(p),native_size=[w,h],preview_only=True,
                     camera_text_sha256=hashlib.sha256((text/'cameras.txt').read_bytes()).hexdigest(),
                     pose_text_sha256=hashlib.sha256((text/'images.txt').read_bytes()).hexdigest(),
                     pixel_convention='COLMAP half-integer centers; OpenGL camera-to-world',
                     photometry='PQ/AP1 to Rec709 preview, PQ-area-prefilter, gain1.6 soft curve')
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(f,name,flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one,[(f,c) for f in a.frames for c in a.cameras]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/raid/lookcloser/s3'))
    p.add_argument('--masks',type=Path,default=Path('/raid/lookcloser/masks8k'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,nargs='+',required=True)
    p.add_argument('--cameras',nargs='+',default=['DC','HC','KA','NC','JE','AB'])
    p.add_argument('--workers',type=int,default=3);prepare(p.parse_args())
