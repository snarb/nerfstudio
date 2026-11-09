"""Native PQ16 crops. No training RGB resize; no model or checkpoint is read."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation
from common import atomic, read, sha


def decode_rgb(code):
    """ST2084/AP1-D65 -> linear Rec709, in source units (52.1336 nits/unit)."""
    def xyz(x, y): return np.array([x/y, 1, (1-x-y)/y])
    def matrix(primaries):
        p = np.stack([xyz(*v) for v in primaries], 1)
        return p*np.linalg.solve(p, xyz(.3127, .3290))
    transform = np.linalg.inv(matrix([(.64,.33),(.3,.6),(.15,.06)])) @ matrix([(.713,.293),(.165,.830),(.128,.044)])
    p = code.astype(np.float32)/65535
    p = p**(1/(2523/4096*128))
    nits = (np.maximum(p-3424/4096, 0)/(2413/4096*32-2392/4096*32*p))**(1/(2610/16384))*10000
    return np.maximum((nits/52.13360318135631) @ transform.T, 0).astype(np.float32)


def display(linear):
    # One explicit fixed exposure/curve for PNG targets and EXR evaluation.
    value = np.maximum(linear, 0)*1.6
    value = np.clip(value/(1+value), 0, 1)
    return np.where(value <= .0031308, 12.92*value, 1.055*np.maximum(value, 1e-8)**(1/2.4)-.055).astype(np.float32)


def cameras(frame):
    roots = Path('/raid/lookcloser/s3').glob('*/root/Marshawn_2_1/working/fullres_pq16')
    result = {}
    for root in roots:
        folder = root/f'{frame:06d}'
        if not (folder/'images').exists(): continue
        text = folder/'sparse/text'
        cams = {int(t[0]):t for l in (text/'cameras.txt').read_text().splitlines() if l and not l.startswith('#') for t in [l.split()]}
        for line in (text/'images.txt').read_text().splitlines():
            t = line.split()
            if len(t) < 10 or line.startswith('#') or not t[9].lower().endswith(('.png','.jpg')): continue
            source = folder/'images'/Path(t[9]).with_suffix('.png')
            if not source.exists(): continue
            c = cams[int(t[8])]
            assert c[1] == 'PINHOLE'
            w,h = map(int,c[2:4]); fx,fy,cx,cy = map(float,c[4:8])
            q = np.array(t[1:5],float)
            rotation = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            pose = np.eye(4); pose[:3,:3] = rotation.T @ np.diag([1,-1,-1]);pose[:3,3] = -rotation.T@np.array(t[5:8],float)
            result[source.stem.split('_')[0]] = dict(source=str(source), w=w,h=h,fl_x=fx,fl_y=fy,cx=cx,cy=cy,transform_matrix=pose.tolist(),camera_text_sha256=sha(text/'cameras.txt'),pose_text_sha256=sha(text/'images.txt'))
    if len(result) != 72: raise ValueError('Expected72 native source cameras, got '+str(len(result)))
    return result


def split(rows, world_to_box):
    """Deterministic three azimuths, restricting camera height to middle60%."""
    names = sorted(rows)
    positions = np.array([rows[c]['transform_matrix'] for c in names])[:,:3,3]
    positions = positions @ world_to_box[:3,:3].T + world_to_box[:3,3]
    center = np.median(positions[:,:2],axis=0)
    angles = np.arctan2(*(positions[:,:2]-center)[:,::-1].T) % (2*np.pi)
    low, high = np.quantile(positions[:,2],[.2,.8])
    eligible = [i for i in range(len(names)) if low<=positions[i,2]<=high]
    import itertools
    def score(ids):
        a=np.sort(angles[list(ids)]); gaps=np.diff(np.r_[a,a[0]+2*np.pi])
        return (float(((gaps-2*np.pi/3)**2).sum()), tuple(names[i] for i in ids))
    selected=min(itertools.combinations(eligible,3),key=score)
    return dict(heldout=sorted(names[i] for i in selected),geometry=[dict(camera=c,box_position=p.tolist(),azimuth=float(a)) for c,p,a in zip(names,positions,angles)],policy='middle60% height; minimize squared circular120degree gap error; camera name tie-break')


def prepare(frame, output, box_path, split_path, workers=3, retain_linear=True):
    cv2.setNumThreads(1)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'native_ready.json').exists(): return read(out/'native_ready.json')
    rows=cameras(frame);box=read(box_path);transform=np.asarray(box['world_to_box'])
    if not Path(split_path).exists(): atomic(split_path,split(rows,transform))
    heldout=read(split_path)['heldout']
    for name in ('images','linear','masks','semantics','qc'): (out/name).mkdir(exist_ok=True)
    def one(item):
        name,row=item
        receipt=out/'qc'/(name+'.json')
        if receipt.exists(): return read(receipt)
        mask_path=Path('/raid/lookcloser/masks8k')/f'{frame:06d}'/f'{name}_{frame:06d}.png'
        alpha=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
        assert alpha is not None and alpha.shape==(row['h'],row['w'])
        yy,xx=np.nonzero(alpha>=16)
        if not len(xx): raise ValueError('Empty production mask '+name)
        x0=max(0,int(xx.min())-128);y0=max(0,int(yy.min())-128)
        x1=min(row['w'],int(xx.max())+129);y1=min(row['h'],int(yy.max())+129)
        source=cv2.imread(row['source'],cv2.IMREAD_UNCHANGED)
        assert source.dtype==np.uint16 and source.shape==(row['h'],row['w'],3)
        linear=decode_rgb(source[y0:y1,x0:x1,::-1]); del source
        rgb=display(linear)
        if retain_linear:np.save(out/'linear'/(name+'.npy'),linear,allow_pickle=False)
        Image.fromarray(np.uint8(np.clip(rgb*255+.5,0,255))).save(out/'images'/(name+'.png'))
        a=alpha[y0:y1,x0:x1];actor=a>=128
        inside=distance_transform_edt(np.pad(actor,1))[1:-1,1:-1]
        outside=distance_transform_edt(~actor)
        # W64 is confidence taper outside subject, not binary alpha supervision.
        confidence=np.maximum(0,1-outside/64).astype(np.float32)
        core=inside>32;band=(outside<=64)&~core
        valid=np.ones(actor.shape,bool)
        np.savez_compressed(out/'semantics'/(name+'.npz'),black=actor,grey=np.zeros_like(actor),core=core,band=band,valid=valid,W32=np.maximum(0,1-outside/32).astype(np.float32),W64=confidence,soft_alpha=a,opaque_core=core)
        Image.fromarray(np.full(actor.shape,255,np.uint8)).save(out/'masks'/(name+'.png'))
        pose=transform@np.asarray(row['transform_matrix'])
        f=dict(physical_camera=name,file_path='images/'+name+'.png',mask_path='masks/'+name+'.png',w=x1-x0,h=y1-y0,fl_x=row['fl_x'],fl_y=row['fl_y'],cx=row['cx']-x0,cy=row['cy']-y0,transform_matrix=pose.tolist())
        # Exact crop-coordinate ray parity, without any focal scaling.
        pts=np.array([[.5,.5],[x1-x0-.5,y1-y0-.5],[(x1-x0)/2,(y1-y0)/2]])
        before=(pts+[x0,y0]-[row['cx'],row['cy']])/[row['fl_x'],row['fl_y']]
        after=(pts-[f['cx'],f['cy']])/[f['fl_x'],f['fl_y']]
        assert np.max(np.abs(before-after))<1e-12
        image=Image.open(out/'images'/(name+'.png'));image.thumbnail((640,640));image.save(out/'qc'/(name+'_source.jpg'))
        edge=cv2.morphologyEx(actor.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)
        overlay=np.uint8(np.clip(rgb*255+.5,0,255));overlay[edge]=[255,0,0]
        im=Image.fromarray(overlay);im.thumbnail((640,640));im.save(out/'qc'/(name+'_mask.jpg'))
        result=dict(camera=name,frame=frame,image=f['file_path'],camera_row=f,native_crop_xyxy=[x0,y0,x1,y1],source=row['source'],source_sha256=sha(row['source']),source_mask=str(mask_path),source_mask_sha256=sha(mask_path),rgb_sha256=sha(out/f['file_path']),semantic_sha256=sha(out/'semantics'/(name+'.npz')),linear_sha256=sha(out/'linear'/(name+'.npy')) if retain_linear else None,ray_parity_max=float(np.max(np.abs(before-after))),native_no_resize=True,calibration={k:row[k] for k in ('camera_text_sha256','pose_text_sha256')})
        atomic(receipt,result);print('PREP',frame,name,f['w'],f['h'],flush=True);return result
    with ThreadPoolExecutor(workers) as pool: records=list(pool.map(one,sorted(rows.items())))
    tf=dict(camera_model='OPENCV',frames=[r['camera_row'] for r in records],train_filenames=[r['image'] for r in records if r['camera'] not in heldout],val_filenames=[r['image'] for r in records if r['camera'] in heldout],test_filenames=[r['image'] for r in records if r['camera'] in heldout],orientation_override='none')
    atomic(out/'transforms.json',tf);atomic(out/'box.json',box)
    ready=dict(frame=frame,train=69,validation=heldout,records=records,aabb=box['aabb'],world_to_box=box['world_to_box'],transforms_sha256=sha(out/'transforms.json'),photometry='PQ16 AP1-D65 -> scene-linear Rec709 /52.13360318135631 nits; PNG=Reinhard gain1.6 -> sRGB8; EXR=unclipped linear, no added source information',native_no_resize=True,linear_masters_retained=retain_linear)
    atomic(out/'native_ready.json',ready);return ready


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frame',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--box',required=True);p.add_argument('--split',required=True);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();prepare(a.frame,a.output,a.box,a.split,a.workers)
