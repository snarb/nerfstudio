"""Uniform train-only head-mask expansion from measured multiview witnesses.

No proposed mesh or target view is used. Each native query must have measured
depth; at least three other physical cameras must agree in depth, roundtrip,
parallax, calibrated patch color and ORIGINAL foreground masks. One pass only:
new seeds never serve as witnesses for another query. Texture masks unchanged.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import distance_transform_edt
from joint_temporal_texture import read,sha,atomic_json,project
from transfer_close_boundary_completion import SOURCE,MOVIE,FRAMES
from study_confidence_depth_prior import load_real,unproject,project_integer
from calibrated_depth_witness import load_images
from forearm_rgb_witnesses import color_errors
from build_measured_foreground_override import add_certified_seeds

ROOT=Path('/mnt/data/dec5_uniform_measured_head_masks')


def foreground_witnesses(points,rows,masks,names):
    """Use the existing integer-lattice mask witness rule, never new masks."""
    foreground=np.zeros((len(rows),len(points)),bool)
    for i,row in enumerate(rows):
        uv,z=project_integer(row,points);xy=np.rint(uv).astype(int)
        good=(z>0)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
        ids=np.flatnonzero(good)
        foreground[i,ids]=masks[names.index(row['physical_camera']),xy[ids,1],xy[ids,0]]
    return foreground


def run(frame):
    root=ROOT/frame;root.mkdir(parents=True,exist_ok=True)
    base=read(SOURCE/frame/'request.json');rows,depths,receipt=load_real(Path(base['depth_root']),frame)
    assert receipt==base['depth_receipt']
    entry=next(r for r in read(MOVIE/'request.json')['inventory'] if r['frame_id']==frame)
    maskroot=Path(entry['source_masks']['root']);names=read(maskroot/'cameras.json')
    assert len(names)==62 and set(names)=={r['physical_camera'] for r in rows}
    assert sha(maskroot/'masks.npz')==base['source_mask_sha256']
    assert sha(maskroot/'cameras.json')==base['mask_names_sha256']
    masks=np.load(maskroot/'masks.npz')['masks'].astype(bool)
    images,_,rgb_receipt=load_images(frame)
    params=dict(exterior_band_px=24,seed_dilation_px=4,minimum_other_foreground_views=3,
        chroma_limit=.04,rgb_limit=.12,min_head_x=-.03,single_pass_original_witness_masks=True,
        source_masks_lattice='existing native integer witness convention')
    scripts=[Path(__file__).name,'calibrated_depth_witness.py','forearm_rgb_witnesses.py',
        'diagnose_forearm_color_witnesses.py','study_confidence_depth_prior.py','build_measured_foreground_override.py']
    request=dict(frame=frame,parameters=params,rows=rows,depth_receipt=receipt,rgb_receipt=rgb_receipt,
        original_mask_root=str(maskroot),original_masks_sha256=sha(maskroot/'masks.npz'),
        mask_names_sha256=sha(maskroot/'cameras.json'),source_request_sha256=sha(SOURCE/frame/'request.json'),
        scripts={str(Path(__file__).resolve().with_name(n)):sha(Path(__file__).with_name(n)) for n in scripts},
        heldout_used=False,candidate_geometry_used=False,texture_masks_changed=False,production_updated=False)
    if (root/'request.json').exists() and read(root/'request.json')!=request:
        raise ValueError('Changed refinement request')
    atomic_json(root/'request.json',request);records=[];updated=masks.copy()
    for ci,(row,depth) in enumerate(zip(rows,depths)):
        name=row['physical_camera'];folder=root/'cameras'/name;folder.mkdir(parents=True,exist_ok=True)
        mi=names.index(name)
        if (folder/'result.json').exists():
            record=read(folder/'result.json');assert record['request_sha256']==sha(root/'request.json')
            for p,h in record['hashes'].items():
                assert sha(folder/p)==h
            updated[mi]=np.load(folder/'mask.npy');records.append(record);continue
        started=time.monotonic();mask=masks[mi]
        band=(~mask)&(distance_transform_edt(~mask)<=24)
        y,x=np.nonzero(band&np.isfinite(depth)&(depth>0))
        points=unproject(row,x,y,depth[y,x]);head=points[:,0]>-.03
        x,y,points=x[head],y[head],points[head]
        qualified=np.zeros(len(x),np.uint8)
        for start in range(0,len(points),4096):
            pts=points[start:start+4096]
            chroma,rgb=color_errors(pts,row,rows,depths,images)
            foreground=foreground_witnesses(pts,rows,masks,names)
            assert not np.isfinite(chroma[ci]).any(),'Query camera cannot count as its own witness'
            qualified[start:start+len(pts)]=((chroma<=.04)&(rgb<=.12)&foreground).sum(0)
        seeds=np.zeros(mask.shape,bool);seeds[y[qualified>=3],x[qualified>=3]]=True
        refined=add_certified_seeds(mask,seeds)
        assert refined[mask].all()
        updated[mi]=refined
        np.save(folder/'mask.npy',refined)
        np.savez_compressed(folder/'evidence.npz',query_xy=np.column_stack((x,y)),query_points=points,
            qualified=qualified,seeds=seeds)
        record=dict(camera=name,request_sha256=sha(root/'request.json'),measured_head_candidates=len(points),
            certified_seeds=int(seeds.sum()),added_mask_pixels=int((refined&~mask).sum()),
            elapsed_seconds=time.monotonic()-started,hashes={n:sha(folder/n) for n in ['mask.npy','evidence.npz']})
        atomic_json(folder/'result.json',record);records.append(record)
        atomic_json(root/'progress.json',dict(frame=frame,cameras_completed=ci+1,current_camera=name,
            added_mask_pixels=sum(r['added_mask_pixels'] for r in records),utc=datetime.now(timezone.utc).isoformat()))
        print(frame,ci+1,name,'candidates',len(points),'seeds',record['certified_seeds'],
            'added',record['added_mask_pixels'],'seconds',round(record['elapsed_seconds'],2),flush=True)
    assert updated[masks].all()
    np.savez_compressed(root/'masks.npz',masks=updated)
    atomic_json(root/'cameras.json',names)
    atomic_json(root/'result.json',dict(request_sha256=sha(root/'request.json'),records=records,
        added_pixels=sum(r['added_mask_pixels'] for r in records),
        hashes={n:sha(root/n) for n in ['masks.npz','cameras.json']},original_masks_preserved=True,
        production_updated=False,texture_masks_changed=False,visual_status='pending'))
    print(frame,'all62 completed',sum(r['added_mask_pixels'] for r in records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=FRAMES,required=True)
    a=p.parse_args();run(a.frame)
