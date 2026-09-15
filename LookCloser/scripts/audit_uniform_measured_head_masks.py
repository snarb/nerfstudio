"""Replay every head-band query and foreground/color witness from original data."""
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from joint_temporal_texture import read,sha,atomic_json
from refine_measured_head_masks import ROOT,foreground_witnesses
from transfer_close_boundary_completion import FRAMES,SOURCE
from calibrated_depth_witness import load_images
from study_confidence_depth_prior import load_real,unproject
from forearm_rgb_witnesses import color_errors
from build_measured_foreground_override import add_certified_seeds


def run():
    records=[]
    for frame in FRAMES:
        root=ROOT/frame;q=read(root/'request.json');r=read(root/'result.json')
        assert r['request_sha256']==sha(root/'request.json')
        for p,h in q['scripts'].items():
            assert sha(p)==h,p
        source=read(SOURCE/frame/'request.json')
        rows,depths,receipt=load_real(Path(source['depth_root']),frame)
        assert receipt==q['depth_receipt'] and rows==q['rows']
        images,_,rgb=load_images(frame);assert rgb==q['rgb_receipt']
        original=Path(q['original_mask_root']);assert sha(original/'masks.npz')==q['original_masks_sha256']
        names=read(original/'cameras.json');assert sha(original/'cameras.json')==q['mask_names_sha256']
        masks=np.load(original/'masks.npz')['masks'].astype(bool)
        refined=np.load(root/'masks.npz')['masks'];total=0;total_seeds=0;queries=0
        assert names==read(root/'cameras.json') and len(r['records'])==62
        for ci,(row,depth) in enumerate(zip(rows,depths)):
            name=row['physical_camera'];folder=root/'cameras'/name;cr=read(folder/'result.json')
            assert cr['request_sha256']==sha(root/'request.json')
            for p,h in cr['hashes'].items():
                assert sha(folder/p)==h
            e=np.load(folder/'evidence.npz');mask=masks[names.index(name)]
            band=(~mask)&(distance_transform_edt(~mask)<=24)
            y,x=np.nonzero(band&np.isfinite(depth)&(depth>0));pts=unproject(row,x,y,depth[y,x]);head=pts[:,0]>-.03
            x,y,pts=x[head],y[head],pts[head]
            np.testing.assert_array_equal(np.column_stack((x,y)),e['query_xy'])
            np.testing.assert_array_equal(pts,e['query_points'])
            counts=np.zeros(len(pts),np.uint8)
            for start in range(0,len(pts),4096):
                p=pts[start:start+4096];chroma,rgb_error=color_errors(p,row,rows,depths,images)
                fg=foreground_witnesses(p,rows,masks,names)
                assert not np.isfinite(chroma[ci]).any()
                counts[start:start+len(p)]=((chroma<=.04)&(rgb_error<=.12)&fg).sum(0)
            np.testing.assert_array_equal(counts,e['qualified'])
            seeds=np.zeros_like(mask);seeds[y[counts>=3],x[counts>=3]]=True
            np.testing.assert_array_equal(seeds,e['seeds'])
            value=add_certified_seeds(mask,seeds)
            np.testing.assert_array_equal(value,refined[names.index(name)])
            np.testing.assert_array_equal(value,np.load(folder/'mask.npy'))
            assert value[mask].all()
            total+=int((value&~mask).sum());total_seeds+=int(seeds.sum());queries+=len(pts)
            if (ci+1)%16==0:
                print(frame,'replayed',ci+1,'cameras',flush=True)
        assert total==r['added_pixels']
        records.append(dict(frame=frame,all62_queries_and_witnesses_replayed=True,queries=queries,
            seeds=total_seeds,added_mask_pixels=total,original_masks_preserved=True,
            query_camera_excluded=True,new_masks_never_used_as_witnesses=True))
    atomic_json(ROOT/'audit.json',dict(utc=datetime.now(timezone.utc).isoformat(),records=records,
        script_sha256=sha(__file__),semantic_accuracy_not_ground_truth=True,production_updated=False))
    print(records,flush=True)


if __name__=='__main__':
    run()
