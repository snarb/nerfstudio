"""Frozen-field view color at the same LookCloser surface point.

Unlike reprojecting source renders, both radiance queries share precisely the
same 3D point. Only direction changes. No density, geometry or grid is fitted.
"""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from luster_hybrid_common import DIGEST,assert_source,geometry,log,native_rows,read,sha,unproject,write
from render_luster_hybrid import load,state_hashes


def paired_query(model,points,target_directions,source_directions):
    levels=model.freq_grid.query(points).float()
    target_density,target_rgb=model.field.query_points(points,target_directions,l_grid=levels)
    source_density,source_rgb=model.field.query_points(points,source_directions,l_grid=levels)
    assert torch.equal(target_density,source_density),'Changing view direction changed density'
    return target_rgb.float(),source_rgb.float()


def run(a):
    _,model=load(a.root,4096);before=state_hashes(model);rows=native_rows();old=a.root/'candidates'/a.parent
    for target in [7,12,95,150]:
        out=a.root/'point_view'/a.parent/f'cam{target:03d}';out.mkdir(parents=True,exist_ok=True)
        pr=read(old/f'cam{target:03d}_receipt.json');parent_geometry=a.root/'geometry'/f'cam{target:03d}'/'receipt.json'
        assert read(parent_geometry)['checkpoint_sha256']==DIGEST and read(parent_geometry)['state_hash']==before
        inputs={str(p):sha(p) for p in [old/f'cam{target:03d}_receipt.json',old/f'cam{target:03d}_sources.npy',parent_geometry]}
        if (out/'receipt.json').exists():
            r=read(out/'receipt.json');assert r['code_sha256']==sha(Path(__file__)) and r['inputs']==inputs
            assert r['state_hash']==before and all(sha(out/f'{k}.npy')==v for k,v in r['hashes'].items())
            continue
        labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r');g=geometry(a.root,target);shape=labels.shape
        source_centers=np.zeros((166,3),np.float32)
        for cid in pr['sources']:
            assert_source(rows,target,cid);source_centers[cid]=np.asarray(rows[cid]['transform_matrix'],np.float32)[:3,3]
        target_center=np.asarray(rows[target]['transform_matrix'],np.float32)[:3,3]
        arrays={k:np.lib.format.open_memmap(out/f'{k}.npy',mode='w+',dtype=np.float32,shape=(*shape,3)) for k in ['target_rgb','source_rgb']}
        for array in arrays.values():array[:]=0
        yy,xx=np.nonzero(labels>=0);start_time=time.monotonic();max_difference=0.
        with torch.inference_mode():
            for start in range(0,len(yy),a.chunk):
                y=yy[start:start+a.chunk];x=xx[start:start+a.chunk];xy=np.stack([x,y],-1).astype(np.float32)
                points=unproject(np.asarray(g['depth_z_q50'][y,x]),rows[target],xy)
                assert np.isfinite(points).all()
                target_dirs=points-target_center;source_dirs=points-source_centers[labels[y,x]]
                target_dirs/=np.linalg.norm(target_dirs,axis=-1,keepdims=True);source_dirs/=np.linalg.norm(source_dirs,axis=-1,keepdims=True)
                values=paired_query(model,*[torch.from_numpy(v).to(model.device) for v in [points,target_dirs,source_dirs]])
                values=[v.cpu().numpy() for v in values]
                assert all(np.isfinite(v).all() for v in values)
                max_difference=max(max_difference,float(np.max(np.abs(values[0]-values[1]))))
                for key,value in zip(arrays,values):arrays[key][y,x]=value
        assert state_hashes(model)==before,'Model state changed'
        for array in arrays.values():array.flush()
        write(out/'receipt.json',dict(camera=target,parent=a.parent,checkpoint_sha256=DIGEST,state_hash=before,inputs=inputs,
            code_sha256=sha(Path(__file__)),hashes={k:sha(out/f'{k}.npy') for k in arrays},pixels=len(yy),max_view_rgb_difference=max_difference,
            geometry='unchanged target camera-z q50',lookup_shift=False,uses_target_rgb=False,uses_masks=False,seconds=time.monotonic()-start_time))
        log(a.root,'point_view_cached',camera=target,max_view_rgb_difference=max_difference,seconds=time.monotonic()-start_time)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='single_cubic75_fixed');p.add_argument('--chunk',type=int,default=65536);a=p.parse_args();run(a)
