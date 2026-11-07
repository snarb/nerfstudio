"""Three-view fresh-frame gate/ray smoke; never eligible production admission."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.ndimage import maximum_filter
from mask_visual_hull_aabb import MaskCamera,classify_cells
from build_marshawn_semantic_resolution import cell_bounds
from audit_marshawn_support_rays_cpu import load_kernel,crosscheck
from prepare_marshawn_tuning import write,sha

def main(root):
 data=root/'frames/001665/data_smoke_partial';tf=json.loads((data/'transforms.json').read_text());ready=json.loads((data/'ready.json').read_text());bounds=np.asarray(ready['aabb']);out=root/'smoke1665/CPU_gate';out.mkdir(exist_ok=True);bg=np.zeros(128**3,np.uint8)
 for f in tf['frames']:
  with np.load(data/'semantics'/f"{f['physical_camera']}.npz") as z:c=MaskCamera(f['physical_camera'],np.asarray(f['transform_matrix']),f['fl_x'],f['fl_y'],f['cx'],f['cy'],z['W64']>0,z['unknown'],z['valid']).prepare(0)
  for offset in range(0,len(bg),32768):
   ids=np.arange(offset,min(offset+32768,len(bg)));ids=ids[bg[ids]<2];lo,hi=cell_bounds(ids,128,bounds);_,_,clear,_=classify_cells([c],lo,hi);bg[ids]+=clear.astype('uint8')
 base=(bg<2).reshape((128,)*3);pad=maximum_filter(base,size=3,mode='constant',cval=False);evaluate,kernel=load_kernel(out);tested=crosscheck(evaluate);gates=np.stack([base,pad]);counts=np.ones((2,1,128,128,128),np.uint8);reports=[]
 for f in tf['frames']:
  with np.load(data/'semantics'/f"{f['physical_camera']}.npz") as z:m=z['black']|z['grey']
  y,x=np.where(m);pose=np.asarray(f['transform_matrix']);d=np.c_[(x+.5-f['cx'])/f['fl_x'],-(y+.5-f['cy'])/f['fl_y'],-np.ones(len(x))]@pose[:3,:3].T;d/=np.linalg.norm(d,axis=1)[:,None];hit,support=evaluate(pose[:3,3],d,bounds,gates,counts)
  reports.append(dict(camera=f['physical_camera'],actor_pixels=len(x),box_misses=int((~hit).sum()),base_ray_misses=int((hit&(support[0,:,0]<0)).sum()),pad1_ray_misses=int((hit&(support[1,:,0]<0)).sum())))
 np.savez_compressed(out/'diagnostic_grid.npz',gate0=base,pad1=pad,aabb=bounds)
 write(out/'receipt.json',dict(kind='DIAGNOSTIC_ONLY_3VIEW_FRESH1665_GATE_SMOKE',eligible_for_training=False,records=reports,counts=dict(base=int(base.sum()),pad1=int(pad.sum())),kernel=kernel,independent_kernel_crosscheck=tested,semantics_coordinate_policy='NativeRAW->baselineAREA+singlecalibratedremap',no_model_or_learned_grid_loaded=True))
 print(json.dumps(reports),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
