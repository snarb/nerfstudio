"""Finite CPU-only conservative lower bound. Never writes a model/runtime grid."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
from scipy.ndimage import maximum_filter
C=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107');O=Path(__file__).parent;S=C/'code/LookCloser/scripts';sys.path.insert(0,str(S))
from mask_visual_hull_aabb import MaskCamera,classify_cells
from build_marshawn_semantic_resolution import cell_bounds

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.time();data=C/'frames/001670/data';snap=C/'operations/contour_runtime_001670/step-000139999_grid.npz';meta=snap.with_suffix('.json');m=json.loads(meta.read_text());assert sha(snap)==m['grid_archive_sha256'];assert m['checkpoint_sha256']=='ccd0da52ad069beaaf93c7ebd3e9b79343242b0d7d0f7e6ce63e9c332cc42015'
 with np.load(snap)as z:active=z['active'][0].copy();bounds=z['cascade_aabbs'].reshape(2,3).copy()
 ids=np.flatnonzero(active.ravel());lo,hi=cell_bounds(ids,128,bounds);assert len(ids)==37821
 tf=json.loads((data/'transforms.json').read_text());frames={r['physical_camera']:r for r in tf['frames']if r['file_path']in tf['train_filenames']};assert len(frames)==66
 seen=np.zeros(len(ids),bool);records=[]
 for name,row in sorted(frames.items()):
  archive=data/'semantics'/(name+'.npz');digest=sha(archive)
  with np.load(archive,allow_pickle=False)as z:
   support=z['W64']>0;valid=z['valid'];invalid_positive=int((support&~valid).sum())
  # Deliberately retain ALL literal positiveW64 pixels, including any invalid ones.
  cam=MaskCamera(name,np.asarray(row['transform_matrix']),row['fl_x'],row['fl_y'],row['cx'],row['cy'],support,np.zeros_like(support),np.ones_like(support)).prepare(0)
  fg,possible,_,_=classify_cells([cam],lo,hi);keep=possible>0;before=int(seen.sum());seen|=keep
  records.append(dict(camera=name,semantic_sha256=digest,positive_W64_pixels=int(support.sum()),positive_outside_valid_pixels=invalid_positive,projected_bbox_positive_cells=int((fg>0).sum()),near_crossing_retained_cells=int(((possible>0)&(fg==0)).sum()),new_mandatory=int(seen.sum())-before,mandatory_union=int(seen.sum())))
  print(json.dumps(records[-1]),flush=True)
 dense=np.zeros_like(active);dense.ravel()[ids[seen]]=True;halo=maximum_filter(dense,size=3,mode='constant',cval=False)&active;n=int(halo.sum());ratio=n/len(ids)
 source={str(p):sha(p)for p in [snap,meta,data/'ready.json',data/'transforms.json',S/'mask_visual_hull_aabb.py',S/'build_marshawn_semantic_resolution.py',Path(__file__)]}
 proof=dict(kind='READ_ONLY_POSITIVE_W64_FULL_RAY_SUPPORT_FEASIBILITY_V1',frame=1670,checkpoint_sha256=m['checkpoint_sha256'],source_files_sha256=source,source_camera_count=66,original_R0_cells=len(ids),fullray_projection_mandatory_cells=int(seen.sum()),one_halo_clipped_to_R0_cells=n,retained_fraction=ratio,prunable_cells=len(ids)-n,records=records,seconds=time.time()-start,GPU_used=False,checkpoint_loaded=False,production_changed=False,decision='STOP_NO_GPU_MORE_THAN_95_PERCENT_RETAINED'if ratio>.95 else 'CPU_FEASIBLE_REQUIRES_PARENT_REVIEW_BEFORE_ANY_GPU',reasoning=['Frozen qualified all8corner projection with outward pixel footprint; integral contains literalW64>0 including protective positive band.','Any projected bbox positive retains entire original occupiedcell, at every depth; near-plane crossings retained. Fully behindcamera cells cannot intersect forward supervisedrays.','One-cell halo then clip originalR0. This is conservative footprint overretention, not physicalgeometry certification.','Retained full source-ray intervals avoid legacyARM global budget resampling changes on positive source rays; pixelbyteparity remains an unmeasured claim, no rerender performed.','Old quantile10/50/90 caches only finite foralpha>=.95; they are not used as evidence for lowalpha absence.','A total-mass tailbound alone cannot bound changedrerenderRGB under corrected_arm_allocator=false because occupancyinterval pruning alters globalintervalcount/samplepositions.','This pilot does not remove semanticboundaries, changeRGB/density/AABB or invent another branch.'])
 (O/'receipt.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items()if k not in ['records','source_files_sha256']},indent=2))
if __name__=='__main__':main()
