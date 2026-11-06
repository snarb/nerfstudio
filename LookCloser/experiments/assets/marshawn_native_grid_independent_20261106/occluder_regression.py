"""Actual independently triangulated TRAIN toe-cell regression; CPU only."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--campaign',type=Path,default=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106'))
p.add_argument('--mat-manifest',type=Path,required=True)
p.add_argument('--new-grid',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();B=a.campaign/'native_semantic_grid_20000';T=a.campaign/'data'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
m=json.loads(a.mat_manifest.read_text());masks={r['camera']:r for r in m['records']};fs={f['physical_camera']:f for f in json.loads((T/'transforms.json').read_text())['frames']}
source=B/'corrected_v2/train_shoe_feature_depth.json';point=np.array(json.loads(source.read_text())['points'][0]['xyz_floor']);z=np.load(B/'margin_W32/semantic_grid.npz');box=z['aabb'];pitch=(box[1]-box[0])/128;idx=np.floor((point-box[0])/pitch).astype(int);corners=box[0]+idx*pitch+np.array([[x,y,z]for x in [0,1]for y in [0,1]for z in [0,1]])*pitch
rows=[]
for name,f in fs.items():
 if name in ['DC','HC','KA']:continue
 pose=np.array(f['transform_matrix']);q=(corners-pose[:3,3])@pose[:3,:3];depth=-q[:,2];uv=q[:,:2]/depth[:,None]*[f['fl_x'],-f['fl_y']]+[f['cx'],f['cy']];lo=np.floor(uv.min(0)-1).astype(int);hi=(np.ceil(uv.max(0))+1).astype(int);full=(depth>1e-5).all()and(lo>=0).all()and(hi<=[1920,1012]).all();sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]))
 with np.load(B/'margin_W32/archives'/(name+'.npz'))as old:support=old['support'];unknown=old['unknown'];valid=old['valid']
 row=masks[name];assert sha(row['mask_path'])==row['mask_sha256']
 with np.load(row['mask_path'])as mat:extra=mat[row['mask_key']]
 assert extra.dtype==np.bool_ and extra.shape==(1012,1920)
 # These three views have unqualified mat visibility; absence of annotation is not clear evidence.
 if name in ['CE','JE','NE']:extra=extra|valid
 old_clear=bool(full and not(support[sl]|unknown[sl]|~valid[sl]).any());new_clear=bool(full and not(support[sl]|unknown[sl]|extra[sl]|~valid[sl]).any())
 rows.append(dict(camera=name,old_clear=old_clear,new_clear=new_clear,footprint=[*lo.tolist(),*hi.tolist()]))
assert {r['camera']for r in rows if r['old_clear']}=={'DE','EE'}
assert not any(r['new_clear']for r in rows)
new_check=None
if a.new_grid:
 with np.load(a.new_grid)as new:
  assert np.array_equal(new['aabb'],box);assert new['semantic_gate0'][tuple(idx)];assert new['semantic_gate_pad1'][tuple(idx)]
 new_check=dict(path=str(a.new_grid),sha256=sha(a.new_grid),gate0=True,pad1=True)
r=dict(passed=True,source_point=str(source),source_point_sha256=sha(source),xyz_floor=point.tolist(),index_xyz=idx.tolist(),mask_manifest_sha256=sha(a.mat_manifest),old_clear_views=['DE','EE'],new_clear_views=[],visibility_unqualified_clear_vote_disabled=['CE','JE','NE'],new_grid_check=new_check,rows=rows,not_surface_truth='One independently triangulated toe-upper feature only; neither sole nor complete hidden geometry.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='rows'}))
