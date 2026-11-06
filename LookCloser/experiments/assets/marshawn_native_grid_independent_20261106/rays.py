"""Independent ray-vs-every-admitted-cell slabs (no plane partition or model)."""
import numpy as np,json,time
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,default=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106'));parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
B=args.campaign/'native_semantic_grid_20000';C=B/'corrected_v2';T=B.parent/'data';O=args.output
z=np.load(C/'semantic_grid.npz');box=z['aabb'];g=z['semantic_gate_pad1'];pitch=(box[1]-box[0])/128;idx=np.argwhere(g);lo=box[0]+idx*pitch;hi=lo+pitch
frames={f['physical_camera']:f for f in json.loads((T/'transforms.json').read_text())['frames']};records={r['camera']:r['semantic']for r in json.loads((T/'ready.json').read_text())['records']}
for r in json.loads((C/'mask_corrections_final.json').read_text())['records']:records[r['camera']]=r['corrected_archive']
def hit(origin,d):
 par=d==0;safe=np.where(par,1,d);a=(lo-origin)/safe;b=(hi-origin)/safe
 entry=np.maximum(np.where(par,-np.inf,np.minimum(a,b)).max(1),0);leave=np.where(par,np.inf,np.maximum(a,b)).min(1)
 return (leave>entry)&~(par&((origin<lo)|(origin>=hi))).any(1)
def ray(f,x,y):
 pose=np.array(f['transform_matrix']);d=np.array([(x+.5-f['cx'])/f['fl_x'],-(y+.5-f['cy'])/f['fl_y'],-1.])@pose[:3,:3].T;return pose[:3,3],d/np.linalg.norm(d)
misses=np.load(C/'train_ray_misses.npz');rows=[]
for name,f in frames.items():
 if name in ['DC','HC','KA']:continue
 with np.load(records[name])as m:mask=m['black']|m['grey']
 points=misses[name+'_semantic_gate_pad1'];qualified=points[mask[points[:,1],points[:,0]]];observed=[]
 for x,y in qualified:observed.append(bool(hit(*ray(f,x,y)).any()))
 if len(qualified):rows.append(dict(camera=name,qualified_prior_misses=len(qualified),independent_hit_count=sum(observed),pixels=qualified.tolist()))
# Broad positive spot-check spanning all 63 TRAIN views (not full coverage rerun).
rng=np.random.default_rng(1907);positive=[]
for name,f in frames.items():
 if name in ['DC','HC','KA']:continue
 with np.load(records[name])as m:ys,xs=np.where(m['black']|m['grey'])
 chosen=rng.choice(len(xs),5,replace=False);hits=[bool(hit(*ray(f,xs[k],ys[k])).any())for k in chosen];positive.append(dict(camera=name,rays=5,hits=sum(hits)))
point=json.loads((C/'train_shoe_feature_depth.json').read_text())['points'][0];p=np.array(point['xyz_floor']);i=np.floor((p-box[0])/pitch).astype(int)
report=dict(method='Independent slab intersection against all50790 admitted cells; strict positive ray lengths, half-open parallel slabs; no existing DDA imported',preferred_grid_array='semantic_gate_pad1',qualified_prior_misses=sum(r['qualified_prior_misses']for r in rows),misses_independently_confirmed=sum(r['qualified_prior_misses']-r['independent_hit_count']for r in rows),miss_rows=rows,random_train_rays=sum(r['rays']for r in positive),random_train_hits=sum(r['hits']for r in positive),random_rows=positive,independent_train_toe_upper_point=dict(xyz_floor=p.tolist(),index_xyz=i.tolist(),gate0=bool(z['semantic_gate0'][tuple(i)]),pad1=bool(g[tuple(i)])),limitations=['Random positive sample does not certify all rays or surfaces.','Single toe-upper feature does not certify sole/contact volumes.'])
(O/'ray_receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['random_rows','miss_rows']}))
