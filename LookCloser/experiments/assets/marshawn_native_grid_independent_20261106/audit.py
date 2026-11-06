"""Independent CPU audit; no source mutations, model loading, or GPU."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image,ImageDraw
import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,default=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106'));parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);parser.add_argument('--grid-dir',type=Path);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
B=args.campaign/'native_semantic_grid_20000';O=args.output;C=args.grid_dir or B/'corrected_v2';T=B.parent/'data'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((C/'grid_receipt.json').read_text());tf=json.loads((T/'transforms.json').read_text());frames={f['physical_camera']:f for f in tf['frames']}
z=np.load(C/'semantic_grid.npz');box=z['aabb'];frame=z['world_to_box'];N=128;pitch=(box[1]-box[0])/N
accepted=B/'code/marshawn_1660_1900_floor.json';assert sha(accepted)==r['box_sha256']
ids=np.arange(N**3);ijk=np.stack([ids//N**2,(ids//N)%N,ids%N],1);lo=box[0]+ijk*pitch
assert np.array_equal(np.ravel_multi_index(ijk.T,(N,N,N)),ids)
assert np.array_equal(np.floor(((lo+pitch/2)-box[0])/pitch).astype(int),ijk)
count=np.zeros(N**3,np.uint8);corners=np.array([[x,y,z]for x in [0,1]for y in [0,1]for z in [0,1]])*pitch
start=time.time();src=[]
for row in r['sources']:
 name=row['camera'];assert name not in ['DC','HC','KA'];assert sha(row['archive'])==row['archive_sha256'];assert sha(row['semantic'])==row['semantic_sha256']
 with np.load(row['archive'])as a:support=a['support'].copy() if 'support' in a else a['W64']>0;unknown=a['unknown'].copy() if 'unknown' in a else a['source_ineligible'].copy()
 with np.load(row['semantic'])as a:valid=a['valid'].copy()
 forbidden=(support&valid)|unknown|~valid
 sat=np.pad(np.cumsum(np.cumsum(forbidden,dtype=np.int64,axis=0),dtype=np.int64,axis=1),((1,0),(1,0)))
 f=frames[name];pose=np.array(f['transform_matrix']);w,h=f['w'],f['h'];active=np.flatnonzero(count<2)
 for startid in range(0,len(active),16384):
  ix=active[startid:startid+16384];camera=(lo[ix,None]+corners[None]-pose[:3,3])@pose[:3,:3];depth=-camera[:,:,2];front=(depth>1e-5).all(1)
  uv=camera[:,:,:2]/np.where(depth[:,:,None]==0,1,depth[:,:,None]);uv*=np.array([f['fl_x'],-f['fl_y']]);uv+=np.array([f['cx'],f['cy']])
  lower=np.floor(uv.min(1)-1).astype(int);upper=(np.ceil(uv.max(1))+1).astype(int)
  inside=(lower>=0).all(1)&(upper<=np.array([w,h])).all(1)&front
  q=ix[inside];l=lower[inside];u=upper[inside];bad=sat[u[:,1],u[:,0]]-sat[l[:,1],u[:,0]]-sat[u[:,1],l[:,0]]+sat[l[:,1],l[:,0]]
  count[q]+=bad==0
 src.append(dict(camera=name,classified=len(active),rejected=int((count>=2).sum())))
 if len(src)%8==0:print(json.dumps(dict(cameras=len(src),elapsed=time.time()-start)),flush=True)
gate=(count<2).reshape((N,N,N));pad=binary_dilation(gate,structure=np.ones((3,3,3)),iterations=1,border_value=0)
report=dict(accepted_box_sha256=sha(accepted),grid_sha256=sha(C/'semantic_grid.npz'),all_cells=N**3,grid0_mismatches=int(np.count_nonzero(gate!=z['semantic_gate0'])),pad1_mismatches=int(np.count_nonzero(pad!=z['semantic_gate_pad1'])),count_mismatches=int(np.count_nonzero(count.reshape((N,)*3)!=z['clear_background_count_capped2'])),sources=src,gate0_count=int(gate.sum()),pad1_count=int(pad.sum()),elapsed=time.time()-start)
assert report['grid0_mismatches']==report['pad1_mismatches']==report['count_mismatches']==0,report
checks={}
for name in ['DC','HC','KA']:
 receipt=json.loads((B/'corrected_v2/renders'/name/'receipt.json').read_text());native=np.array(receipt['camera_runtime']['transform_matrix']);native=np.vstack([native,[0,0,0,1]])if native.shape==(3,4)else native;floor=np.array(frames[name]['transform_matrix']);checks[name]=dict(pose_absmax=float(np.max(np.abs(frame@native-floor))),fx_absdiff=abs(frames[name]['fl_x']-receipt['camera_runtime']['fl_x']))
report['floor_native_camera_checks']=checks
for axis in [0,1,2]:
 atlas=Image.new('RGB',(8*132,16*148),(20,20,20));draw=ImageDraw.Draw(atlas)
 for index in range(N):
  core=np.take(gate,index,axis=axis);halo=np.take(pad,index,axis=axis);rgb=np.zeros((N,N,3),np.uint8);rgb[halo]=[210,165,35];rgb[core]=[65,190,130];x=(index%8)*132;y=(index//8)*148;atlas.paste(Image.fromarray(rgb[::-1]),(x,y+16));draw.text((x,y),f'{"XYZ"[axis]}={index}',fill='white')
 atlas.save(O/f'all_slices_axis{axis}.png')
np.savez_compressed(O/'independent_classification.npz',clear_background_count_capped2=count.reshape((N,)*3))
(O/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='sources'}),flush=True)
