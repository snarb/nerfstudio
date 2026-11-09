"""Audit sparse COLMAP candidates against a floor-aligned review box.

Mask votes are geometric support, not a semantic identity certificate. Preserve
all outside candidates for source-image review; do not discard low-vote points
by proximity to more visible body regions.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,numpy as np,concurrent.futures,time
p=argparse.ArgumentParser();p.add_argument('--first',type=int);p.add_argument('--last',type=int);p.add_argument('--sources',type=Path);p.add_argument('--box',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();box=json.loads(a.box.read_text());F=np.array(box['rig_world_to_floor']);Q=np.array(box['floor_to_box_rotation']);bounds=np.array(box['floor_bounds_rotated_rig_units']);cams={d.name:json.loads((d/'receipt.json').read_text())['camera_row']for d in a.sources.iterdir()};six={'AE','BE','KE','LD','LE','ME'};a.output.mkdir(parents=True,exist_ok=True)
def audit(f):
 path=Path('/tmp/marshawn_seeds_908_1268')/f'{f:06d}'/'sparse/text/points3D.txt';rows=[s.split()for s in path.read_text().splitlines()if s and not s.startswith('#')];xyz=np.array([[float(v)for v in r[1:4]]for r in rows]);qual=np.array([float(r[7])<2 and len(r)>=14 for r in rows]);votes=np.zeros(len(xyz),int);w=896+16*((f-896)//16)
 for cam,c in cams.items():
  path=Path('/raid/lookcloser/mask8k')/('coarse6'if cam in six else'coarse66')/f'w{w}_f21/masks/props/subj/f1ds8'/cam/f'{f:06d}.npy';mask=np.load(path);T=np.array(c['transform_matrix']);v=(xyz-T[:3,3])@T[:3,:3];z=-v[:,2];u=np.floor((v[:,0]/z*c['fl_x']+c['cx'])*mask.shape[1]/c['w']).astype(int);vv=np.floor((-v[:,1]/z*c['fl_y']+c['cy'])*mask.shape[0]/c['h']).astype(int);ok=(z>0)&(u>=0)&(u<mask.shape[1])&(vv>=0)&(vv<mask.shape[0]);ix=np.where(ok)[0];votes[ix]+=mask[vv[ix],u[ix]]
 strong=(votes>=40)&qual;keep=qual&(votes>=24);rot=(xyz@F[:3,:3].T+F[:3,3])@Q.T;inside=((rot>=bounds[0])&(rot<=bounds[1])).all(1);outside=keep&~inside;r=dict(frame=f,total=len(xyz),strong=int(strong.sum()),qualified=int(keep.sum()),outside=int(outside.sum()),strong_outside=int((strong&~inside).sum()),bounds=np.stack([rot[keep].min(0),rot[keep].max(0)]).tolist(),outside_points=rot[outside].tolist(),outside_votes=votes[outside].tolist())
 if outside.any():np.savez_compressed(a.output/f'{f}_outside.npz',world_xyz=xyz[outside],box_xyz=rot[outside],votes=votes[outside])
 return r
records=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=6)as pool:
 for r in pool.map(audit,range(a.first,a.last+1)):
  records.append(r);(a.output/'progress.json').write_text(json.dumps(dict(done=len(records),frame=r['frame'],outside=r['outside'])));print(r['frame'],r['qualified'],r['outside'],flush=True)
(a.output/'summary.json').write_text(json.dumps(dict(frames=records,frame_range=[a.first,a.last],criterion='error<2,>=3 tracks,>=24 of72 coarse masks without proximity filtering; sparse only; fixed rig from source receipts',bounds= bounds.tolist()),indent=2))
