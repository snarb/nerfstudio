#!/usr/bin/env python3
"""Fixed24 novel HD views inside the observed rig arc in prepared-world coordinates.

The B-height camera tier avoids low floor-level views and the unobserved rear
half of the rig. No held-out RGB or learned field is read.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from PIL import Image, ImageDraw
from audit_marshawn_geometry import project_points


def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def generate(data,boxes_file,output):
 if output.exists():raise FileExistsError(output)
 meta=json.loads((data/'transforms.json').read_text());boxes=json.loads(boxes_file.read_text());allrows=meta['frames'];train=set(meta['train_filenames'])
 if boxes['normalization_sha256']!=sha(data/'preparation.json'):raise ValueError('Box normalization does not match prepared data')
 anchors=[r for r in allrows if r['file_path'] in train and r['physical_camera'].endswith('B')]
 center=np.array(boxes['scopes']['subject']['medium']).mean(0)
 # All current B-tier anchors lie on one observed arc, without crossing ±pi.
 angles=np.array([np.arctan2(r['transform_matrix'][1][3]-center[1],r['transform_matrix'][0][3]-center[0]) for r in anchors])
 order=np.argsort(angles);anchors=[anchors[i] for i in order];angles=angles[order]
 if len(anchors)<3 or np.diff(angles).max()>np.deg2rad(25):raise ValueError('Insufficient adjacent anchors or an unobserved angular gap')
 poses=np.array([r['transform_matrix'] for r in anchors],float);positions=poses[:,:3,3]
 distances=np.linalg.norm(np.diff(positions,axis=0),axis=1);arc=np.r_[0,np.cumsum(distances)]
 target=(np.arange(24)+.5)/24*arc[-1];path=[];records=[];rows=[]
 w,h=int(anchors[0]['w']),int(anchors[0]['h'])
 if any((r['w'],r['h'])!=(w,h) for r in anchors):raise ValueError('Consistent prepared HD raster required')
 for i,s in enumerate(target):
  j=min(int(np.searchsorted(arc,s,side='right')-1),len(anchors)-2);t=float((s-arc[j])/distances[j]);pose=np.eye(4)
  pose[:3,:3]=Slerp([0.,1.],Rotation.from_matrix(poses[j:j+2,:3,:3]))([t]).as_matrix()[0]
  pose[:3,3]=(1-t)*positions[j]+t*positions[j+1]
  f0=np.sqrt(anchors[j]['fl_x']*anchors[j]['fl_y']);f1=np.sqrt(anchors[j+1]['fl_x']*anchors[j+1]['fl_y']);f=(1-t)*f0+t*f1
  fov=float(np.degrees(2*np.arctan(h/(2*f))))
  path.append(dict(camera_to_world=pose.reshape(-1).tolist(),fov=fov,aspect=w/h))
  row=dict(transform_matrix=pose.tolist(),fl_x=f,fl_y=f,cx=w/2,cy=h/2,w=w,h=h);rows.append(row)
  records.append(dict(index=i,anchors=[anchors[j]['physical_camera'],anchors[j+1]['physical_camera']],fraction=t,position=pose[:3,3].tolist(),focal=f,vertical_fov_degrees=fov,baseline=float(distances[j])))
 # One fixed focal for the entire path avoids FOV pumping and source-tier head cuts.
 loose_lo,loose_hi=np.array(boxes['scopes']['subject']['loose'])
 loose_corners=np.array([[x,y,z] for x in [loose_lo[0],loose_hi[0]] for y in [loose_lo[1],loose_hi[1]] for z in [loose_lo[2],loose_hi[2]]])
 caps=[]
 for row in rows:
  uv,front=project_points(row,loose_corners)
  if not front.all():raise ValueError('Subject crosses camera plane')
  normalized=(uv-[w/2,h/2])/row['fl_x']
  caps.append(min(row['fl_x'],.90*w/2/np.abs(normalized[:,0]).max(),.90*h/2/np.abs(normalized[:,1]).max()))
 common_focal=float(min(caps))
 for entry,row,record in zip(path,rows,records):
  record['source_interpolated_focal']=record['focal'];record['focal']=common_focal
  record['vertical_fov_degrees']=float(np.degrees(2*np.arctan(h/(2*common_focal))))
  row['fl_x']=row['fl_y']=common_focal;entry['fov']=record['vertical_fov_degrees']
 result=dict(camera_type='perspective',render_width=w,render_height=h,fps=6,seconds=4.,is_cycle=False,camera_path=path)
 output.mkdir(parents=True);(output/'camera_path.json').write_text(json.dumps(result,indent=2)+'\n')
 # Parse through actual nerfstudio API, checking camera count/poses/focals/raster.
 from nerfstudio.cameras.camera_paths import get_path_from_json
 parsed=get_path_from_json(result)
 if parsed.size!=24:raise AssertionError('Wrong camera count')
 if not np.allclose(parsed.camera_to_worlds.numpy(),np.array([r['transform_matrix'] for r in rows])[:,:3],atol=1e-6):raise AssertionError('Pose parse mismatch')
 if not np.allclose(parsed.fx.numpy().reshape(-1),[r['fl_x'] for r in rows],rtol=1e-6):raise AssertionError('Focal parse mismatch')
 lo,hi=np.array(boxes['scopes']['subject']['medium']);corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
 npz=np.load(boxes_file.parent/'train_witnesses.npz');points=npz['xyz'];subject=points[((points>=lo)&(points<=hi)).all(1)]
 sheet=Image.new('RGB',(6*320,4*210));draw=ImageDraw.Draw(sheet);coverage=[]
 for i,row in enumerate(rows):
  tile=Image.new('RGB',(320,169),'#15191f');d=ImageDraw.Draw(tile);uv,front=project_points(row,corners);projected,front_points=project_points(row,subject);scale=320/w
  for u,v in projected[front_points]:
   if 0<=u<w and 0<=v<h:d.ellipse((u*scale-1,v*scale-1,u*scale+1,v*scale+1),fill='#44bbaa')
  for k in range(8):
   for bit in [1,2,4]:
    n=k^bit
    if n>k and front[k] and front[n]:d.line([tuple(uv[k]*scale),tuple(uv[n]*scale)],fill='#ffcc44',width=1)
  inside=front&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
  coverage.append(dict(index=i,all_subject_corners_in_front=bool(front.all()),corners_inside_raster=int(inside.sum()),projected_bbox=[uv.min(0).tolist(),uv.max(0).tolist()]))
  x=i%6*320;y=i//6*210;sheet.paste(tile,(x,y+35));r=records[i];draw.text((x+3,y+3),f'{i:02d} {r["anchors"][0]}→{r["anchors"][1]} t={r["fraction"]:.2f}\n8 corners inside: {inside.sum()==8}',fill='white')
 sheet.save(output/'subject_projection_contact.jpg')
 receipt=dict(prepared_world=True,views=24,heldout_rgb_used=False,direct_train_rgb_read=False,train_derived_sparse_witnesses_used=True,anchors=[r['physical_camera'] for r in anchors],path_type='Open observed arc; adjacent B-tier anchors; positions interpolate linearly; rotations quaternion SLERP; no rear wrap',focal_policy='One common isotropic focal <= all interpolated source focals, fitting loose subject box with 5% margin on each image edge; centered principal point' ,raster=[w,h],normalization_sha256=sha(data/'preparation.json'),transforms_sha256=sha(data/'transforms.json'),boxes_sha256=sha(boxes_file),path_sha256=sha(output/'camera_path.json'),script_sha256=sha(__file__),witnesses_sha256=sha(boxes_file.parent/'train_witnesses.npz'),parse_smoke_passed=True,records=records,projection_coverage=coverage,limitation='Projection contact is a calibrated diagram of sparse witnesses and box; not a model render or novel-view ground truth')
 (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(dict(output=str(output),anchors=len(anchors),views=24,raster=[w,h],minimum_in_raster_corners=min(r['corners_inside_raster'] for r in coverage),parse_smoke_passed=True)))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--boxes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();generate(a.data,a.boxes,a.output)
