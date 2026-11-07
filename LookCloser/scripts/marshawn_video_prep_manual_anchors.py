"""Source-bound manual multi-view actor landmark triangulation, no training."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image,ImageDraw
from audit_marshawn_geometry import projection
from prepare_marshawn_tuning import write,sha

def images(root):
 tail=root/'tail_native';out=tail/'manual_anchor_sources';out.mkdir(exist_ok=True);records=[]
 for frame in [1960,2020]:
  for camera in ['HC','DC','KA','FC','MC','NA']:
   r=json.loads((tail/'sources'/str(frame)/camera/'receipt.json').read_text());im=Image.open(r['image']).convert('RGB');d=ImageDraw.Draw(im)
   for x in range(0,im.width,100):d.line((x,0,x,im.height),fill=(160,160,160),width=1);d.text((x+2,12),str(x),fill='yellow',stroke_width=1,stroke_fill='black')
   for y in range(0,im.height,100):d.line((0,y,im.width,y),fill=(160,160,160),width=1);d.text((5,y+2),str(y),fill='yellow',stroke_width=1,stroke_fill='black')
   dest=out/f'{frame}_{camera}_HD_array_xy.jpg';im.save(dest,quality=96);records.append(dict(frame=frame,camera=camera,image=str(dest),source_receipt_sha256=sha(tail/'sources'/str(frame)/camera/'receipt.json'),coordinate_convention='Arrayxyclicks; helperadds.5forCOLMAPcentres'))
 write(out/'manifest.json',dict(records=records))

def triangulate(root,request,output):
 tail=root/'tail_native';T=np.asarray(json.loads((root/'frames/001660/data/box.json').read_text())['world_to_box']);results=[]
 for item in json.loads(request.read_text())['landmarks']:
  matrices=[];equations=[];sources=[]
  for camera,point in item['array_xy'].items():
   source=tail/'sources'/str(item['frame'])/camera/'receipt.json';r=json.loads(source.read_text());p,*_=projection(r['camera_row']);uv=np.asarray(point)+.5;equations.extend([uv[0]*p[2]-p[0],uv[1]*p[2]-p[1]]);matrices.append((camera,p,uv));sources.append(dict(camera=camera,receipt_sha256=sha(source)))
  if len(matrices)<2:raise ValueError('At least2 independent calibrated views per landmark')
  _,_,v=np.linalg.svd(equations);xyz=v[-1,:3]/v[-1,3];errors=[]
  for camera,p,uv in matrices:
   q=p@np.r_[xyz,1]
   if q[2]<=0:raise ValueError('Landmark behind camera')
   errors.append(dict(camera=camera,array_xy=(uv-.5).tolist(),reprojection_error=float(np.linalg.norm(q[:2]/q[2]-uv))))
  if max(r['reprojection_error'] for r in errors)>2:raise ValueError('Manual point correspondence exceeds2HDpx reprojection')
  results.append(dict(frame=item['frame'],name=item['name'],world_xyz=xyz.tolist(),floor_xyz=(T[:3,:3]@xyz+T[:3,3]).tolist(),observations=errors,sources=sources))
 write(output,dict(request_sha256=sha(request),records=results,no_field_or_learned_state=True))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--request',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();triangulate(a.root,a.request,a.output) if a.request else images(a.root)
