#!/usr/bin/env python3
"""Project actual traced near-cloud density samples into real train-camera frusta/GT.
Frustum inclusion is not physical visibility. No model/grid changes or rendering.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from audit_marshawn_geometry import project_points

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;a.output.mkdir(parents=True,exist_ok=False);source=r/'color_variants/gain4_scalar_trainfit_poseeval/sdr_png';meta=json.loads((source/'transforms.json').read_text());train=set(meta['train_filenames']);frames=[f for f in meta['frames']if f['file_path']in train];obs=np.load(r/'frequency_seed/observations.npz');xyz=np.load(r/'geometry/train_witnesses.npz')['xyz'];camera_names=obs['camera_names'].tolist();idx_by_cam={name:np.flatnonzero(obs['camera_indices']==i)for i,name in enumerate(camera_names)};trees={name:cKDTree(obs['uv'][ids])for name,ids in idx_by_cam.items()};cloud=r/'final_pair/release_audit/novel_cloud_traces/audit.json';audit=json.loads(cloud.read_text());rows=[];images={}
 for qi,q in enumerate(audit['rows']):
  query=q['query'];native=q['profiles']['native'];samples=[('peak',native['peak'])]+[('q50',x)for x in native['termination_quantiles']if x['quantile']==.5]
  for kind,sample in samples:
   point=np.array(sample['xyz']);label=f'view{query["view"]:02d}_x{query["x"]}_y{query["y"]}_{kind}';support=[]
   for f in frames:
    uv,front=project_points(f,point[None]);u,v=uv[0];pose=np.array(f['transform_matrix']);local=(point-pose[:3,3])@pose[:3,:3];depth=-local[2];inside=bool(front[0]and 0<=u<f['w']and 0<=v<f['h']);distance=float(np.linalg.norm(point-pose[:3,3]));entry=dict(camera=f['physical_camera'],uv=[float(u),float(v)],camera_z=float(depth),camera_distance=distance,inside_frustum=inside)
    if inside:
     name=f['physical_camera'];duv,j=trees[name].query([u,v]);ob=int(idx_by_cam[name][j]);pi=int(obs['point_indices'][ob]);wlocal=(xyz[pi]-pose[:3,3])@pose[:3,:3];entry.update(nearest_actual_track_observation=ob,nearest_track_point=pi,nearest_track_pixel_distance=float(duv),nearest_track_camera_z=float(-wlocal[2]),track_minus_cloud_depth=float(-wlocal[2]-depth));support.append(entry)
   support.sort(key=lambda x:x['camera_distance']);panels=[]
   for fentry in support:
    f=next(f for f in frames if f['physical_camera']==fentry['camera']);name=fentry['camera'];path=source/f['file_path']
    if name not in images:images[name]=Image.open(path).convert('RGB')
    u,v=fentry['uv'];crop=images[name].crop((round(u)-100,round(v)-70,round(u)+100,round(v)+70));draw=ImageDraw.Draw(crop);draw.ellipse((96,66,104,74),outline='red',width=2);panel=Image.new('RGB',(320,200),'#161616');panel.paste(crop,(60,55));d=ImageDraw.Draw(panel);d.text((3,3),name+' realTRAIN GT at projected '+kind,fill='white');d.text((3,17),f'z={fentry["camera_z"]:.4f}; track p{fentry["nearest_track_point"]}',fill='white');d.text((3,31),f'UVgap={fentry["nearest_track_pixel_distance"]:.1f}px dz={fentry["track_minus_cloud_depth"]:.4f}',fill='white');panels.append(panel)
   pages=[]
   for start in range(0,len(panels),12):
    canvas=Image.new('RGB',(1280,640),'#101010');d=ImageDraw.Draw(canvas);d.text((3,3),label+' | red = cloud XYZ projection; frustum inclusion DOES NOT prove visibility',fill='yellow')
    for j,im in enumerate(panels[start:start+12]):canvas.paste(im,((j%4)*320,(j//4)*200+40))
    name=label+f'_GT_{start//12+1}.jpg';canvas.save(a.output/name,quality=94);pages.append(name)
   rows.append(dict(label=label,query=query,kind=kind,xyz=point.tolist(),distance=float(sample['distance']),sigma=float(sample['sigma']),source_declared_frustum_count=sample['train_frustum_count'],independent_frustum_count=len(support),train_projections=support,GT_pages=pages))
 report=dict(source_cloud_audit_sha256=sha(cloud),source_transforms_sha256=sha(source/'transforms.json'),source_points_sha256=sha(r/'geometry/train_witnesses.npz'),source_observations_sha256=sha(r/'frequency_seed/observations.npz'),script_sha256=sha(__file__),rows=rows,limitations=['All crops are real training images at projected traced world density locations, not model substitutes.','Insidefrustum only; actual visibility/occlusion not independently proved.','Nearby track pixel/depth gaps are correspondence clues, not wholevoxel free-space labels.','A peak sample and expected-depth buffer are not independently reconstructed surfaces.'])
 (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print([(x['label'],x['independent_frustum_count'],x['source_declared_frustum_count'])for x in rows])
if __name__=='__main__':main()
