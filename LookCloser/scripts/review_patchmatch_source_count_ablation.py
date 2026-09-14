#!/usr/bin/env python3
"""GT-only regions and matched-camera diagnostics for the isolated count study."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path('/mnt/data/dec5_patchmatch_source_count_ablation')
# Half-resolution portrait coordinates, manually selected on staged held-out GT
# before any ablation prediction was seen (2026-09-14). No geometry consumes them.
FACE={
 '001083':[(232,447),(279,416),(323,390),(356,388),(386,413),(406,451),(414,478),(407,508),(392,543),(378,574),(351,604),(326,614),(292,598),(264,568),(246,533),(231,503),(218,481),(216,459)],
 '001123':[(261,450),(303,424),(355,400),(388,402),(415,425),(431,459),(429,494),(420,520),(425,548),(410,556),(399,584),(376,612),(354,622),(331,611),(303,586),(279,552),(274,514),(251,495),(247,468)],
}
TRAIN_CAMERA='G004_A005_121071'
TRAIN_REGIONS={
 '001083':dict(crown_top_band=[(238,257),(267,260),(287,253),(309,263),(333,273),(346,288),(320,295),(290,285),(260,282),(239,280)],crown_interior=[(241,276),(320,279),(375,300),(385,335),(319,344),(269,371),(231,382),(220,333)],cheek_skin=[(249,449),(280,469),(323,489),(325,521),(303,535),(272,518),(249,490)]),
 '001123':dict(crown_top_band=[(249,289),(273,283),(296,289),(320,298),(333,311),(310,319),(284,311),(260,310),(242,306)],crown_interior=[(238,304),(313,302),(368,319),(393,347),(366,372),(318,366),(270,385),(243,406),(220,372)],cheek_skin=[(265,458),(290,469),(327,491),(345,519),(324,539),(291,529),(267,501)]),
}

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def regions():
 for frame,points in FACE.items():
  data=ROOT/'data'/frame/'staged63';payload=read(data/'transforms.json');gt=data/payload['val_filenames'][0]
  polygon=[[1919-2*y,2*x] for x,y in points]
  result=dict(selection_method='manual_polygon_on_heldout_gt_only',prediction_used_for_selection=False,ground_truth_sha256=sha(gt),ground_truth=str(gt),include_polygons=[polygon],exclude_polygons=[],annotation_time='2026-09-14T15:56:00Z',notes='Drawn on the half-resolution portrait GT preview before any ablation render; mapped exactly to native landscape coordinates. Skin/ear only; excludes hair, clothing and background. Cast shadow on skin retained.')
  path=ROOT/'regions'/f'{frame}_face.json'
  if path.exists():assert read(path)==result
  else:write(path,result)
  im=Image.open(gt).convert('RGB');d=ImageDraw.Draw(im);d.line([tuple(p) for p in polygon]+[tuple(polygon[0])],fill='red',width=3)
  im.transpose(Image.Transpose.ROTATE_90).resize((540,960)).save(ROOT/'regions'/f'{frame}_face_preview.png')
  train=next(r for r in payload['frames'] if r['physical_camera']==TRAIN_CAMERA)
  rgb=data/train['file_path'];polygons={name:[[1919-2*y,2*x] for x,y in pts] for name,pts in TRAIN_REGIONS[frame].items()}
  masks={}
  for name,pts in polygons.items():
   path=ROOT/'regions'/f'{frame}_{name}_mask.png';Image.fromarray(mask_for(pts).astype(np.uint8)*255).save(path)
   masks[name]=dict(file=path.name,sha256=sha(path))
  write(ROOT/'regions'/f'{frame}_train.json',dict(physical_camera=TRAIN_CAMERA,file_path=train['file_path'],source_sha256=sha(rgb),polygons=polygons,masks=masks,selection_method='manual_on_train_RGB_before_ablation_predictions',prediction_used_for_selection=False,notes='Interior hair and skin regions; not whole-silhouette masks. No assumption that every silhouette or dark region is missing anatomy. Frozen PNG raster masks avoid differences between PIL versions on local and remote hosts.'))
  im=Image.open(rgb).convert('RGB');d=ImageDraw.Draw(im)
  for pts in polygons.values():d.line([tuple(p) for p in pts]+[tuple(pts[0])],fill='red',width=3)
  im.transpose(Image.Transpose.ROTATE_90).resize((540,960)).save(ROOT/'regions'/f'{frame}_train_preview.png')

def mask_for(points,shape=(1080,1920)):
 canvas=Image.new('L',(shape[1],shape[0]));ImageDraw.Draw(canvas).polygon([tuple(p) for p in points],fill=1)
 return np.array(canvas,dtype=bool)

def mesh_review(frame,mesh,metadata,destination):
 # Explicit CPU raycasts; never reads a target image to construct geometry.
 import open3d as o3d
 from render_patchmatch_camera_path import normalize_frame
 data=ROOT/'data'/frame/'staged63';payload=read(data/'transforms.json')
 meta=read(metadata);m=o3d.io.read_triangle_mesh(str(mesh));v=np.asarray(m.vertices,np.float32);t=np.asarray(m.triangles,np.uint32)
 scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(t))
 reg=read(ROOT/'regions'/f'{frame}_train.json');train=next(r for r in payload['frames'] if r['physical_camera']==TRAIN_CAMERA)
 train=normalize_frame(train,payload,meta)
 oldroot=Path('/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150/frames')/frame/'mesh'
 oldmeta=read(oldroot/'colmap_patchmatch_tsdf.json')
 for key in ['dataparser_transform','dataparser_scale']:assert np.allclose(meta[key],oldmeta[key],atol=1e-7)
 virtual=next(r for r in read('/mnt/data/dec5_screen_travel_dynamic_150_v2/request.json')['inventory'] if r['frame_id']==frame)['camera']
 destination.mkdir(parents=True,exist_ok=True);stats={}
 for name,row in [('train',train),('virtual',virtual)]:
  pose=np.asarray(row['transform_matrix']);ext=np.linalg.inv(pose@np.diag([1.,-1.,-1.,1.])).astype(np.float32)
  k=np.array([[row['fl_x'],0,row['cx']],[0,row['fl_y'],row['cy']],[0,0,1]],np.float32)
  hit=scene.cast_rays(scene.create_rays_pinhole(o3d.core.Tensor(k),o3d.core.Tensor(ext),row['w'],row['h']))
  depth=hit['t_hit'].numpy();ids=hit['primitive_ids'].numpy();good=np.isfinite(depth)
  normals=np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]]);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
  shade=(.25+.75*np.abs(normals@pose[:3,2]))*230
  clay=np.zeros((*good.shape,3),np.uint8);clay[good]=shade[ids[good],None].astype(np.uint8)
  Image.fromarray(np.rot90(clay)).save(destination/f'{name}_clay.png')
  np.savez_compressed(destination/f'{name}_depth.npz',depth=np.where(good,depth,0))
  stats[name]=dict(hit_fraction=float(good.mean()))
  if name=='train':
   for region,points in reg['polygons'].items():
    mask=mask_for(points);stats[name][region]=dict(pixels=int(mask.sum()),mesh_hit_pixels=int((mask&good).sum()),mesh_hit_fraction=float(good[mask].mean()))
   rgb=np.asarray(Image.open(data/reg['file_path']).convert('RGB')).copy();rgb[~good]=(rgb[~good]*.3+[178,0,0]).clip(0,255)
   Image.fromarray(np.rot90(rgb).astype(np.uint8)).save(destination/'train_missing_overlay.png')
 write(destination/'mesh_review.json',dict(frame=frame,mesh=str(mesh),mesh_sha256=sha(mesh),triangles=len(t),stats=stats,regions_sha256=sha(ROOT/'regions'/f'{frame}_train.json')))

def baseline_review():
 for frame in FACE:
  base=Path('/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150/frames')/frame/'mesh'
  mesh_review(frame,base/'colmap_patchmatch_tsdf.ply',base/'colmap_patchmatch_tsdf.json',ROOT/'old_baseline_review'/frame)

def collect():
 remote=Path('/fsx/oregon/dec5_patchmatch_source_count_ablation')
 completed=[]
 for count in [12,24,36]:
  for frame in FACE:
   src=remote/'arms'/frame/f'sources{count}';dst=ROOT/'arms'/frame/f'sources{count}'
   if (dst/'collected.json').exists():continue
   probe=subprocess.run(['ssh','ubuntu@dev3','test','-f',str(src/'pipeline_manifest.json')])
   if probe.returncode:continue
   dst.mkdir(parents=True,exist_ok=True)
   command=['/home/ubuntu/anaconda3/envs/nerfstudio/bin/python',str(remote/'summarize_patchmatch_source_count_depth.py'),'--arm',str(src),'--regions',str(remote/'regions'/f'{frame}_train.json')]
   with (ROOT/'logs'/f'{frame}_{count}_stats.log').open('w') as log:subprocess.run(['ssh','ubuntu@dev3',shlex.join(command)],check=True,stdout=log,stderr=subprocess.STDOUT)
   subprocess.run(['rsync','-rl','--no-owner','--no-group','--no-perms','--exclude=dense','--exclude=depth_dataset','--exclude=texture_subset','--exclude=fixed_model',f'ubuntu@dev3:{src}/',str(dst)+'/'],check=True)
   # Only source12 full real geometric depth is needed by the parallel prior study.
   if count==12:
    (dst/'dense/stereo/depth_maps').mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-rl','--include=*/','--include=*.geometric.bin','--exclude=*',f'ubuntu@dev3:{src}/dense/stereo/depth_maps/',str(dst/'dense/stereo/depth_maps')+'/'],check=True)
    subprocess.run(['rsync','-rl',f'ubuntu@dev3:{src}/dense/sparse/',str(dst/'dense/sparse')+'/'],check=True)
    for row in read(dst/'depth_comparison.json')['rows']:
     assert sha(dst/'dense/stereo/depth_maps'/(row['file_path']+'.geometric.bin'))==row['sha256']
   assert sha(dst/'colmap_patchmatch_tsdf.ply')==read(dst/'colmap_patchmatch_tsdf.json')['output_sha256']
   mesh_review(frame,dst/'colmap_patchmatch_tsdf.ply',dst/'colmap_patchmatch_tsdf.json',dst/'matched_review')
   data=ROOT/'data'/frame/'staged63';payload=read(data/'transforms.json');gt=data/payload['val_filenames'][0]
   command=[sys.executable,str(ROOT/'frozen_code/score_colmap_patchmatch_tsdf_face.py'),'--frame-id',frame,'--prediction',str(dst/'render/nearest_fill16/eval_pred_0000.exr'),'--ground-truth',str(gt),'--face-polygons',str(ROOT/'regions'/f'{frame}_face.json'),'--output-dir',str(dst/'face_metrics'),'--device','cpu']
   if not (dst/'face_metrics').exists():
    with (ROOT/'logs'/f'{frame}_{count}_metrics.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENCV_IO_ENABLE_OPENEXR='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'))
   if not (dst/'face_metrics/metrics.json').exists():raise RuntimeError(f'Incomplete metric output: {dst}')
   subprocess.run(['rsync','-rl',f'ubuntu@dev3:{remote}/timings/{frame}_{count}.json',str(dst/'timing.json')],check=True)
   write(dst/'collected.json',dict(frame=frame,count=count,mesh_sha256=sha(dst/'colmap_patchmatch_tsdf.ply'),status='collected_pending_visual_review'))
   completed.append([frame,count]);print('collected',frame,count,flush=True)
 subprocess.run(['rsync','-rl',f'ubuntu@dev3:{remote}/checks.jsonl',f'ubuntu@dev3:{remote}/status.json',str(ROOT)+'/'],check=True)
 print(json.dumps(completed))

def watch_collect():
 while True:
  collect()
  status=read(ROOT/'status.json')
  print(json.dumps({k:status.get(k) for k in ['timestamp','status','frame','source_count','photometric_maps','geometric_maps','worker_alive']}),flush=True)
  if len(list((ROOT/'arms').glob('*/*/collected.json')))==6:break
  if status.get('worker_alive') is False and status.get('status')!='complete':
   # A worker exit is sampled briefly before the next arm is launched.
   time.sleep(5)
  time.sleep(60)

def summarize(allow_partial=False):
 subprocess.run(['rsync','-rl','ubuntu@dev3:/fsx/oregon/dec5_patchmatch_source_count_ablation/gpu_samples.jsonl',str(ROOT)+'/'],check=True)
 gpu_samples=[json.loads(line) for line in (ROOT/'gpu_samples.jsonl').read_text().splitlines()]
 rows=[]
 for frame in FACE:
  data=ROOT/'data'/frame/'staged63';payload=read(data/'transforms.json');gt=Image.open(data/payload['val_filenames'][0]).transpose(Image.Transpose.ROTATE_90)
  eval_images=[('GT',gt)];train_images=[];virtual_images=[];overlays=[]
  region=read(ROOT/'regions'/f'{frame}_train.json')
  train_images.append(('Train GT',Image.open(data/region['file_path']).transpose(Image.Transpose.ROTATE_90)))
  old=ROOT/'old_baseline_review'/frame
  virtual_images.append(('Historical 12',Image.open(old/'virtual_clay.png')))
  overlays.append(('Historical 12',Image.open(old/'train_missing_overlay.png')))
  commands_by_count={}
  for count in [12,24,36]:
   arm=ROOT/'arms'/frame/f'sources{count}'
   if not (arm/'collected.json').exists():
    if allow_partial:continue
    raise ValueError(f'Arm incomplete: {arm}')
   manifest=read(arm/'pipeline_manifest.json');depth=read(arm/'depth_comparison.json');mesh=read(arm/'colmap_patchmatch_tsdf.json');metrics=read(arm/'face_metrics/metrics.json');review=read(arm/'matched_review/mesh_review.json')
   request=read(arm/'pipeline_request.json');remoteout=str(Path(request['commands'][0]['command'][request['commands'][0]['command'].index('--output')+1]).parent)
   normalized=[]
   for stage in request['commands']:
    command=[token.replace(remoteout,'<ARM>') for token in stage['command']]
    if '--source-count' in command:command[command.index('--source-count')+1]='<COUNT>'
    normalized.append(dict(stage=stage['stage'],command=command))
   commands_by_count[count]=normalized
   row=dict(frame=frame,source_count=count,face_psnr=metrics['face_psnr'],face_ssim=metrics['face_ssim'],face_lpips=metrics['face_lpips'],triangles=mesh['triangles'],components=mesh['connected_components'],geometric_coverage_qc=depth['geometric_coverage_mean'],local_depth=depth['local']['local_regions'],local_mesh=review['stats']['train'],stage_seconds={s['name']:s['seconds'] for s in manifest['stages']})
   row['sampled_peak_gpu_mib']=max(int(s['gpu'].split(',')[0]) for s in gpu_samples if s['frame']==frame and s['source_count']==count)
   baseline=np.load(ROOT/'arms'/frame/'sources12/matched_review/train_depth.npz')['depth']
   candidate=np.load(arm/'matched_review/train_depth.npz')['depth'];row['local_depth_change_vs_12']={}
   baseline_geom=np.load(ROOT/'arms'/frame/'sources12/review_train_maps.npz')['geometric']
   candidate_geom=np.load(arm/'review_train_maps.npz')['geometric']
   for name,mask_meta in region['masks'].items():
    mask=np.array(Image.open(ROOT/'regions'/mask_meta['file']),bool)
    oldhit=baseline>0;newhit=candidate>0;both=mask&oldhit&newhit
    values=np.abs(candidate[both]-baseline[both])/baseline[both]
    row['local_depth_change_vs_12'][name]=dict(added_hits=int((mask&~oldhit&newhit).sum()),removed_hits=int((mask&oldhit&~newhit).sum()),relative_depth_p50_p95=np.percentile(values,[50,95]).tolist() if values.size else None)
    oldvalid=np.isfinite(baseline_geom)&(baseline_geom>0);newvalid=np.isfinite(candidate_geom)&(candidate_geom>0)
    row['local_depth_change_vs_12'][name].update(added_geometric_valid=int((mask&~oldvalid&newvalid).sum()),removed_geometric_valid=int((mask&oldvalid&~newvalid).sum()))
   rows.append(row)
   eval_images.append((f'{count} sources',Image.open(arm/'render/nearest_fill16/eval_pred_0000.png').transpose(Image.Transpose.ROTATE_90)))
   train_images.append((f'{count} sources',Image.open(arm/'matched_review/train_clay.png')))
   virtual_images.append((f'{count} sources',Image.open(arm/'matched_review/virtual_clay.png')))
   overlays.append((f'{count} sources',Image.open(arm/'matched_review/train_missing_overlay.png')))
  assert all(value==commands_by_count[12] for value in commands_by_count.values()),f'Confounded commands: {frame}'
  out=ROOT/('partial_comparisons' if allow_partial else 'comparisons')/frame;out.mkdir(parents=True,exist_ok=True)
  for name,images,box in [('heldout_head',eval_images,(270,480,960,1280)),('matched_train_clay',train_images,(250,380,1050,1180)),('virtual_clay',virtual_images,(250,380,1050,1180)),('train_ray_miss_overlay',overlays,(250,380,1050,1180)),('train_crown',train_images,(350,480,820,760)),('virtual_crown',virtual_images,(440,520,820,750))]:
   w,h=box[2]-box[0],box[3]-box[1];panel=Image.new('RGB',(len(images)*w,h+24));draw=ImageDraw.Draw(panel)
   for i,(label,im) in enumerate(images):panel.paste(im.crop(box),(i*w,24));draw.text((i*w+5,5),label,fill='white')
   panel.save(out/f'{name}_native.png');panel.resize((round(panel.width*.5),round(panel.height*.5))).save(out/f'{name}_preview.png')
 write(ROOT/('partial_summary.json' if allow_partial else 'summary.json'),dict(rows=rows,complete=len(rows)==6,command_equivalence='pass_except_source_count_and_output_paths',face_protocol='manual GT-only polygon, exact selected-pixel PSNR, tight-bbox zero-outside SSIM/LPIPS-Alex',visual_verdict='pending'))
 print(json.dumps(rows,indent=2))

def partial_summarize():summarize(allow_partial=True)

def record_visual_review():
 """Seal a separately authored manual review, only after images were inspected."""
 review=read(ROOT/'visual_review.json');summary=read(ROOT/'summary.json')
 assert summary['complete'] and read(ROOT/'validation.json')['checked_arms']==6
 assert review['verdict'] and len(review['reviewed_relative_files'])>=2
 review['reviewed_file_sha256']={name:sha(ROOT/name) for name in review['reviewed_relative_files']}
 review['analysis_code_sha256']={name:sha(Path(__file__).parent/name) for name in ['run_patchmatch_source_count_ablation.py','review_patchmatch_source_count_ablation.py','summarize_patchmatch_source_count_depth.py']}
 write(ROOT/'visual_review.json',review)
 summary['visual_verdict']=review['verdict'];summary['visual_review_sha256']=sha(ROOT/'visual_review.json')
 write(ROOT/'summary.json',summary)
 for row in summary['rows']:
  path=ROOT/'arms'/row['frame']/f"sources{row['source_count']}"/'collected.json'
  receipt=read(path);receipt.update(status='complete',visual_review_sha256=summary['visual_review_sha256']);write(path,receipt)
 print(json.dumps(dict(status='complete',arms=6,verdict=review['verdict'])))

def validate_results():
 """Independent arithmetic checks on retained pixels, not summary helpers."""
 os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
 import cv2
 checks=[]
 for frame in FACE:
  data=ROOT/'data'/frame/'staged63';payload=read(data/'transforms.json')
  gt=np.asarray(Image.open(data/payload['val_filenames'][0]).convert('RGB'),np.float32)/255
  region=read(ROOT/'regions'/f'{frame}_train.json')
  masks={name:np.array(Image.open(ROOT/'regions'/value['file']),bool) for name,value in region['masks'].items()}
  for count in [12,24,36]:
   arm=ROOT/'arms'/frame/f'sources{count}'
   if not (arm/'collected.json').exists():continue
   summary=read(arm/'depth_comparison.json');review=read(arm/'matched_review/mesh_review.json')
   assert summary['reference_count']==62 and summary['sources_per_reference']==count
   assert summary['regions_sha256']==sha(ROOT/'regions'/f'{frame}_train.json')
   assert summary['local']['consistency_graph_valid_depth_agreement']==1
   mask=np.array(Image.open(arm/'face_metrics/face_mask.png'),bool)
   render=arm/'render/nearest_fill16'
   pred=cv2.imread(str(render/'eval_pred_0000.exr'),cv2.IMREAD_UNCHANGED)[...,[2,1,0]].clip(0,1)
   retained_gt=cv2.imread(str(render.parent/'eval_gt_0000.exr'),cv2.IMREAD_UNCHANGED)[...,[2,1,0]]
   assert np.array_equal(retained_gt,gt)
   error=pred[mask].astype(np.float64)-gt[mask].astype(np.float64)
   psnr=float(-10*np.log10(np.square(error).mean()))
   recorded=read(arm/'face_metrics/metrics.json')['face_psnr']
   assert abs(psnr-recorded)<1e-5
   maps=np.load(arm/'review_train_maps.npz');rays=np.load(arm/'matched_review/train_depth.npz')['depth']
   valid=np.isfinite(maps['geometric'])&(maps['geometric']>0)
   photo=np.isfinite(maps['photometric'])&(maps['photometric']>0)
   for name,m in masks.items():
    observed=summary['local']['local_regions'][name]
    assert np.count_nonzero(valid[m])==observed['geometric_valid']
    assert np.count_nonzero(photo[m]&~valid[m])==observed['photo_valid_geo_invalid']
    assert np.count_nonzero(rays[m]>0)==review['stats']['train'][name]['mesh_hit_pixels']
    histogram=np.asarray(observed['consistency_graph_observation_histogram'])
    assert histogram.sum()==observed['geometric_valid'] and not histogram[count+1:].any()
    mean=float(np.dot(np.arange(len(histogram)),histogram)/histogram.sum())
    assert abs(mean-observed['consistency_graph_mean_observations_valid_depth'])<1e-10
   checks.append(dict(frame=frame,source_count=count,gt_pixel_identical=True,independent_face_psnr=psnr,face_psnr_error=psnr-recorded,local_map_and_mesh_counts='pass',graph_depth_bitmap_agreement=1))
 write(ROOT/'validation.json',dict(checked_arms=len(checks),checks=checks,scope='Independent NumPy float64 face PSNR and exact pixel-count recomputation. SSIM/LPIPS reuse the frozen established scorer; not independently reimplemented. Visual review is separate.'))
 print(json.dumps(checks))

def verify_design():
 sys.path.insert(0,str(ROOT/'frozen_code'))
 from build_colmap_patch_match_config import explicit_source_views
 request=read(ROOT/'experiment_request.json')
 historical=read('/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150/campaign_request.json')
 for name,digest in request['code'].items():assert sha(ROOT/'frozen_code'/name)==digest
 checks=[]
 for frame in FACE:
  root=ROOT/'data'/frame/'staged63';d=read(root/'transforms.json');train=set(d['train_filenames']);frames=[f for f in d['frames'] if f['file_path'] in train]
  for name,digest in request['inputs'][frame].items():assert sha(root/name)==digest
  assert len(frames)==62 and not train.intersection(d['val_filenames'])
  old=next(r for r in historical['source_frames'] if Path(r['source_dataset']).name==frame)
  oldhash={r['physical_camera']:r['sha256'] for r in old['source_images']}
  staging=read(root/'staging_manifest.json')
  for row in staging['conversion_rows']:assert row['source_sha256']==oldhash[row['physical_camera']]
  maps={n:explicit_source_views(frames,source_count=n) for n in [12,24,36]}
  for ref in train:
   assert maps[12][ref]==maps[24][ref][:12]==maps[36][ref][:12]
   assert maps[24][ref]==maps[36][ref][:24]
   for n in maps:assert len(maps[n][ref])==n and ref not in maps[n][ref] and set(maps[n][ref])<=train
  checks.append(dict(frame=frame,references=62,source_counts=[12,24,36],nested=True,heldout_excluded=True,hashes_verified=True,source_exrs_match_original_campaign=True))
 write(ROOT/'design_audit.json',checks)
 print(json.dumps(checks))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['regions','verify_design','validate_results','record_visual_review','baseline_review','collect','watch_collect','summarize','partial_summarize']);a=p.parse_args();globals()[a.action]()
