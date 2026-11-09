#!/usr/bin/env python3
"""Train-observed SfM frequency seeds; no occupancy or learned field changes.

Each observed2Dresolution is pulled back by the normalized-image Jacobian,
then median per point and maximum per exact live frequency-grid cell. Only
actual track-camera witnesses are used, never arbitrary all-camera visibility.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def indices(points,box,res=128):
 p=torch.as_tensor(points,dtype=torch.float32);b=torch.as_tensor(box,dtype=torch.float32)
 return torch.clamp((p-b[0])/(b[1]-b[0])*(res-1),0,res-1.001).long().numpy()
def summarize(levels):
 levels=np.asarray(levels);return dict(count=len(levels),nonzero_fraction=float((levels>0).mean()) if len(levels) else None,maxlevel_fraction=float((levels==15).mean()) if len(levels) else None,histogram=np.bincount(levels.astype(int),minlength=16).tolist(),median=float(np.median(levels)) if len(levels) else None)
def run(a):
 sys.path.insert(0,str(a.code_root))
 from nerfstudio.utils.lookcloser_projection import normalized_frequency_resolution
 from nerfstudio.model_components.lookcloser_grid import FrequencyGridManager
 from nerfstudio.data.scene_box import SceneBox
 torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=False)
 manifest=json.loads((a.campaign/'campaign.json').read_text());data=Path(manifest['data']);geometry=a.witnesses.parent
 meta=json.loads((data/'transforms.json').read_text());train=set(meta['train_filenames']);rows=[r for r in meta['frames'] if r['file_path'] in train];by={r['physical_camera']:r for r in rows}
 qa=json.loads(a.frequency_review.read_text());assert qa['complete_gate'] and qa['valid_maps']==63
 assert sha(data/'transforms.json')==manifest['transforms_sha256']==qa['transforms_sha256']
 assert sha(manifest['boxes'])==manifest['boxes_sha256']
 with np.load(a.witnesses) as z:xyz=z['xyz'].astype(np.float64);camera_names=z['cameras'].tolist();pairs=z['source_pairs'].copy()
 geo_audit=json.loads((geometry/'audit.json').read_text());assert geo_audit['transforms_sha256']==sha(data/'transforms.json')
 tracks_file=geometry/'strict_tracks.json'
 if tracks_file.exists():
  tracks=json.loads(tracks_file.read_text());assert tracks['points']==len(xyz);observations=tracks['camera_ids'];observation_source='all actual track-camera IDs from current-frame >=3view SfM; UV reprojected from triangulated XYZ, track residual<1HDpx'
 else:observations=pairs.tolist();observation_source='two actual source-pair cameras per point; UV reprojected from triangulated XYZ'
 if len(observations)!=len(xyz):raise ValueError('Observation/point count mismatch')
 for p,views in enumerate(observations):
  if len(set(views))!=len(views) or len(views)<2 or not set(pairs[p]).issubset(views):raise ValueError('Invalid observed-view graph')
 obs_point=np.array([i for i,vs in enumerate(observations) for _ in vs]);obs_camera=np.array([c for vs in observations for c in vs]);N=len(obs_point)
 obs_pose=np.empty((N,3,4));obs_focal=np.empty((N,2));obs_wh=np.empty((N,2));f2d=np.zeros(N);uv=np.zeros((N,2));valid=np.zeros(N,bool);map_hashes={}
 for camera,name in enumerate(camera_names):
  row=by[name];assert row['file_path'] in train
  ids=np.flatnonzero(obs_camera==camera)
  pose=np.array(row['transform_matrix']);local=(xyz[obs_point[ids]]-pose[:3,3])@pose[:3,:3];z=-local[:,2];u=row['fl_x']*local[:,0]/z+row['cx'];v=-row['fl_y']*local[:,1]/z+row['cy'];uv[ids]=np.column_stack([u,v]);valid[ids]=(z>0)&(u>=0)&(u<row['w'])&(v>=0)&(v<row['h'])
  prefix=data/'lookcloser_frequencies'/Path(row['file_path']).stem;pt=prefix.with_suffix('.pt');jp=prefix.with_suffix('.json');md=json.loads(jp.read_text());q=next(q for q in qa['records'] if q['camera']==name)
  assert sha(pt)==q['map_sha256'] and sha(jp)==q['metadata_sha256'] and sha(data/row['file_path'])==q['source_sha256'];assert md['ssim_threshold']==.95 and md['train_steps_per_level']==1000
  fmap=torch.load(pt,map_location='cpu',weights_only=True).numpy();stride=md['stride'];ix=np.clip(np.floor(u/stride).astype(int),0,fmap.shape[1]-1);iy=np.clip(np.floor(v/stride).astype(int),0,fmap.shape[0]-1);f2d[ids]=fmap[iy,ix]
  obs_pose[ids]=pose[:3];obs_focal[ids]=[row['fl_x'],row['fl_y']];obs_wh[ids]=[row['w'],row['h']]
  map_hashes[name]=dict(map_sha256=sha(pt),metadata_sha256=sha(jp),source_sha256=q['source_sha256'])
 if not valid.all():raise ValueError(f'{int((~valid).sum())} actual track observations outside calibrated raster; inspect before seeding')
 counts=np.array([len(v) for v in observations]);cuts=np.cumsum(counts)[:-1]
 median2d=np.array([np.median(v) for v in np.split(f2d,cuts)])
 np.savez_compressed(a.output/'observations.npz',point_indices=obs_point,camera_indices=obs_camera,uv=uv,f2d=f2d,camera_names=np.array(camera_names))
 specs=[json.loads(Path(s).read_text()) for s in manifest['variants']];unique={};variants={};report=[]
 subject_xyz=np.load(a.subject_witnesses)['xyz'].astype(float)
 from scipy.spatial import cKDTree
 dist,subject_point_ids=cKDTree(xyz).query(subject_xyz)
 if dist.max()>1e-6:raise ValueError('Subject witnesses not subset of current train SfM')
 for spec in specs:
  box=np.asarray(spec['aabb'],np.float64);key=hashlib.sha256(box.tobytes()).hexdigest();name=spec['name'].rsplit('_halo',1)[0]
  if key not in unique:
   resolution,ok=normalized_frequency_resolution(torch.from_numpy(xyz[obs_point]),torch.from_numpy(obs_pose),torch.from_numpy(obs_focal),torch.from_numpy(obs_wh),torch.from_numpy(box),torch.from_numpy(f2d))
   if not ok.all():raise ValueError('Invalid normalized Jacobian observation')
   median=np.array([np.median(v) for v in np.split(resolution.numpy(),cuts)],dtype=np.float32)
   schedule=16*(8192/16)**(np.arange(16)/15)
   point_levels=np.argmin(np.abs(median[:,None]-schedule[None,:]),axis=1).astype(np.float32)
   inside=((xyz>=box[0])&(xyz<=box[1])).all(1);ids=np.flatnonzero(inside);cell=indices(xyz[inside],box);flat=np.ravel_multi_index(cell.T,(128,128,128));dense=np.zeros(128**3,np.float32);np.maximum.at(dense,flat,point_levels[inside]);unique_flat=np.unique(flat);seed_ix=np.column_stack(np.unravel_index(unique_flat,(128,128,128))).astype(np.int64);seed_levels=dense[unique_flat]
   file=name+'.npz';np.savez_compressed(a.output/file,indices_xyz=seed_ix,levels=seed_levels,aabb=box,grid_resolution=np.array(128),num_levels=np.array(16),min_res=np.array(16.),max_res=np.array(8192.),point_indices=ids,point_levels=point_levels[inside],point_median_3d_resolution=median[inside],point_median_2d_resolution=median2d[inside])
   subject_inside=inside[subject_point_ids];subject_ids=subject_point_ids[subject_inside];subject_cell=indices(xyz[subject_ids],box);subject_seed=dense[np.ravel_multi_index(subject_cell.T,(128,128,128))]
   loaded=dense[flat]
   if (loaded<point_levels[inside]).any():raise AssertionError('Max per cell failed witness coverage')
   manager=FrequencyGridManager(SceneBox(aabb=torch.tensor(box,dtype=torch.float32)),resolution=128,min_res=16,max_res=8192)
   manager.grid[seed_ix[:,0],seed_ix[:,1],seed_ix[:,2]]=torch.tensor(seed_levels)
   live=manager.query(torch.tensor(xyz[inside],dtype=torch.float32)).squeeze(-1).numpy()
   if (live<point_levels[inside]).any():raise AssertionError('Actual FrequencyGridManager FP32 query failed point coverage')
   row=dict(scope_size=name,file=file,sha256=sha(a.output/file),aabb=box.tolist(),grid_resolution=128,unique_seed_cells=len(unique_flat),nonzero_seed_cells=int((seed_levels>0).sum()),full_grid_nonzero_fraction=float((dense>0).mean()),outside_train_witnesses=int((~inside).sum()),all_inside_witnesses_before=summarize(np.zeros(len(ids))),all_inside_witnesses_after=summarize(loaded),all_inside_witnesses_required=summarize(point_levels[inside]),subject_witnesses_outside=int((~subject_inside).sum()),subject_witnesses_before=summarize(np.zeros(len(subject_ids))),subject_witnesses_after=summarize(subject_seed),all_inside_witnesses_seed_at_least_target=True,actual_live_fp32_query_coverage_passed=True,frequency_halo_radius=0)
   unique[key]=row;report.append(row)
  variants[spec['name']]=dict(file=unique[key]['file'],sha256=unique[key]['sha256'],aabb=box.tolist())
 final=dict(schema_version=1,kind='frequency_only_train_observed_sfm_seed',source_campaign=str(a.campaign),source_campaign_sha256=sha(a.campaign/'campaign.json'),transforms_sha256=sha(data/'transforms.json'),normalization_sha256=sha(data/'preparation.json'),calibration_sha256=json.loads((data/'preparation.json').read_text())['rig_sha256'],observations_file='observations.npz',observations_sha256=sha(a.output/'observations.npz'),paper_sha256=sha(Path(__file__).resolve().parents[1]/'Paper LookCloser.md'),boxes_sha256=sha(manifest['boxes']),points_sha256=sha(a.witnesses),subject_points_sha256=sha(a.subject_witnesses),tracks_sha256=sha(tracks_file) if tracks_file.exists() else None,frequency_review_sha256=sha(a.frequency_review),projection_source_sha256=sha(a.code_root/'nerfstudio/utils/lookcloser_projection.py'),script_sha256=sha(__file__),observations=N,points=len(xyz),train_cameras=len(camera_names),uses_eval_rgb=False,observation_source=observation_source,aggregation='Per observed camera normalized_jacobian(2Dscalarresolution), median 3Dresolution per point, argmin absolute resolution distance to discrete schedule (Paper7.3), MAX per exact frequency cell; median2D also retained for audit',before_reference='Fresh cold initialization: all frequency levels zero; not a modification of the existing4k checkpoint',grid_mapping='EXACT FP32 FrequencyGridManager: floor(clamp(normalized_xyz*(resolution-1),0,resolution-1.001)); outside witnesses excluded',lookup_clamp_note='Highest plane index127 is unused by current live frequency mapping; preserve mapping parity rather than change frozen geometry',level_quantization='Paper7.3 argmin absolute linear resolution difference; ties select lower level; intentionally differs from legacy round(log) runtime conversion',frequency_halo_radius=0,occupancy_admission=False,variants=variants,seeds=report,map_bindings=map_hashes,limitations=['Sparse exact point cells only; adjacent cells remain zero unless separately tested frequency halo is explicitly enabled later','Observed-camera UV derived from current calibrated3Dpoint; stored track-camera provenance has<1pixel fitted reprojection error','This cold-start seed is not measured training quality and never changes occupancy'])
 write(a.output/'manifest.json',final);print(json.dumps(dict(output=str(a.output),seeds=len(report),variants=len(variants),observations=N,coverage=[dict(name=r['scope_size'],cells=r['unique_seed_cells'],inside=r['all_inside_witnesses_after']['count'],nonzero=r['all_inside_witnesses_after']['nonzero_fraction'],subject_nonzero=r['subject_witnesses_after']['nonzero_fraction']) for r in report])),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['campaign','code-root','witnesses','subject-witnesses','frequency-review','output']:p.add_argument('--'+k,type=Path,required=True)
 run(p.parse_args())
