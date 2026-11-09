"""Read-only native/dense field termination against52TRAINphotometricdepthcandidates."""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
R=Path('/fsx/tmp/marshawn_001660_hd')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(a):
 for p in [R/'code_sfm/LookCloser/scripts',R/'code_native/LookCloser/scripts',R/'final_pair/code',Path(__file__).parent,a.code_root,a.stage/'code']:sys.path.insert(0,str(p))
 import torch
 from nerfstudio.utils.eval_utils import eval_setup
 from marshawn_release_common import trace_native,ray_box_interval,point_cells
 from audit_marshawn_body_rays import interval_mass
 from probe_marshawn_occupancy import integrate,digest_model
 from audit_marshawn_geometry import project_points
 torch.set_num_threads(2)
 def configure(c):c.eval_checkpoint=a.checkpoint;c.load_checkpoint=None;c.load_dir=None;c.load_step=None;c.load_config=None;return c
 cfg,pipe,actual,step=eval_setup(a.config,test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=configure);pipe.eval();model=pipe.model;model.current_train_step=step
 if actual.resolve()!=a.checkpoint.resolve():raise ValueError('Wrongcheckpoint')
 ds=pipe.datamanager.train_dataset;names=[Path(p).stem.split('_')[-1]for p in ds.image_filenames];data=Path(cfg.pipeline.datamanager.dataparser.data);tf=json.loads((data/'transforms.json').read_text());by={q['physical_camera']:q for q in tf['frames']if q['file_path']in tf['train_filenames']};normalization=pipe.datamanager.train_dataparser_outputs;assert normalization.dataparser_scale==1;np.testing.assert_allclose(normalization.dataparser_transform.cpu().numpy(),np.eye(4)[:3],atol=1e-7)
 mvs=R/'floor_plane_sweep_HD_PILOT/HD_52_reviewed_candidates_NOT_PRIOR.json';points=json.loads(mvs.read_text())['points'];xyz=np.array([q['candidate_XYZ']for q in points]);n=np.array(json.loads((R/'floor_band_prior_PROPOSAL/proposal.json').read_text())['fit_normal']);before=digest_model(model);grid=model.occupancy_grid.binaries.detach().cpu().numpy().copy();boxes=model.occupancy_grid.aabbs.detach().cpu().numpy();box=model.field.aabb.detach().cpu().numpy().reshape(2,3);np.testing.assert_allclose(boxes[0].reshape(2,3),box,atol=1e-6);ij,inside,on=point_cells(xyz,grid,boxes)
 with torch.no_grad():sigma=model.field.density_fn(torch.tensor(xyz,device=model.device,dtype=torch.float32)).float().cpu().numpy().reshape(-1)
 a.output.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.output/'point_core_occupancy.npz',candidate_ids=np.array([q['candidate_id']for q in points]),xyz=xyz,cells=ij,inside=inside,occupied=on,sigma=sigma);rows=[];saved={}
 def analyze(st,en,sig,truth,cosine,o,v):
  alpha,T,w=integrate(st,en,sig);mid=(st+en)/2;opacity=float(w.sum());mean=float((mid*w).sum()/opacity)if opacity>1e-12 else None;quant={}
  for q in [.1,.5,.9]:
   if opacity>1e-12:
    i=min(int(np.searchsorted(np.cumsum(w),q*opacity)),len(mid)-1);quant[str(q)]=dict(t=float(mid[i]),normal_height_error=float((mid[i]-truth)*(v@n)))
  bands={}
  for delta in [.003,.005,.01]:
   dt=delta/cosine;lo,hi=truth-dt,truth+dt;pre=interval_mass(st,en,sig,-1e12,lo);mass=interval_mass(st,en,sig,lo,hi);bands[str(delta)]=dict(pre_surface_mass=pre,surface_band_mass=mass,post_surface_mass=float(max(0,opacity-pre-mass)),ray_halfwidth=dt)
  return dict(opacity=opacity,conditional_mean_depth=mean,quantiles=quant,bands=bands,samples=len(st))
 def dense(o,v,lo,hi,N):
  with torch.no_grad():
   edges=torch.linspace(lo,hi,N+1,device=model.device);mid=(edges[:-1]+edges[1:])/2;world=torch.tensor(o,device=model.device,dtype=torch.float32)[None]+mid[:,None]*torch.tensor(v,device=model.device,dtype=torch.float32)[None];sig=model.field.density_fn(world).float().cpu().numpy().reshape(-1);return edges[:-1].cpu().numpy(),edges[1:].cpu().numpy(),sig
 for j,point in enumerate(points):
  for name in ['MA','NA','NB','LB']:
   ci=names.index(name);camera=ds.cameras[ci:ci+1].to(model.device);uv=project_points(by[name],xyz[j:j+1])[0][0];ray=camera.generate_rays(0,coords=torch.tensor(uv[::-1].copy()[None],device=model.device,dtype=torch.float32),keep_shape=False);origin=ray.origins[0].cpu().numpy();direction=ray.directions[0].cpu().numpy();truth=float((xyz[j]-origin)@direction);perp=np.linalg.norm(xyz[j]-origin-truth*direction);assert perp<2e-6;cosine=abs(float(n@direction));assert cosine>.05;near=float(model.config.near_plane);far=float(model.config.far_plane);lo,hi=ray_box_interval(origin,direction,box,near,far);assert lo<truth<hi;native=trace_native(model,ray);profiles={'native':analyze(native['starts'],native['ends'],native['sigma'],truth,cosine,origin,direction)}
   # Independentall-cellquadrature; originalgridnevermutated.
   for N in [8192,16384]:
    st,en,sig=dense(origin,direction,lo,hi,N);profiles['uniform'+str(N)]=analyze(st,en,sig,truth,cosine,origin,direction)
    if N==16384 and(j%8==0 or not on[j]):
     prefix=f'p{point["candidate_id"]}_{name}_u{N}';saved[prefix+'_starts']=st;saved[prefix+'_ends']=en;saved[prefix+'_sigma']=sig
   u,w=profiles['uniform8192'],profiles['uniform16384'];differences=[abs(u['opacity']-w['opacity'])]+[abs(u['bands'][str(d)][k]-w['bands'][str(d)][k])for d in [.003,.005,.01]for k in ['pre_surface_mass','surface_band_mass']];qdiff=abs(u['quantiles']['0.5']['normal_height_error']-w['quantiles']['0.5']['normal_height_error'])if u['quantiles']and w['quantiles']else None;row=dict(candidate_id=point['candidate_id'],camera=name,uv=uv.tolist(),origin=origin.tolist(),direction=direction.tolist(),MVS_XYZ=xyz[j].tolist(),MVS_ray_distance=truth,normal_cosine=cosine,source_near=near,source_far=far,source_core_cell_occupied=bool(on[j]),source_sigma_at_MVS=float(sigma[j]),profiles=profiles,uniform8192_16384_mass_max_abs_difference=max(differences),uniform8192_16384_q50_normal_height_difference=qdiff);rows.append(row)
   if len(rows)%16==0:(a.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(points)*4)))
 # Atmost16worstconvergence checks at32768, plus explicitperraymetrics.
 priority=sorted(range(len(rows)),key=lambda i:rows[i]['uniform8192_16384_mass_max_abs_difference'],reverse=True)[:16]
 for i in priority:
  q=rows[i];o=np.array(q['origin']);v=np.array(q['direction']);lo,hi=ray_box_interval(o,v,box,q['source_near'],q['source_far']);st,en,sig=dense(o,v,lo,hi,32768);q['profiles']['uniform32768']=analyze(st,en,sig,q['MVS_ray_distance'],q['normal_cosine'],o,v);u,w=q['profiles']['uniform16384'],q['profiles']['uniform32768'];q['uniform16384_32768_mass_max_abs_difference']=max([abs(u['opacity']-w['opacity'])]+[abs(u['bands'][str(d)][k]-w['bands'][str(d)][k])for d in [.003,.005,.01]for k in ['pre_surface_mass','surface_band_mass']])
 assert digest_model(model)==before;assert np.array_equal(model.occupancy_grid.binaries.cpu().numpy(),grid);summary={}
 for profile in ['native','uniform8192','uniform16384']:
  values=[q['profiles'][profile]for q in rows];z=dict(opacity_percentiles=np.percentile([v['opacity']for v in values],[0,10,50,90,100]).tolist(),q50_normal_height_percentiles=np.percentile([v['quantiles']['0.5']['normal_height_error']for v in values if v['quantiles']],[0,10,50,90,100]).tolist(),bands={})
  for delta in [.003,.005,.01]:
   z['bands'][str(delta)]={k:dict(median=float(np.median([v['bands'][str(delta)][k]for v in values])),p10=float(np.percentile([v['bands'][str(delta)][k]for v in values],10)),p90=float(np.percentile([v['bands'][str(delta)][k]for v in values],90)))for k in ['pre_surface_mass','surface_band_mass','post_surface_mass']};z['bands'][str(delta)]['band_mass_ge05_count']=sum(v['bands'][str(delta)]['surface_band_mass']>=.5 for v in values);z['bands'][str(delta)]['pre_mass_gt01_count']=sum(v['bands'][str(delta)]['pre_surface_mass']>.1 for v in values)
  summary[profile]=z
 np.savez_compressed(a.output/'representative_density_profiles.npz',**saved);report=dict(model_label=a.label,checkpoint=str(actual),checkpoint_sha256=sha(actual),config_sha256=sha(a.config),script_sha256=sha(__file__),MVS_candidates_sha256=sha(mvs),model_kind=type(model).__name__,near_plane=float(model.config.near_plane),core_policy='Actualcheckpointbinaries; nohalos/noCPUthresholdreconstruction',point_count=52,rays=208,core_knownMVS_misses=int((inside&~on).sum()),outside_core_AABB=int((~inside).sum()),summary=summary,rows=rows,grid_and_field_state_unchanged=True,no_training_or_constraints_or_carve=True,uses_eval_RGB=False,diagnostic_normal_band_halfwidths=[.003,.005,.01],bands_are_NOT_calibrated_MVS_CI=True,uniform_samples=[8192,16384],worst16_additional32768=True);(a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(label=a.label,misses=report['core_knownMVS_misses'],native=summary['native'],dense=summary['uniform16384'])),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();
 for key in ['config','checkpoint','stage','code-root','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--label',required=True);main(p.parse_args())
