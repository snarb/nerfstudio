"""Read-only actual canonical HDR whole-grid numerical audit; no optimizer/callback."""
from pathlib import Path
import argparse,hashlib,json,sys,time,itertools
import numpy as np

R=Path('/fsx/tmp/marshawn_001660_hd')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):
 a=np.asarray(a);v=a[np.isfinite(a)];return dict(count=int(a.size),finite=int(v.size),nan=int(np.isnan(a).sum()),posinf=int(np.isposinf(a).sum()),neginf=int(np.isneginf(a).sum()),minimum=float(v.min())if v.size else None,maximum=float(v.max())if v.size else None,quantiles=np.quantile(v,[0,.5,.9,.99,.999,1]).tolist()if v.size else None)
def main(a):
 s=R/'canonical_linear_HDR_NEAR052_RESUME4k_STAGED';old=R/'canonical_linear_HDR_PQMSE_STAGED'
 for p in reversed([R/'code_sfm',R/'code_sfm/LookCloser/scripts',R/'final_pair/code',old/'code',s/'code',R/'final_pair/candidate_inference_code']):sys.path.insert(0,str(p))
 import torch
 from scipy.ndimage import maximum_filter
 from nerfstudio.utils.eval_utils import eval_setup
 from probe_marshawn_occupancy import digest_model
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(12*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
 a.output.mkdir(exist_ok=False);policy=json.loads((s/'grids/step000006000_receipt.json').read_text());ck=Path(policy['checkpoint']);assert sha(ck)==policy['checkpoint_sha256']=='15268fe947b6425951b389e27c02dbba06055a77b1b1b5a14def2d23021082bb'
 def configure(c):c.eval_checkpoint=ck;c.pipeline.model.eval_num_rays_per_chunk=128;return c
 cfg,pipe,loaded,step=eval_setup(s/'config.yml',test_mode='val',eval_num_rays_per_chunk=128,update_config_callback=configure);assert step==6000 and Path(loaded).resolve()==ck.resolve();pipe.eval();m=pipe.model;f=m.field;before=digest_model(m)
 assert f.density_geometry_encoding is None and f.embedding_appearance is None and not f.enable_feature_reweighting and not f.freq_grid.enabled
 assert f.density_activation=='trunc_exp'and f.rgb_output_parameterization=='linear_softplus'
 with np.load(s/'grids/step000006000_core0.npz')as z:core=z['active'].copy();ema=z['occs'].copy();aabbs=z['aabb'].copy()
 assert np.array_equal(core,m.occupancy_grid.binaries.cpu().numpy())and np.array_equal(ema,m.occupancy_grid.occs.cpu().numpy())
 assert core.shape==(1,128,128,128);halo=maximum_filter(core,size=(1,3,3,3),mode='constant');lo=f.aabb[0];hi=f.aabb[1];N=128**3;device=m.device;dirs=torch.tensor(list(itertools.product([-1.,1.],repeat=3)),device=device);dirs/=dirs.norm(dim=-1,keepdim=True)
 def positions(ids):
  ids=torch.as_tensor(np.ascontiguousarray(ids),device=device,dtype=torch.long);cells=torch.stack([ids//16384,(ids//128)%128,ids%128],-1);return lo+(cells.float()+.5)/128*(hi-lo)
 def geo(pos):
  norm,sel,_,world=f._normalize_positions(pos);features=f._encode_with_optional_reweighting(norm,query_positions_flat=world);h=f.mlp_geo(features);logits=f._density_logits(norm,world,h);assert torch.equal(logits,h[:,:1]);return h,f.activate_density(logits)*sel[:,None]
 arrays={};modes={};color_extremes=[];started=time.time()
 try:
  for training,amp,label in [(False,False,'eval_noAMP'),(True,True,'train_AMP')]:
   f.train(training);h0=np.empty(N,np.float32);gmax=np.empty(N,np.float32);density=np.empty(N,np.float32)
   with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
    for start in range(0,N,a.chunk):
     end=min(start+a.chunk,N);h,sigma=geo(positions(np.arange(start,end)));h0[start:end]=h[:,0].float().cpu().numpy();gmax[start:end]=h[:,1:].float().abs().amax(-1).cpu().numpy();density[start:end]=sigma[:,0].float().cpu().numpy()
   arrays[label]=(h0,gmax,density);modes[label]=dict(h0=stats(h0),geo_absmax=stats(gmax),density=stats(density),h0_plus1_gt88=int((h0+1>88).sum()),h0_plus1_gt_log_float32max=int((h0.astype(np.float64)+1>np.log(np.finfo(np.float32).max)).sum()),raw_mlp_geo_dtype=str(h.dtype),activated_density_dtype=str(sigma.dtype))
   print(json.dumps(dict(stage='density_complete',mode=label,elapsed=time.time()-started,summary=modes[label])),flush=True)
  # Every cell in all eight fixed corner directions, bounded chunks; aggregate maxima only.
  f.eval();rgbmax=np.full(N,-np.inf,np.float32);rawmax=np.full(N,-np.inf,np.float32);rgb_bad=np.zeros(N,bool);raw_bad=np.zeros(N,bool);dir_report=[]
  with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):
   for di,d in enumerate(dirs):
    counts=dict(raw_nonfinite=0,activated_nonfinite=0);rm=-np.inf;am=-np.inf
    for start in range(0,N,a.chunk):
     end=min(start+a.chunk,N);h,_=geo(positions(np.arange(start,end)));enc=f.encode_directions(d.expand(len(h),3));raw=f.mlp_color(torch.cat([h[:,1:],enc],-1));rgb=f._activate_rgb(raw);rn=raw.float().cpu().numpy();an=rgb.float().cpu().numpy();raw_bad[start:end]|=~np.isfinite(rn).all(-1);rgb_bad[start:end]|=~np.isfinite(an).all(-1);counts['raw_nonfinite']+=int((~np.isfinite(rn)).sum());counts['activated_nonfinite']+=int((~np.isfinite(an)).sum());rawmax[start:end]=np.maximum(rawmax[start:end],np.max(np.abs(rn),axis=-1));rgbmax[start:end]=np.maximum(rgbmax[start:end],np.max(np.abs(an),axis=-1));rm=max(rm,float(np.nanmax(np.abs(rn))));am=max(am,float(np.nanmax(np.abs(an))))
    row=dict(direction=d.cpu().tolist(),raw_absmax=rm,activated_absmax=am,raw_dtype=str(raw.dtype),activated_dtype=str(rgb.dtype),**counts);dir_report.append(row);print(json.dumps(dict(stage='RGB_direction_complete',index=di,elapsed=time.time()-started,summary=row)),flush=True)
  # Union of bad and extreme cells: capture all eight raw and activated RGB outputs in both precision modes.
  h0=arrays['eval_noAMP'][0];bad=np.flatnonzero(~np.isfinite(arrays['eval_noAMP'][2])|~np.isfinite(arrays['train_AMP'][2])|rgb_bad|raw_bad);ids=np.unique(np.r_[bad[:a.bad_limit],np.argsort(np.nan_to_num(h0,nan=np.inf))[-512:],np.argsort(np.nan_to_num(rawmax,nan=np.inf))[-512:]]);details={};extreme_dtypes={};pos=positions(ids)
  for training,amp,label in [(False,False,'eval_noAMP'),(True,True,'train_AMP')]:
   f.train(training)
   with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
    h,sigma=geo(pos);raws=[];rgbs=[]
    for d in dirs:
     raw=f.mlp_color(torch.cat([h[:,1:],f.encode_directions(d.expand(len(h),3))],-1));rgb=f._activate_rgb(raw);raws.append(raw.float().cpu().numpy());rgbs.append(rgb.float().cpu().numpy())
   extreme_dtypes[label]=dict(raw_geometry=str(h.dtype),density=str(sigma.dtype),raw_RGB=str(raw.dtype),activated_RGB=str(rgb.dtype))
   details[label+'_h']=h.float().cpu().numpy();details[label+'_sigma']=sigma.float().cpu().numpy();details[label+'_raw_RGB']=np.stack(raws);details[label+'_RGB']=np.stack(rgbs)
  f.eval();assert digest_model(m)==before and np.array_equal(core,m.occupancy_grid.binaries.cpu().numpy())and np.array_equal(ema,m.occupancy_grid.occs.cpu().numpy())
  cells=np.stack([ids//16384,(ids//128)%128,ids%128],-1);np.savez_compressed(a.output/'extreme_and_bad_cells.npz',flat_ids=ids,cell_xyz=cells,world_xyz=pos.cpu().numpy(),native_core=core.reshape(-1)[ids],ordinary_onehalo=halo.reshape(-1)[ids],directions=dirs.cpu().numpy(),**details)
  def counts(mask):return dict(total=int(mask.sum()),native_core=int((mask&core.reshape(-1)).sum()),ordinary_onehalo=int((mask&halo.reshape(-1)).sum()),outside_onehalo=int((mask&~halo.reshape(-1)).sum()))
  top_density_ids=np.argsort(arrays['eval_noAMP'][2])[-16:][::-1];top_rgb_ids=np.argsort(rawmax)[-16:][::-1]
  def top_rows(ids):
   ids=np.asarray(ids);xyz=positions(ids).cpu().numpy();return [dict(flat_id=int(i),cell_xyz=[int(i//16384),int((i//128)%128),int(i%128)],world_xyz=xyz[k].tolist(),h0=float(arrays['eval_noAMP'][0][i]),density=float(arrays['eval_noAMP'][2][i]),geo_absmax=float(arrays['eval_noAMP'][1][i]),raw_RGB_absmax_8dirs=float(rawmax[i]),activated_RGB_absmax_8dirs=float(rgbmax[i]),native_core=bool(core.reshape(-1)[i]),ordinary_onehalo=bool(halo.reshape(-1)[i]))for k,i in enumerate(ids)]
  log_edges=[-np.inf,-20,-10,-5,0,5,10,20,40,60,80,87,88,89,100,np.inf];hist={label:dict(edges=log_edges,counts=np.histogram(v[0],log_edges)[0].tolist())for label,v in arrays.items()}
  report=dict(passed=True,checkpoint=str(ck),checkpoint_sha256=sha(ck),config_sha256=sha(s/'config.yml'),helper_sha256=sha(__file__),grid_receipt_sha256=sha(s/'grids/step000006000_receipt.json'),grid_sha256=sha(s/'grids/step000006000_core0.npz'),field_source=__import__('inspect').getsourcefile(type(f)),field_source_sha256=sha(__import__('inspect').getsourcefile(type(f))),torch=torch.__version__,gpu=torch.cuda.get_device_name(),cells=N,resolution=[128]*3,flat_order='x*128*128+y*128+z',field_aabb=f.aabb.cpu().tolist(),chunk=a.chunk,modes=modes,h0_histograms=hist,color_directions=dir_report,extreme_mode_dtypes=extreme_dtypes,top16_density=top_rows(top_density_ids),top16_raw_RGB=top_rows(top_rgb_ids),eval_color_raw_absmax=stats(rawmax),eval_color_activated_absmax=stats(rgbmax),bad_membership=dict(eval_density_nonfinite=counts(~np.isfinite(arrays['eval_noAMP'][2])),train_density_nonfinite=counts(~np.isfinite(arrays['train_AMP'][2])),eval_raw_RGB_nonfinite=counts(raw_bad),eval_activated_RGB_nonfinite=counts(rgb_bad),h0_plus1_gt88=counts(h0+1>88),sigma_gt88=counts(arrays['eval_noAMP'][2]>88),sigma_gt65504=counts(arrays['eval_noAMP'][2]>65504)),bad_cell_total=int(len(bad)),bad_cell_rows_saved=min(len(bad),a.bad_limit),extreme_rows_saved=int(len(ids)),extreme_archive_sha256=sha(a.output/'extreme_and_bad_cells.npz'),frozen_state_unchanged=True,state_sha256=before,optimizer_steps=0,callbacks_run=0,training_iterations=0,elapsed_seconds=time.time()-started,limitations=['Voxel centers are a finite numerical sampling, not whole continuous-volume proof.','Sixk finite snapshot cannot reconstruct unavailable failed7k state.','Eight corner directions do not bound every continuous RGB direction.','Core/halo are traversal admission, not physical occupancy labels.']);(a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(stage='DONE',elapsed=time.time()-started,bad_membership=report['bad_membership'])),flush=True)
 finally:
  f.eval();assert digest_model(m)==before
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--chunk',type=int,default=65536);p.add_argument('--bad-limit',type=int,default=10000);args=p.parse_args();assert 0<args.chunk<=65536;main(args)
