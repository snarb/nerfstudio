"""Frozen14k density-supported normal/tangent RGB sensitivity; no optimizers."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
R=Path('/fsx/tmp/marshawn_001660_hd');S=R/'native_png_random10k_floor_band_PRIOR_STAGED';O=S/'directional_floor14k_DIAGNOSTIC'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 for p in [R/'code_sfm/LookCloser/scripts',R/'code_native/LookCloser/scripts',R/'code_native',R/'final_pair/code',S/'code']:sys.path.insert(0,str(p))
 import torch
 from nerfstudio.utils.eval_utils import eval_setup
 from nerfstudio.cameras.rays import RaySamples,Frustums
 from nerfstudio.field_components.field_heads import FieldHeadNames
 from probe_marshawn_occupancy import digest_model
 torch.set_num_threads(2);a=json.loads((O/'audit.json').read_text());z=np.load(O/'field_fixedXYZ_arrays.npz');points=z['XYZ'];names=z['cameras'].tolist();normal=np.array(json.loads((R/'floor_band_prior_PROPOSAL/proposal.json').read_text())['fit_normal']);directions=z['directions'][:,2];offsets=np.arange(-.04,.04001,.0005);xyz=points[:,None,:]+offsets[None,:,None]*normal;ckpt=S/'run/marshawn_001660_random10k_finite_floor_PRIOR/instant-ngp-bounded/warm_parent10000/nerfstudio_models/step-000014000.ckpt'
 def configure(cfg):cfg.eval_checkpoint=ckpt;cfg.load_dir=None;cfg.load_step=None;cfg.load_checkpoint=None;cfg.load_config=None;return cfg
 _,pipe,_,_=eval_setup(S/'config.yml',test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=configure);pipe.eval();m=pipe.model;before=digest_model(m)
 def query(X,D):
  with torch.inference_mode():
   xp=torch.as_tensor(X,dtype=torch.float32,device=m.device);v=torch.as_tensor(D,dtype=torch.float32,device=m.device);zero=torch.zeros((len(X),1),device=m.device);samples=RaySamples(frustums=Frustums(origins=xp,directions=v,starts=zero,ends=zero,pixel_area=torch.ones_like(zero)),camera_indices=torch.zeros_like(zero,dtype=torch.long));density,embedding=m.field.get_density(samples);rgb=m.field.get_outputs(samples,density_embedding=embedding)[FieldHeadNames.RGB];return density.float().cpu().numpy(),rgb.float().cpu().numpy()
 X=np.repeat(xyz[:,:,None,:],len(names),2);D=np.repeat(directions[:,None,:,:],len(offsets),1);sig,rgb=query(X.reshape(-1,3),D.reshape(-1,3));sig=sig.reshape(9,len(offsets),len(names));rgb=rgb.reshape(9,len(offsets),len(names),3);assert (sig.max(2)-sig.min(2)).max()<1e-5;band=abs(offsets)<=.02001;ridgeidx=np.flatnonzero(band)[np.argmax(sig[:,band,0],axis=1)];ridge=xyz[np.arange(9),ridgeidx];e1=np.array([1,0,-normal[0]/normal[2]]);e1/=np.linalg.norm(e1);e2=np.cross(normal,e1);step=np.array([-.005,-.002,0,.002,.005]);tangent= ridge[:,None,None,:]+np.array([e1,e2])[None,:,None,:]*step[None,None,:,None];XX=np.repeat(tangent[:,:,:,None,:],len(names),3);DD=np.broadcast_to(directions[:,None,None,:,:],XX.shape);ts,tr=query(XX.reshape(-1,3),DD.reshape(-1,3));tr=tr.reshape(9,2,5,len(names),3);ts=ts.reshape(9,2,5,len(names));assert digest_model(m)==before;del pipe,m;torch.cuda.empty_cache();np.savez_compressed(O/'normal_dense_arrays.npz',offsets=offsets,sigma=sig,rgb=rgb,ridge_offset=offsets[ridgeidx],tangent_step=step,tangent_rgb=tr,tangent_sigma=ts,cameras=np.array(names))
 summaries=[]
 for j in range(9):
  density=sig[j,:,0];peak=density[ridgeidx[j]];support=(density>=.1*peak)&band;summary=dict(point=j,inside_prior=a['rows'][j]['inside_finite_prior'],ridge_offset=float(offsets[ridgeidx[j]]),ridge_sigma=float(peak),plane_sigma=float(density[np.argmin(abs(offsets))]),density_support_offsets=[float(offsets[support].min()),float(offsets[support].max())],directions=[])
  for name in a['rows'][j]['vetted_visible_train_cameras']+['KA']:
   k=names.index(name);r=rgb[j,:,k];entry=dict(camera=name,dense_supported_RGB_range=(r[support].max(0)-r[support].min(0)).tolist(),ridge_RGB=r[ridgeidx[j]].tolist(),GT=a['rows'][j]['KA_observation']['GT_encoded_median_RGB']if name=='KA'else a['rows'][j]['source_observations'][name]['GT_encoded_median_RGB'],local_normal={})
   for delta in [.001,.002,.005]:
    lo=int(np.argmin(abs(offsets-(offsets[ridgeidx[j]]-delta))));hi=int(np.argmin(abs(offsets-(offsets[ridgeidx[j]]+delta))));entry['local_normal'][str(delta)]=dict(pair_RGB_absdiff=abs(r[hi]-r[lo]).tolist(),sigma_pair=[float(density[lo]),float(density[hi])],sigma_pair_over_peak=[float(density[lo]/peak),float(density[hi]/peak)],RGB_lo=r[lo].tolist(),RGB_hi=r[hi].tolist())
   entry['tangent_plusminus002_absdiff']=abs(tr[j,:,3,k]-tr[j,:,1,k]).tolist();entry['tangent_plusminus005_absdiff']=abs(tr[j,:,4,k]-tr[j,:,0,k]).tolist();summary['directions'].append(entry)
  summaries.append(summary)
 (O/'normal_dense_audit.json').write_text(json.dumps(dict(rows=summaries,checkpoint_sha256=sha(ckpt),script_sha256=sha(__file__),field_state_unchanged=True,same_direction_at_all_normal_positions=True,density_support_rule='>=10%bandpeakσ withinassumed±.02; notphysicalsurfaceGT',no_optimizer_or_source_data_modification=True),indent=2)+'\n');print('done',[(q['point'],q['ridge_offset'],q['ridge_sigma'])for q in summaries],flush=True)
if __name__=='__main__':main()
