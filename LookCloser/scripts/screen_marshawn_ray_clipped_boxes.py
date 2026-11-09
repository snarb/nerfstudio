"""Read-only teacher AABB adequacy screen: clip ray t, NEVER renormalize field."""
from pathlib import Path
import json,sys,copy,types,hashlib
import numpy as np
from PIL import Image,ImageDraw
R=Path('/fsx/tmp/marshawn_001660_hd');S=R/'native_png_radialK4_COLD_STAGED';OUT=R/'radial_bbox_teacher_clip_PRECHECK'
for p in [R/'code_sfm/LookCloser/scripts',R/'code_native/LookCloser/scripts',R/'code_native',S/'code',Path(__file__).parent]:sys.path.insert(0,str(p))
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def slab(origins,directions,box,near=.052,far=1000.,dt=.003):
 o,d=origins.double(),directions.double();b=torch.as_tensor(box,device=o.device,dtype=torch.double);parallel=abs(d)<1e-12;den=torch.where(parallel,torch.ones_like(d),d);a=(b[0]-o)/den;c=(b[1]-o)/den;lo=torch.where(parallel,torch.full_like(a,-torch.inf),torch.minimum(a,c)).amax(-1);hi=torch.where(parallel,torch.full_like(a,torch.inf),torch.maximum(a,c)).amin(-1);valid=~(parallel&((o<b[0])|(o>b[1]))).any(-1);lo=lo.clamp_min(near);hi=hi.clamp_max(far)
 # Conservative whole-step entry/exit: same .052+n*.003 lattice, cone0.
 lo=near+torch.ceil(((lo-near)/dt).clamp_min(0)-1e-10)*dt;hi=near+torch.floor((hi-near)/dt+1e-10)*dt;valid&=hi>lo
 lo=torch.where(valid,lo,torch.full_like(lo,near));hi=torch.where(valid,hi,torch.full_like(hi,near));return lo.float(),hi.float(),valid

def unit_tests():
 o=torch.tensor([[0.,0.,0.],[2,0,0],[1,0,0],[0,0,0],[0,0,0]]);d=torch.tensor([[1.,0,0],[0,1,0],[0,1,0],[0,0,-1],[0,0,1.]])
 lo,hi,v=slab(o,d,[[-1,-1,-1],[1,1,1]],near=0,dt=.1)
 assert v.tolist()==[True,False,True,True,True];assert torch.allclose(hi[v],torch.ones(4));assert lo[1]==hi[1]==0
 lo,hi,v=slab(torch.tensor([[-2.,0,0]]),torch.tensor([[1.,0,0]]),[[-1,-1,-1],[1,1,1]])
 assert abs(float((lo-.052)/.003)-round(float((lo-.052)/.003)))<1e-4;assert lo>=1 and hi<=3
 return dict(parallel_outside_miss=True,parallel_on_boundary_admitted=True,inside_entry_and_both_direction_exit=True,phase_rounded_entry_inside=True,invalid_near_equals_far_zero_length=True)

def main():
 tests=unit_tests();OUT.mkdir(exist_ok=False);torch.set_num_threads(2)
 from nerfstudio.utils.eval_utils import eval_setup
 from probe_marshawn_occupancy import digest_model
 from marshawn_release_common import native_bounds,trace_native
 from torchmetrics.functional import structural_similarity_index_measure
 from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
 ck=S/'immutable_selected_12000/step-000012000.ckpt';config=S/'config.yml'
 def configure(c):c.eval_checkpoint=ck;c.load_checkpoint=c.load_dir=c.load_step=c.load_config=None;return c
 cfg,pipe,actual,step=eval_setup(config,test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=configure);pipe.eval();m=pipe.model;m.current_train_step=step;assert actual.resolve()==ck.resolve();assert m.config.cone_angle==0 and m.config.render_step_size==.003 and m.config.near_plane==.052
 bounds=native_bounds(m);before=digest_model(m);grid=m.occupancy_grid.binaries.clone();ema=m.occupancy_grid.occs.clone();fieldbox=m.field.aabb.clone();boxes=json.loads((R/'geometry/boxes_approved.json').read_text())['scopes']['scene'];original=m.get_outputs;old_override=m.__dict__.get('get_outputs');had_override='get_outputs'in m.__dict__;active=[None];stats={};phase=[]
 def clipped(self,ray_bundle):
  if active[0]is not None:
   lo,hi,ok=slab(ray_bundle.origins,ray_bundle.directions,boxes[active[0]])
   if ray_bundle.nears is not None:lo=torch.maximum(lo,ray_bundle.nears.reshape(-1));hi=torch.maximum(lo,hi)
   if ray_bundle.fars is not None:hi=torch.minimum(hi,ray_bundle.fars.reshape(-1));lo=torch.minimum(lo,hi)
   ray_bundle.nears=lo[:,None];ray_bundle.fars=hi[:,None];z=stats.setdefault(active[0],dict(rays=0,misses=0));z['rays']+=len(lo);z['misses']+=int((~ok).sum())
  return original(ray_bundle)
 m.get_outputs=types.MethodType(clipped,m);rows=[];panels=[];lp=LearnedPerceptualImagePatchSimilarity(normalize=True).to(m.device).eval()
 def im(v):return Image.fromarray(np.rint(np.clip(v,0,1)*255).astype(np.uint8))
 try:
  ds=pipe.datamanager.eval_dataset
  for ci in range(3):
   name=Path(ds.image_filenames[ci]).stem.split('_')[-1];cam=copy.deepcopy(ds.cameras[ci:ci+1]).to(m.device);cam.rescale_output_resolution(640/1920);w,h=int(cam.width.item()),int(cam.height.item());import cv2
   gt=cv2.resize(ds.get_image_float32(ci).cpu().numpy()[...,:3],(w,h),interpolation=cv2.INTER_AREA);images={'GT':gt};record=dict(camera=name,variants={});baseline=None
   for variant in ['default','tight','medium','loose']:
    active[0]=None if variant=='default'else variant
    with torch.inference_mode():out=m.get_outputs_for_camera(cam)
    arr={k:v.detach().float().cpu().numpy()for k,v in out.items()if torch.is_tensor(v)};rgb=arr['rgb'];assert np.isfinite(rgb).all();images[variant]=rgb;im(rgb).save(OUT/f'{name}_{variant}.png');np.savez_compressed(OUT/f'{name}_{variant}.npz',**arr,gt=gt)
    if baseline is None:baseline=rgb
    a=torch.tensor(rgb,device=m.device).permute(2,0,1)[None];b=torch.tensor(gt,device=m.device).permute(2,0,1)[None]
    with torch.inference_mode():ssim=float(structural_similarity_index_measure(a,b,data_range=1.));distance=float(lp(a,b))
    lp.reset();diff=abs(rgb-baseline);mse=float(np.mean((rgb-gt)**2));record['variants'][variant]=dict(PSNR=float(-10*np.log10(max(mse,1e-12))),SSIM=ssim,LPIPS=distance,mean_abs_default_delta=float(diff.mean()),max_abs_default_delta=float(diff.max()),channels_delta_over1code=float((diff>1/255).mean()))
   # Trace actual retained sample positions to qualify phase assumptions.
   for x,y in [(320,170),(160,220),(500,200)]:
    coords=torch.tensor([[y+.5,x+.5]],device=m.device);active[0]=None;ray=cam.generate_rays(0,coords=coords,keep_shape=False);base=trace_native(m,ray)
    for variant in ['tight','medium','loose']:
     active[0]=variant;ray=cam.generate_rays(0,coords=coords,keep_shape=False);test=trace_native(m,ray);bs,ts=base['starts'],test['starts'];errors=np.min(abs(ts[:,None]-bs[None,:]),axis=1)if len(bs)and len(ts)else np.array([]);inside=(ts>=bs.min()-1e-5)&(ts<=bs.max()+1e-5)if len(bs)and len(ts)else np.zeros(len(ts),bool);shared=errors[inside]if len(errors)else errors;phase.append(dict(camera=name,pixel=[x,y],variant=variant,baseline_samples=len(bs),clipped_samples=len(ts),retained_range_sample_nearest_start_max_error=float(shared.max())if len(shared)else None,phase_comparison_scope='Samples withinsharedbaseline termination range; behind newly visible surfaces excluded'))
   panel=Image.new('RGB',(w*5,h+25),'black');d=ImageDraw.Draw(panel)
   for col,label in enumerate(['GT','default','tight','medium','loose']):panel.paste(im(images[label]),(col*w,25));d.text((col*w+4,4),name+' '+label,fill='white')
   panel.save(OUT/f'{name}_comparison.jpg',quality=95);panels.append(panel);rows.append(record);(OUT/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=3)))
 finally:
  if had_override:m.get_outputs=old_override
  else:delattr(m,'get_outputs')
 assert digest_model(m)==before;assert torch.equal(grid,m.occupancy_grid.binaries)and torch.equal(ema,m.occupancy_grid.occs)and torch.equal(fieldbox,m.field.aabb)
 report=dict(kind='TEACHER_RAY_ONLY_AABB_CLIP_DIAGNOSTIC_NO_TRAINING',checkpoint=str(actual),checkpoint_sha256=sha(actual),config_sha256=sha(config),helper_sha256=sha(__file__),boxes=boxes,unit_tests=tests,live_bounds=bounds,field_normalization_AABB_EMA_grid_and_NN_byte_unchanged=True,phase_policy='entryceil/exitfloor on original.052+n*.003 lattice, cone0',miss_handling='equalnear/far=>zero optical length, nativeemptydummyintervalzero opacity',ray_counts=stats,phase_checks=phase,rows=rows,limitations=['TeacherfieldoutsidephysicalAABBmaybefalsegeometry; sourceGTmanualreviewrequired','Conservativewhole-stepedgeclipremovesupto.003raydistance','Not a cold-training comparison, not alternate field normalization or grid clipping'])
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(output=str(OUT),rows=rows)),flush=True)
if __name__=='__main__':main()
