#!/usr/bin/env python3
"""Frozen native-grid diagnostic: fixed EMA core + actual SfM witness cells + one halo.
No model training or checkpoint/grid persistence. Restore full native state hash.
"""
import argparse,copy,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from probe_marshawn_occupancy import digest_model,file_sha,integrate


def build_candidate(ema,original,point_indices,threshold=.01):
    if original.ndim!=4 or original.shape[0]!=1:raise ValueError('One native cascade required')
    core=(ema.reshape_as(original)>=threshold);points=torch.zeros_like(original)
    points[0,point_indices[:,0],point_indices[:,1],point_indices[:,2]]=True
    union=core|points
    guarded=torch.nn.functional.max_pool3d(union[None].float(),3,1,1)[0].bool()
    return core,points,union,guarded


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;task=r/'native_ngp4_corrected_sdr';run=task/'marshawn_001660_native_ngp4_corrected_sdr/instant-ngp-bounded/cold_seed42';a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 sys.path.insert(0,str(r/'code_native'))
 from nerfstudio.utils.eval_utils import eval_setup
 from nerfstudio.model_components.scene_colliders import AABBBoxCollider
 spec=importlib.util.spec_from_file_location('native_free_audit',task/'audit_native_free_space.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 def configure(c):c.eval_checkpoint=run/'nerfstudio_models/step-000008000.ckpt';return c
 _,pipe,checkpoint,step=eval_setup(run/'config.yml',eval_num_rays_per_chunk=512,test_mode='val',update_config_callback=configure);model=pipe.model;before=digest_model(model);grid=model.occupancy_grid;original=grid.binaries.clone();ema=grid.occs.clone();box=grid.aabbs[0].detach().cpu().numpy();shape=np.array(original.shape[1:]);xyz=np.load(r/'geometry/train_witnesses.npz')['xyz'];inside=((xyz>=box[:3])&(xyz<box[3:])).all(1)
 if not inside.all():raise ValueError('All witness points must fit native cube')
 ijk=np.floor((xyz-box[:3])/(box[3:]-box[:3])*shape).astype(int);indices=torch.tensor(ijk,device=model.device);core,points,union,candidate=build_candidate(ema,original,indices);halo=torch.nn.functional.max_pool3d(original[None].float(),3,1,1)[0].bool();masks=dict(default=original,halo1=halo,fixed001_sfm_halo1=candidate,all_cells_reference=torch.ones_like(original));np.savez_compressed(a.output/'candidate_masks.npz',fixed_001_core=core.cpu().numpy(),observed_witness_cells=points.cpu().numpy(),union_before_halo=union.cpu().numpy(),guarded_candidate=candidate.cpu().numpy(),aabb=box)
 old=json.loads((task/'free_space_8k/audit.json').read_text());obs=np.load(r/'frequency_seed/observations.npz');meta=json.loads((r/'color_variants/gain4/sdr_png/transforms.json').read_text());byfile={Path(f['file_path']).name:f for f in meta['frames']};train=pipe.datamanager.train_dataset;frames=[byfile[Path(f).name] for f in train.image_filenames];camindex={f['physical_camera']:i for i,f in enumerate(frames)};cameras=train.cameras.to(model.device);facade=module.NativeAudit(model);collider=AABBBoxCollider(model.scene_box,near_plane=model.config.near_plane);renders={};reports={};traces={};archive=np.load(task/'free_space_8k/known_ray_evidence.npz');known_cells=archive['candidate_free_ray_counts']>0
 try:
  for label,mask in masks.items():
   grid.binaries.copy_(mask);renders[label]={};rows=[]
   for index,path in enumerate(pipe.datamanager.eval_dataset.image_filenames):
    camera=copy.deepcopy(pipe.datamanager.eval_dataset.cameras[index:index+1]).to(model.device);camera.rescale_output_resolution(320/int(camera.width.item()))
    with torch.inference_mode():out=model.get_outputs_for_camera(camera)
    arrays={k:out[k].float().cpu().numpy() for k in ['rgb','accumulation','depth']};name=Path(path).stem;np.savez_compressed(a.output/(label+'_'+name+'.npz'),**arrays);Image.fromarray(np.rint(np.clip(arrays['rgb'],0,1)*255).astype(np.uint8)).save(a.output/(label+'_'+name+'.png'));renders[label][name]=arrays
   for j,row in enumerate(old['rows']):
    camera=row['camera'];uv=obs['uv'][row['observation']];ci=camindex[camera];rays=cameras[ci:ci+1].generate_rays(0,coords=torch.tensor(uv[::-1].copy()[None],device=model.device,dtype=torch.float32),keep_shape=False);rays=collider(rays)
    with torch.inference_mode():out=facade(rays)
    w=out['packed_weights'].cpu().numpy();near=float(rays.nears.item());far=float(rays.fars.item());starts=near+out['packed_spacing_starts'].cpu().numpy()*(far-near);ends=near+out['packed_spacing_ends'].cpu().numpy()*(far-near);mid=(starts+ends)/2;opacity=float(w.sum());free_weight=float(w[mid<row['free_end']].sum());expected=float((w*mid).sum()/max(opacity,1e-20));rows.append(dict(observation=row['observation'],camera=camera,opacity=opacity,native_weight_before_margin=free_weight,conditional_depth=expected,relative_depth_error=(expected-row['witness_distance'])/row['witness_distance'],samples=len(w)));traces[f'{label}_{j}_weights']=w;traces[f'{label}_{j}_mid']=mid
   occupied=mask[0].cpu().numpy();reports[label]=dict(admitted_cells=int(occupied.sum()),admitted_volume_fraction=float(occupied.mean()),false_empty_observed_witnesses=int((~occupied[tuple(ijk.T)]).sum()),known_free_ray_cell_admission=float(occupied[known_cells].mean()),rays=rows,summary={key:dict(median=float(np.median([z[key] for z in rows])),p90=float(np.quantile([z[key] for z in rows],.9))) for key in ['native_weight_before_margin','relative_depth_error','opacity','samples']});(a.output/'progress.json').write_text(json.dumps(dict(completed_variant=label)))
 finally:grid.binaries.copy_(original)
 if not torch.equal(grid.occs,ema) or digest_model(model)!=before:raise ValueError('Frozen native model state not restored')
 for name in renders['default']:
  h,w=renders['default'][name]['rgb'].shape[:2];canvas=Image.new('RGB',(w*4,h+24));draw=ImageDraw.Draw(canvas)
  for j,label in enumerate(masks):canvas.paste(Image.open(a.output/(label+'_'+name+'.png')),(j*w,24));draw.text((j*w+3,3),label,fill='white')
  canvas.save(a.output/(name+'_comparison.jpg'),quality=94)
 np.savez_compressed(a.output/'native_ray_traces.npz',**traces);report=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),script_sha256=file_sha(__file__),step=step,source_state_sha256=before,frozen_state_restored_unchanged=True,policy='Core EMA>=FIXED.01 UNION actual observed train9022witness pointcells THEN one noncumulative radius1 dilation. No RGB/predictions/masks used in candidate.',candidate_counts=dict(core=int(core.sum()),witness_cells=int(points.sum()),union=int(union.sum()),guarded=int(candidate.sum())),variants=reports,limitations=['Witness pointcells are positive sparse evidence, not complete surfaces or occupiedwholevoxels.','Radius1 guard is conservative admission, not measured geometry; hidden volume unconstrained.','Known free ray counts do not certify whole cells empty.','Allcellsreference uses native sampler+alpha threshold+earlystop; not exact continuousfield integration.','Free margin matches original128audit (3voxel diagonals or5%distance), separatelyreported fromLOS objective max(.01,5%distance).','Fixed.01 may remove lowdensity physical thinstructure or density needed for RGB; diagnostic only; no checkpoint edits.']);(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:{f:v for f,v in z.items() if f!='rays'} for k,z in reports.items()}))
if __name__=='__main__':main()
