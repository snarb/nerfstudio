"""FixedXYZ14k directionalRGB identifiability; no training or GT-fitted modifications."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,cv2
from PIL import Image,ImageDraw
R=Path('/fsx/tmp/marshawn_001660_hd');S=R/'native_png_random10k_floor_band_PRIOR_STAGED';O=S/'directional_floor14k_DIAGNOSTIC'
EXTRA={'KB':[[5,815],[130,820],[185,860],[205,945],[190,1007],[5,1007]],'LB':[[15,700],[150,690],[200,750],[200,880],[30,895],[5,830]],'LA':[[35,650],[180,650],[220,720],[220,870],[65,870],[20,790]],'JA':[[80,930],[235,920],[295,955],[295,1007],[80,1007]]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inv(x):
 t=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4);return t/np.maximum(1-t,1e-15)
def display(x):
 x=np.maximum(x,0);t=x/(1+x);return np.where(t<=.0031308,12.92*t,1.055*t**(1/2.4)-.055)
def main():
 for p in [R/'code_sfm/LookCloser/scripts',R/'code_native/LookCloser/scripts',R/'code_native',R/'final_pair/code',S/'code']:sys.path.insert(0,str(p))
 import torch
 from nerfstudio.utils.eval_utils import eval_setup
 from nerfstudio.cameras.rays import RaySamples,Frustums
 from nerfstudio.field_components.field_heads import FieldHeadNames
 from probe_marshawn_occupancy import digest_model
 from audit_marshawn_geometry import project_points
 torch.set_num_threads(2);cv2.setNumThreads(1);O.mkdir(exist_ok=True);data=R/'color_variants/gain4_scalar_trainfit_poseeval/sdr_png';linear=R/'color_variants/gain4_scalar_trainfit_poseeval/linear_rec709_exr';tf=json.loads((data/'transforms.json').read_text());by={q['physical_camera']:q for q in tf['frames']};trainnames=[q['physical_camera']for q in tf['frames']if q['file_path']in tf['train_filenames']];names=trainnames+['DC','HC','KA'];contract=json.loads((data/'color_contract.json').read_text());evs={q['camera']:q['exposure_correction_ev']for q in contract['rows']};proposal=json.loads((R/'floor_band_prior_PROPOSAL/proposal.json').read_text());poly={q['camera']:q['clear_floor_polygon']for q in proposal['floor_source_visuals']};poly.update(EXTRA);allobs=list(poly)+['KA'];sel=json.loads((S/'black_KA_gate12000/comparison.json').read_text());points=np.array([q['prior_plane_XYZ']for q in sel['rows']]);in_domain=[q['prior_domain_control']for q in sel['rows']];normal=np.array(proposal['fit_normal']);offsets=np.array([-.01,-.005,0,.005,.01]);XYZ=(points[:,None,:]+offsets[None,:,None]*normal).reshape(-1,3);origins=np.array([np.array(by[name]['transform_matrix'])[:3,3]for name in names]);directions=XYZ[:,None,:]-origins[None,:,:];directions/=np.linalg.norm(directions,axis=2,keepdims=True)
 ckpt=S/'run/marshawn_001660_random10k_finite_floor_PRIOR/instant-ngp-bounded/warm_parent10000/nerfstudio_models/step-000014000.ckpt';config=S/'config.yml'
 def configure(cfg):cfg.eval_checkpoint=ckpt;cfg.load_dir=None;cfg.load_step=None;cfg.load_checkpoint=None;cfg.load_config=None;return cfg
 _,pipe,_,_=eval_setup(config,test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=configure);pipe.eval();m=pipe.model;before=digest_model(m);assert m.field.embedding_appearance is None;assert m.field.direction_encoding.tcnn_encoding is not None
 # Allpositions exactlyfixed independentofdirection: zero-lengthsamples, get_density ONCE.
 with torch.inference_mode():
  X=torch.tensor(XYZ,dtype=torch.float32,device=m.device);v=torch.tensor(directions.reshape(-1,3),dtype=torch.float32,device=m.device);xp=X.repeat_interleave(len(names),0);zero=torch.zeros((len(xp),1),device=m.device);samples=RaySamples(frustums=Frustums(origins=xp,directions=v,starts=zero,ends=zero,pixel_area=torch.ones_like(zero)),camera_indices=torch.zeros_like(zero,dtype=torch.long));density,embedding=m.field.get_density(samples);outputs=m.field.get_outputs(samples,density_embedding=embedding);rgb=outputs[FieldHeadNames.RGB].float().cpu().numpy().reshape(9,5,len(names),3);sigma=density.float().cpu().numpy().reshape(9,5,len(names));assert (sigma.max(2)-sigma.min(2)).max()<1e-5
 assert digest_model(m)==before;del pipe,m;torch.cuda.empty_cache();np.savez_compressed(O/'field_fixedXYZ_arrays.npz',XYZ=points,normal_offsets=offsets,cameras=np.array(names),directions=directions.reshape(9,5,len(names),3),rgb=rgb,sigma=sigma)
 source={};sources=[]
 for name in allobs:
  row=by[name];src=data/row['file_path'];im=np.array(Image.open(src).convert('RGB'));path=linear/'images'/Path(row['file_path']).with_suffix('.exr').name;lin=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)[...,::-1].copy();assert lin.dtype==np.float32
  uv,front=project_points(row,points);entries=[]
  for j,(u,v)in enumerate(uv):
   visible=bool(front[j]and u>=4 and u<1916 and v>=4 and v<1008)
   if name in poly:
    for off in [-.02,0,.02]:
     vv,ff=project_points(row,(points[j]+off*normal)[None]);visible&=bool(ff[0]and cv2.pointPolygonTest(np.array(poly[name],np.float32),tuple(vv[0]),True)>8)
   if not visible:entries.append(dict(point=j,visible=False,uv=[float(u),float(v)]));continue
   x,y=int(np.floor(u)),int(np.floor(v));patch=im[y-3:y+4,x-3:x+4].astype(float)/255;patchlin=lin[y-3:y+4,x-3:x+4].astype(float);gt=np.median(patch.reshape(-1,3),axis=0);gtlin=np.median(patchlin.reshape(-1,3),axis=0);original=gtlin/2**evs[name];idx=names.index(name);pred=rgb[j,2,idx];yp=float(inv(pred)@np.array([.2126,.7152,.0722]));yg=float(gtlin@np.array([.2126,.7152,.0722]));entries.append(dict(point=j,visible=True,uv=[float(u),float(v)],GT_encoded_median_RGB=gt.tolist(),GT_exact_linear_median_RGB=gtlin.tolist(),GT_inverse_scalar_commonlinear_RGB=original.tolist(),input_EV=evs[name],field_plane_RGB=pred.tolist(),field_minus_GT_RGB=(pred-gt).tolist(),field_vs_GT_linear_EV=float(np.log2(max(yp,1e-12)/max(yg,1e-12))),GT_patch_linear_Y_std=float(np.std(patchlin@np.array([.2126,.7152,.0722]))),XYZoffset_modelRGB_range=(rgb[j,:,idx].max(0)-rgb[j,:,idx].min(0)).tolist(),source_png_vs_tonemap_linear_max_code_error=int(abs(np.rint(display(patchlin)*255).astype(int)-im[y-3:y+4,x-3:x+4].astype(int)).max())))
  source[name]=entries;sources.append(dict(camera=name,png=str(src),png_sha256=sha(src),linear=str(path),linear_sha256=sha(path)))
 # Perpointactualsourcecontacts; includeonlyvettedtrain sources; KA diagnostic, notfit.
 for j in range(9):
  obs=[(name,source[name][j])for name in allobs if source[name][j]['visible']];sheet=Image.new('RGB',(4*150,max(1,len(obs))*160),(25,25,25));draw=ImageDraw.Draw(sheet)
  for rr,(name,e)in enumerate(obs):
   row=by[name];im=Image.open(data/row['file_path']).convert('RGB');u,v=e['uv'];crop=im.crop((int(u)-32,int(v)-32,int(u)+32,int(v)+32));crop=crop.resize((128,128),Image.Resampling.NEAREST);sheet.paste(crop,(0,rr*160+30));draw.text((3,rr*160+3),f'q{j} {name} source',fill='white')
   vals=[e['GT_encoded_median_RGB'],e['field_plane_RGB'],display(np.array(e['GT_inverse_scalar_commonlinear_RGB']))]
   for cc,(val,label)in enumerate(zip(vals,['GTmedian','fixedXYZfield','GTundoScalar'])):
    col=tuple(np.rint(np.clip(val,0,1)*255).astype(int));sheet.paste(Image.new('RGB',(128,128),col),((cc+1)*150,rr*160+30));draw.text(((cc+1)*150+3,rr*160+3),label,fill='white')
  sheet.save(O/f'q{j:02d}_directional_contacts.jpg',quality=95)
 summaries=[];weights=np.array([.2126,.7152,.0722])
 for j in range(9):
  train=[name for name in poly if source[name][j]['visible']];values=np.array([source[name][j]['GT_exact_linear_median_RGB']for name in train]);base=np.array([source[name][j]['GT_inverse_scalar_commonlinear_RGB']for name in train]);pred=np.array([source[name][j]['field_plane_RGB']for name in train]);Y=values@weights;Ybase=base@weights;Ypred=inv(pred)@weights
  rec=dict(point=j,inside_finite_prior=in_domain[j],vetted_visible_train_cameras=train,corrected_GT_train_EV_span=float(np.ptp(np.log2(Y)))if len(train)else None,undo_scalar_GT_train_EV_span=float(np.ptp(np.log2(Ybase)))if len(train)else None,field_train_EV_span=float(np.ptp(np.log2(Ypred)))if len(train)else None,corrected_GT_train_logY_std=float(np.std(np.log2(Y)))if len(train)else None,undo_scalar_GT_train_logY_std=float(np.std(np.log2(Ybase)))if len(train)else None,KA_observation=source['KA'][j],source_observations={name:source[name][j]for name in train});summaries.append(rec)
 report=dict(checkpoint_sha256=sha(ckpt),config_sha256=sha(config),script_sha256=sha(__file__),input_contract_sha256=sha(data/'color_contract.json'),same_XYZ_density_embedding_all_directions=True,appearance_embedding=False,direction_encoding='NativeTCNN SHlevels4 =>samefrozenMLPhead',model_full_state_unchanged=True,prior_is_assumption_not_depthGT=True,XYZoffsets_tested=offsets.tolist(),extra_train_floor_polygons_manual=EXTRA,source_records=sources,rows=summaries,no_fit_or_training_or_output_changes=True,no_eval_RGB_used_to_fit=True,limitations=['Sourcepixels correspondviaassumedfiniteplane; depth/texture/pixel-footprint uncertainty remains.','7x7nativepatchmedians localflooronly; notexactphotometricpointmeasurement.','Realviewdependentreflectance andcameraresponseconfounded; undoKNOWNscalar isdiagnosticonly, notnewcalibration.','FieldRGBatfixedXYZ isnotnativeintegratedrenderRGB; integration/otherdensitycancontribute.']);(O/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print('done',[(q['point'],len(q['vetted_visible_train_cameras']),q['corrected_GT_train_EV_span'],q['undo_scalar_GT_train_EV_span'])for q in summaries],flush=True)
if __name__=='__main__':main()
