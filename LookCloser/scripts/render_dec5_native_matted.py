#!/usr/bin/env python3
"""Opt-in hair-qualified native source RGBA over a fixed ray-background layer.

Depth is unchanged. Person matte is not hair semantics or independent geometry.
Face/exterior retain the frozen parent RGB exactly. JPEG inverse display is an
approximate radiometric convention; no recovered-HDR claim.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json
import cv2,numpy as np,torch
from dec5_hair import cv_pose,VALIDATION,TEST
from study_dec5_hair_geometry import Renderer,display
from add_dec5_neural_fallback import inverse_display
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['data','source-cameras','source-depth','semantics','matting','parent','fallback','cameras','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--style',choices=['binary','alpha','fb'],default='fb');p.add_argument('--source-reference-camera',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(24*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
 source=json.load(open(a.source_depth/'complete.json'));parent=json.load(open(a.parent/'complete.json'));matte=json.load(open(a.matting/'complete.json'));fallback=json.load(open(a.fallback/'complete.json'));data=json.load(open(a.data/'transforms.json'));train=[f for f in data['frames'] if f['file_path'] in data['train_filenames']];by={f['physical_camera']:f for f in train};key=parent.get('depth_key','median')
 if len(train)!=59 or any(n in by for n in VALIDATION+TEST):raise ValueError('Train59 RGB required')
 if parent['source_receipt_sha256']!=sha(a.source_depth/'complete.json') or source['checkpoint_sha256']!=parent['checkpoint_sha256'] or source['camera_manifest_sha256']!=sha(a.source_cameras):raise ValueError('Frozen source/parent geometry mismatch')
 if matte['data_sha256']!=sha(a.data/'transforms.json') or parent['source_rgb_dataset_sha256']!=sha(a.data/'transforms.json') or parent['fallback_receipt_sha256']!=sha(a.fallback/'complete.json') or fallback['camera_manifest_sha256']!=sha(a.cameras):raise ValueError('RGB/matting/fallback mismatch')
 anchor_sha=sha(a.source_reference_camera) if a.source_reference_camera else None
 if parent.get('source_reference_camera_sha256')!=anchor_sha:raise ValueError('Source order differs from parent')
 if any(r.get('uses_validation_rgb') or r.get('uses_test_rgb') for r in [source,parent,matte,fallback]):raise ValueError('No target RGB fitting inputs accepted')
 frames=json.load(open(a.source_cameras))['frames'];source_by={r['camera']:r for r in source['records']};matte_by={r['camera']:r for r in matte['records']};parents={r['camera']:r for r in parent['records']};fallbacks={r['camera']:r for r in fallback['records']}
 renderer=object.__new__(Renderer);scene=SimpleNamespace(cameras=frames,frames=frames,centers=np.stack([cv_pose(f)[:3,3] for f in frames]),allowed_visibility_sources=set(source_by));renderer.scene=scene;renderer.visibility_depth_sampling='bilinear';renderer.depth=[];renderer.semantic=[];renderer.rgb=[];renderer.matte_hair_domain=[]
 if a.source_reference_camera:scene.source_reference_center=cv_pose(json.load(open(a.source_reference_camera)))[:3,3]
 for f in frames:
  n=f['physical_camera'];native=by[n];depth_path=a.source_depth/(n+'.npz');matte_path=a.matting/(n+'.npz');rgb_path=a.data/native['file_path']
  if sha(depth_path)!=source_by[n]['sha256'] or sha(matte_path)!=matte_by[n]['sha256'] or sha(rgb_path)!=matte_by[n]['source_rgb_sha256']:raise ValueError('Changed source/matte payload')
  np.testing.assert_allclose(f['transform_matrix'],native['transform_matrix'],rtol=0,atol=1e-10);sx,sy=native['w']/f['w'],native['h']/f['h'];np.testing.assert_allclose([f['fl_x']*sx,f['fl_y']*sy,f['cx']*sx,f['cy']*sy],[native[k] for k in ['fl_x','fl_y','cx','cy']],rtol=0,atol=1e-5)
  sem=np.load(a.semantics/(n+'.npz'))['probability'].astype(np.float32)/65535;image=cv2.imread(str(rgb_path))[...,::-1].astype(np.float32)/255;renderer.depth.append(torch.as_tensor(np.load(depth_path)[key],device='cuda')[None]);renderer.semantic.append(torch.as_tensor(sem,device='cuda').permute(2,0,1));renderer.rgb.append(torch.as_tensor(inverse_display(image),device='cuda').permute(2,0,1));domain=cv2.dilate((sem[...,0]>.1).astype(np.uint8),np.ones((33,33),np.uint8)).astype(bool)&(sem[...,1]<.7);renderer.matte_hair_domain.append(torch.as_tensor(domain,device='cuda',dtype=torch.float32)[None])
 renderer.load_mattes(a.matting)
 if a.style=='binary':renderer.alpha=[(v>=.5).float() for v in renderer.alpha]
 records=[];all_centers=np.stack([cv_pose(f)[:3,3] for f in train])
 for f in json.load(open(a.cameras))['frames']:
  n=f['physical_camera'];path=a.parent/(n+'.npz');fp=a.fallback/(n+'.npz')
  if sha(path)!=parents[n]['npz_sha256'] or sha(fp)!=fallbacks[n]['sha256']:raise ValueError('Changed frozen parent/fallback')
  center=getattr(scene,'source_reference_center',cv_pose(f)[:3,3]);expected=[train[i]['physical_camera'] for i in np.argsort(np.linalg.norm(all_centers-center,axis=-1),kind='stable')[:8]]
  if [frames[i]['physical_camera'] for i in renderer.source_order(f)]!=expected:raise ValueError('Source subset changes nearest-eight')
  arrays=dict(np.load(path));original=arrays['rgb'].copy();old_png=cv2.imread(str(a.parent/(n+'.png')));depth=arrays['front_depth'];roi=arrays['roi'];bg=np.load(fp);np.testing.assert_array_equal(roi,bg['roi']);layer={k:v.cpu().numpy() for k,v in renderer.warp_layer(f,torch.as_tensor(depth,device='cuda'),foreground=True,decontaminate=a.style=='fb',premultiply=True,allow_zero_alpha=True).items()};alpha=layer['alpha'].clip(0,1);padding=max(1,round(16*f['w']/1920));domain=cv2.dilate((arrays['hair_probability']>.1).astype(np.uint8),np.ones((2*padding+1,2*padding+1),np.uint8)).astype(bool);skin=arrays['skin_probability']>=.5;positive=(alpha>.001)&(layer['support']>=2);zero=(alpha<=.001)&(layer['support']>=3);gate=roi&domain&~skin&(layer['source_id']>=0)&(positive|zero);mixed=layer['rgb']+inverse_display(bg['rgb_display'])*(1-alpha[...,None])
  if not np.isfinite(mixed[gate]).all():raise ValueError('Nonfinite RGBA composition')
  arrays['rgb'][gate]=mixed[gate];arrays['source_id'][gate]=np.where(alpha[gate]>.001,layer['source_id'][gate],-1);arrays['support'][gate]=layer['support'][gate];coverage=arrays.get('hair_coverage',arrays['hair_probability']).copy();coverage[gate]=arrays['hair_probability'][gate]*alpha[gate];arrays['hair_coverage']=coverage;arrays['rgba_gate']=gate;arrays['source_alpha']=alpha;arrays['foreground_premultiplied']=layer['rgb']
  np.testing.assert_array_equal(arrays['rgb'][~gate],original[~gate]);np.testing.assert_array_equal(arrays['front_depth'],depth);np.savez_compressed(a.output/(n+'.npz'),**arrays);png=old_png.copy();new=np.rint(display(mixed)*255).astype(np.uint8)[...,::-1];png[gate]=new[gate];np.testing.assert_array_equal(png[~gate],old_png[~gate]);cv2.imwrite(str(a.output/(n+'.png')),png);cv2.imwrite(str(a.output/(n+'_review.jpg')),cv2.resize(np.rot90(png),(360,640),interpolation=cv2.INTER_AREA));records.append(dict(camera=n,npz_sha256=sha(a.output/(n+'.npz')),png_sha256=sha(a.output/(n+'.png')),edited_pixels=int(gate.sum()),zero_alpha_pixels=int((gate&(alpha<=.001)).sum())));atomic_json(a.output/'progress.json',dict(completed=len(records),total=len(parents)))
 atomic_json(a.output/'complete.json',dict(records=records,parent_receipt_sha256=sha(a.parent/'complete.json'),matting_receipt_sha256=sha(a.matting/'complete.json'),source_receipt_sha256=sha(a.source_depth/'complete.json'),source_rgb_dataset_sha256=sha(a.data/'transforms.json'),fallback_receipt_sha256=sha(a.fallback/'complete.json'),checkpoint_sha256=parent['checkpoint_sha256'],depth_key=key,style=a.style,source_reference_camera_sha256=anchor_sha,source_selection='same ordered nearest-eight, prefer valid hair alpha; zero alpha needs three known source witnesses',hair_domain='source coarse hair dilated 16 HD px excluding confident face; target hair dilated 16 HD-equivalent px with protected skin',geometry_unchanged=True,exact_rgb_outside_gate=True,background='fixed train-only ray-space color function; not finite background depth',uses_validation_rgb=False,uses_test_rgb=False,implementation_sha256=sha(__file__)))
if __name__=='__main__':main()
