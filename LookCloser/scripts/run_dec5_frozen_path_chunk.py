#!/usr/bin/env python3
"""Export and render one disjoint frozen camera-path chunk; never fits weights."""
from pathlib import Path
import argparse,json,subprocess,sys
import cv2,numpy as np,torch
from dec5_hair import cv_pose
from dec5_ray_background import RayBackgroundNetwork
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['root','training','checkpoint','cameras','source-depth','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--source-reference-camera',type=Path)
 p.add_argument('--depth-keys',nargs='+',default=['median']);p.add_argument('--roi-dir',type=Path)
 p.add_argument('--minimum-opacity',type=float,default=.1);p.add_argument('--prepare-only',action='store_true')
 p.add_argument('--data',choices=['crop_hd','crop_native'],required=True);a=p.parse_args();torch.set_num_threads(2)
 n=a.root;a.output.mkdir(parents=True,exist_ok=False);scripts=Path(__file__).parent
 def run(script,*args):subprocess.run([sys.executable,str(scripts/script),*map(str,args)],check=True)
 depth=a.output/'depth';roi_dir=a.roi_dir or n/'roi_pilot_hull_v2';extra=[]
 for key in a.depth_keys:
  if key in ['median','expected']:continue
  if len(key)!=5 or key[0]!='q' or not key[1:].isdigit():raise ValueError('Unknown depth key')
  extra.append(str(int(key[1:])/10000))
 run('export_dec5_native_depth.py','--data',n/a.data,'--training',a.training,'--checkpoint',a.checkpoint,'--parent',roi_dir,'--camera-manifest',a.cameras,'--output',depth,'--uint8-cpu-cache','--minimum-opacity',a.minimum_opacity,*(['--include-median'] if 'median' in a.depth_keys else []),*(['--extra-quantiles',*extra] if extra else []))
 # Evaluate the same frozen train-only ray background at every target ray.
 bgpath=n/'ray_background/background_portable.pt';state=torch.load(bgpath,map_location='cpu',weights_only=False)
 request=json.load(open(n/'ray_background/request.json'))
 if request['allowed_serving_dataset_sha256']!=sha(n/'crop_native/transforms.json'):raise ValueError('Background serving calibration mismatch')
 model=RayBackgroundNetwork().cuda().eval();model.load_state_dict(state['state_dict']);center=torch.as_tensor(state['center'],device='cuda');radius=state['radius'];fallback=a.output/'fallback';fallback.mkdir();records=[]
 for f in json.load(open(a.cameras))['frames']:
  name=f['physical_camera'];roi=np.load(roi_dir/(name+'.npz'))['roi'];y,x=np.nonzero(roi);image=np.zeros((*roi.shape,3),np.float32);pose=torch.as_tensor(cv_pose(f),device='cuda',dtype=torch.float32);origin=(pose[:3,3]-center)/radius
  with torch.no_grad():
   for begin in range(0,len(x),65536):
    xx=torch.as_tensor(x[begin:begin+65536],device='cuda');yy=torch.as_tensor(y[begin:begin+65536],device='cuda');direction=torch.stack(((xx+.5-f['cx'])/f['fl_x'],(yy+.5-f['cy'])/f['fl_y'],torch.ones_like(xx)),1).float()@pose[:3,:3].T;direction=torch.nn.functional.normalize(direction,dim=-1);rgb=model(origin.expand(len(xx),3),direction);image[y[begin:begin+len(xx)],x[begin:begin+len(xx)]]=torch.round(rgb.clamp(0,1)*255).cpu().numpy()/255
  path=fallback/(name+'.npz');np.savez_compressed(path,rgb_display=image,roi=roi);records.append(dict(camera=name,sha256=sha(path)))
 del model;torch.cuda.empty_cache()
 atomic_json(fallback/'complete.json',dict(records=records,camera_manifest_sha256=sha(a.cameras),checkpoint_sha256=sha(bgpath),uses_validation_rgb=False,uses_test_rgb=False,artifact_kind='common frozen train-only background fallback'))
 if a.prepare_only:
  atomic_json(a.output/'complete.json',dict(status='prepared',checkpoint_sha256=sha(a.checkpoint),camera_manifest_sha256=sha(a.cameras),fitting=False,minimum_opacity=a.minimum_opacity));return
 prediction_receipts={}
 for depth_key in a.depth_keys:
  prediction=a.output/('prediction' if depth_key=='median' else 'prediction_'+depth_key)
  run('render_dec5_native_head_benchmark.py','--source-rgb-data',n/'crop_native','--source-cameras',n/'source_hd_cameras.json','--source-depth',a.source_depth,'--semantics',n/'semantics_hd','--target-depth',depth,'--fallback',fallback,'--cameras',a.cameras,'--depth-key',depth_key,'--output',prediction,*(['--source-reference-camera',a.source_reference_camera] if a.source_reference_camera else []))
  prediction_receipts[depth_key]=sha(prediction/'complete.json')
 atomic_json(a.output/'complete.json',dict(status='complete',checkpoint_sha256=sha(a.checkpoint),camera_manifest_sha256=sha(a.cameras),prediction_receipt_sha256=prediction_receipts.get('median'),prediction_receipts=prediction_receipts,depth_keys=a.depth_keys,fitting=False,uses_validation_rgb=False,uses_test_rgb=False))

if __name__=='__main__':main()
