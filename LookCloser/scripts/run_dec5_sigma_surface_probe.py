#!/usr/bin/env python3
"""One excluded-view sigma-surface screen, with identical native RGB transfer."""
from pathlib import Path
import argparse,json,subprocess,sys,os,shutil
import numpy as np
from dec5_hair import cv_pose
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['root','campaign','calibration','output']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);n=a.root;s=Path(__file__).parent
 def run(script,*args):subprocess.run([sys.executable,str(s/script),*map(str,args)],check=True)
 data=json.load(open(n/'crop_native/transforms.json'));train=[f for f in data['frames'] if f['file_path'] in data['train_filenames']];target=json.load(open(n/'validation_4k_cameras.json'))['frames'][0];center=cv_pose(target)[:3,3];order=np.argsort([np.linalg.norm(cv_pose(f)[:3,3]-center) for f in train],kind='stable')[:8];names=[train[i]['physical_camera'] for i in order];source_by={f['physical_camera']:f for f in json.load(open(n/'source_hd_cameras.json'))['frames']};source_manifest=a.output/'source_cameras.json';target_manifest=a.output/'target_camera.json';atomic_json(source_manifest,dict(frame=data['frame_id'],frames=[source_by[name] for name in names]));atomic_json(target_manifest,dict(frame=data['frame_id'],frames=[target]));background=a.output/'fallback';background.mkdir();base=n/'ray_bg_fallback_uhd_hull_v2';filename=target['physical_camera']+'.npz';shutil.copy2(base/filename,background/filename);bg=json.load(open(base/'complete.json'));bg['records']=[r for r in bg['records'] if r['camera']==target['physical_camera']];bg['camera_manifest_sha256']=sha(target_manifest);atomic_json(background/'complete.json',bg)
 calibration=json.load(open(a.calibration));calibration['parent_calibration_sha256']=sha(a.calibration);calibration['refine_steps']=16;calibration['numerical_revision']='16 bisections chosen before receiver rendering; threshold and witnesses unchanged';calibration_path=a.output/'surface_refinement16.json';atomic_json(calibration_path,calibration)
 common=['--data',n/'crop_native','--training',n/'background_separated5k','--checkpoint',n/'background_separated5k/gate_004999.pt','--sigma-surface-calibration',calibration_path,'--rays-per-chunk','2048','--uint8-cpu-cache'];source=a.output/'sources';targets=a.output/'targets'
 run('export_dec5_native_depth.py',*common,'--source-semantics',n/'semantics_hd','--camera-manifest',source_manifest,'--output',source)
 run('export_dec5_native_depth.py',*common,'--parent',n/'roi_4k_hull_v2','--camera-manifest',target_manifest,'--output',targets)
 prediction=a.output/'prediction';metrics=a.output/'metrics.json'
 run('render_dec5_native_head_benchmark.py','--source-rgb-data',n/'crop_native','--source-cameras',source_manifest,'--source-depth',source,'--semantics',n/'semantics_hd','--target-depth',targets,'--fallback',background,'--cameras',target_manifest,'--depth-key','sigma_surface','--output',prediction)
 run('score_dec5_hair.py','--root',n/'benchmark_scene','--prediction',prediction,'--output',metrics,'--boundary-radius','5')
 atomic_json(a.output/'complete.json',dict(metrics_sha256=sha(metrics),prediction_receipt_sha256=sha(prediction/'complete.json'),source_order=names,source_order_is_same_global_nearest_eight=True,threshold_from_train_only=True,single_excluded_camera=True,uses_validation_rgb_for_fitting=False,uses_test_rgb=False))
if __name__=='__main__':main()
