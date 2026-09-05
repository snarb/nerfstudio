#!/usr/bin/env python3
"""Re-hash the isolated DEC5 color canaries, without certifying a repaired recipe."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import subprocess
from PIL import Image
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256


def check_hash(path,expected):
    if sha256(path)!=expected:raise ValueError(f'Checksum mismatch: {path}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root;diagnostic=root/'diagnostics/000973'
    fits=[];renders=[];reviews=[];metrics=[]
    for folder in [diagnostic/'color_calibration',root/'frames/001059/color_calibration']:
        for path in sorted(folder.glob('calibration*.json')):
            fit=json.loads(path.read_text())
            cameras=fit['cameras']
            if len(cameras)!=62 or {'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}&cameras.keys():
                raise ValueError('Wrong train inventory')
            if fit['uses_eval_rgb'] or fit['uses_semantic_masks'] or fit['source_averaging']:
                raise ValueError('Forbidden calibration inputs or RGB averaging')
            for camera in cameras.values():
                check_hash(Path(camera['image']),camera['image_sha256'])
                for key in ('exposure_gain','rgb_gain'):
                    if not all(math.isfinite(v) and v>0 for v in camera[key]):raise ValueError('Invalid gains')
            for values in fit['validation_display_pair_l1'].values():
                if not all(math.isfinite(values[k]) for k in ('median','p90')):raise ValueError('Nonfinite validation')
            fits.append({'path':str(path),'sha256':sha256(path),'train_cameras':len(cameras),
                         'pair_residuals':fit['validation_display_pair_l1']})
    cases=[(diagnostic/'mvs_texture_global','path3_texel/result.json'),
           (diagnostic/'mvs_texture_exposure_clamp_v2','path3/result.json'),
           (diagnostic/'color_calibration/spatial16_cut_three_views','path_manifest.json'),
           (diagnostic/'color_calibration/overlap16_cut_three_views','path_manifest.json')]
    for folder,manifest_name in cases:
        manifest=json.loads((folder/manifest_name).read_text())
        if manifest['state']!='complete' or len(manifest['views'])!=3:raise ValueError('Incomplete render canary')
        for row in manifest['views']:
            path=Path(row['render']);check_hash(path,row['sha256'])
            with Image.open(path) as image:
                if image.size!=(1920,1080):raise ValueError('Wrong render raster')
            renders.append({'path':str(path),'sha256':row['sha256']})
        verdict_path=folder/'visual_review.json';verdict=json.loads(verdict_path.read_text())
        for key in ('manifest','review_inputs'):
            check_hash(folder/verdict[key]['path'],verdict[key]['sha256'])
        if verdict['accepted_as_repair'] or verdict['visual_status']!='fail':
            raise ValueError('Known failed canary must not be silently promoted')
        review_path=folder/verdict['review_inputs']['path']
        for item in json.loads(review_path.read_text())['images']:check_hash(Path(item['path']),item['sha256'])
        reviews.append({'path':str(verdict_path),'sha256':sha256(verdict_path),'status':verdict['visual_status']})
        metric_path=folder/'face_metrics/metrics.json';score=json.loads(metric_path.read_text())
        for prefix in ('prediction','ground_truth','face_polygons'):
            check_hash(Path(score[prefix]),score[prefix+'_sha256'])
        for key in ('face_psnr','face_ssim','face_lpips'):
            if not math.isfinite(score[key]):raise ValueError('Nonfinite face metric')
        if any(key in score for key in ('psnr','ssim','lpips','loss','full_frame_psnr','actor_psnr')):
            raise ValueError('Unexpected non-face metric')
        metrics.append({'path':str(metric_path),'sha256':sha256(metric_path),
                        **{k:score[k] for k in ('face_psnr','face_ssim','face_lpips')}})
        if (folder/'textured').exists():
            retained=json.loads((folder/'result.json').read_text())
            for relative,digest in retained['artifacts'].items():check_hash(folder/relative,digest)
    report={'timestamp':datetime.now(timezone.utc).isoformat(),'audit_status':'pass',
            'repair_goal_complete':False,'temporal_flythrough_gate_passed':False,
            'fits':fits,'renders':renders,'visual_reviews':reviews,'face_metrics':metrics,
            'code_sha256':sha256(Path(__file__))}
    atomic_json(root/'color_canary_audit.json',report)
    checks={'timestamp':report['timestamp'],'stage':'color_canaries_audited_no_promotion',
            'audit_sha256':sha256(root/'color_canary_audit.json')}
    for name,command in {
        'workers':['pgrep','-af','[r]ender_patchmatch_camera_path.py|[t]exture_patchmatch_mesh_mvs.py|[c]alibrate_patchmatch_camera_colors.py'],
        'gpu':['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],
        'disk':['df','-Pk',str(root),'/home/brans']}.items():
        result=subprocess.run(command,capture_output=True,text=True,timeout=20)
        checks[name]={'returncode':result.returncode,'text':result.stdout+result.stderr}
    with (root/'campaign_checks.jsonl').open('a') as out:out.write(json.dumps(checks,sort_keys=True)+'\n')
    print(json.dumps({'audit_status':'pass','fits':len(fits),'renders':len(renders),'failed_canaries':len(reviews),
                      'repair_goal_complete':False}))


if __name__=='__main__':main()
