#!/usr/bin/env python3
"""Full96-view confirmation of the opacity-floor pilot, outside silver-sable."""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np
from dec5_hair import cv_pose
from run_dec5_opacity_sweep import threshold
from run_dec5_hair_campaign import atomic_json,sha


def main():
    p=argparse.ArgumentParser()
    for key in ['root','inputs','static','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();n=a.root;out=a.output;out.mkdir(parents=True,exist_ok=True);scripts=Path(__file__).parent
    def run(script,*args):subprocess.run([sys.executable,str(scripts/script),*map(str,args)],check=True)
    cameras=a.inputs/'full96_cameras.json';anchor=a.inputs/'full96_source_anchor.json'
    payload=json.loads((n/'crop_native/transforms.json').read_text());train=[f for f in payload['frames'] if f['file_path'] in payload['train_filenames']]
    center=cv_pose(json.loads(anchor.read_text()))[:3,3]
    expected=[train[i]['physical_camera'] for i in np.argsort([np.linalg.norm(cv_pose(f)[:3,3]-center) for f in train],kind='stable')[:8]]
    available={f['physical_camera'] for f in json.loads((n/'source_hd_cameras.json').read_text())['frames']}
    if not set(expected)<=available:raise ValueError('Fixed full-path source set is not cached')
    prepared=out/'prepared'
    if not (prepared/'complete.json').exists():
        run('run_dec5_frozen_path_chunk.py','--root',n,'--training',n/'background_separated5k',
            '--checkpoint',n/'background_separated5k/gate_004999.pt','--cameras',cameras,
            '--source-depth',a.static/'source_low','--output',prepared,'--data','crop_native',
            '--depth-keys','q1000','--roi-dir',a.inputs/'roi_full96_model_bound','--minimum-opacity','.001','--prepare-only')
    for label,floor in [('both100',.1),('both010',.01)]:
        target=out/f'target_{floor:g}';source=a.static/f'source_{floor:g}'
        threshold(prepared/'depth',target,floor)
        if not (out/label/'complete.json').exists():
            run('render_dec5_native_head_benchmark.py','--source-rgb-data',n/'crop_native',
                '--source-cameras',n/'source_hd_cameras.json','--source-depth',source,
                '--semantics',n/'semantics_hd','--target-depth',target,'--fallback',prepared/'fallback',
                '--cameras',cameras,'--depth-key','q1000','--source-reference-camera',anchor,'--output',out/label)
    if not (out/'review/metrics.json').exists():
        run('review_dec5_hair_path.py','--path',cameras,'--baseline',out/'both100','--candidate',out/'both010',
            '--expected-frames','96','--output',out/'review')
    if not (out/'rgb_proxy.json').exists():
        run('score_dec5_rgb_proxy.py','--path',cameras,'--baseline',out/'both100','--candidate',out/'both010','--output',out/'rgb_proxy.json')
    atomic_json(out/'complete.json',dict(frame_count=96,checkpoint_sha256=sha(n/'background_separated5k/gate_004999.pt'),
        fixed_source_cameras=expected,minimum_opacity_control=.1,minimum_opacity_candidate=.01,
        same_field_source_rgb_camera_path_and_fallback=True,viewpoint_path_only_not_physical_time=True))


if __name__=='__main__':main()
