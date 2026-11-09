#!/usr/bin/env python3
"""Frozen-field source/target opacity ablation; one low-floor export per domain."""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np
from dec5_hair import cv_pose
from run_dec5_hair_campaign import atomic_json,sha


def threshold(source,destination,floor):
    if (destination/'complete.json').exists():return
    destination.mkdir(parents=True,exist_ok=True);receipt=json.loads((source/'complete.json').read_text());records=[]
    for record in receipt['records']:
        path=source/(record['camera']+'.npz')
        if sha(path)!=record['sha256']:raise ValueError('Changed low-floor cache')
        with np.load(path) as archive:arrays={k:archive[k] for k in archive.files}
        valid=arrays['opacity']>=floor
        arrays['expected']=np.where(valid,arrays['expected'],0)
        for key in list(arrays):
            if key=='median' or (len(key)==5 and key.startswith('q') and key[1:].isdigit()):
                arrays[key]=np.where(valid&(arrays['weight_mass']>=floor),arrays[key],0)
        target=destination/path.name;np.savez_compressed(target,**arrays)
        records.append(dict(record,sha256=sha(target)))
    receipt.update(records=records,minimum_opacity=floor,low_floor_parent_sha256=sha(source/'complete.json'),
                   quantile_mass_gate='same FP64 sum as the reference CDF',field_requeried=False)
    atomic_json(destination/'complete.json',receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','inputs','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();n=args.root;out=args.output;out.mkdir(parents=True,exist_ok=True);scripts=Path(__file__).parent
    def run(script,*argv):subprocess.run([sys.executable,str(scripts/script),*map(str,argv)],check=True)
    anchor=args.inputs/'fg_q10_anchored_counterfactual/source_anchor.json';cameras=args.inputs/'pilot_cameras.json'
    frames=json.loads((n/'source_hd_cameras.json').read_text())['frames'];center=cv_pose(json.loads(anchor.read_text()))[:3,3]
    order=np.argsort([np.linalg.norm(cv_pose(f)[:3,3]-center) for f in frames],kind='stable')[:8]
    source_manifest=out/'source8.json';atomic_json(source_manifest,dict(frames=[frames[i] for i in order]))
    for role,manifest,extra in [('source',source_manifest,['--source-semantics',n/'semantics_hd']),
                              ('target',cameras,['--parent',args.inputs/'roi_pilot_hull_v2'])]:
        destination=out/(role+'_low')
        if not (destination/'complete.json').exists():
            run('export_dec5_native_depth.py','--data',n/'crop_native','--training',n/'background_separated5k',
                '--checkpoint',n/'background_separated5k/gate_004999.pt','--output',destination,
                '--camera-manifest',manifest,'--extra-quantiles','.1','--minimum-opacity','.001',
                '--uint8-cpu-cache',*extra)
    arms=[('both001',.001,.001),('both010',.01,.01),('both030',.03,.03),('both100',.1,.1),
          ('both200',.2,.2),('source010',.01,.1),('target010',.1,.01)]
    for label,source_floor,target_floor in arms:
        for role,floor in [('source',source_floor),('target',target_floor)]:threshold(out/(role+'_low'),out/f'{role}_{floor:g}',floor)
        destination=out/label
        if not (destination/'complete.json').exists():
            run('render_dec5_native_head_benchmark.py','--source-rgb-data',n/'crop_native',
                '--source-cameras',source_manifest,'--source-depth',out/f'source_{source_floor:g}',
                '--semantics',n/'semantics_hd','--target-depth',out/f'target_{target_floor:g}',
                '--fallback',args.inputs/'path_fg_front_v2_0/fallback','--cameras',cameras,
                '--depth-key','q1000','--source-reference-camera',anchor,'--output',destination)
        atomic_json(out/'progress.json',dict(phase='render',arm=label))
    for label,source_floor,target_floor in arms:
        if label=='both100':continue
        review=out/(label+'_review')
        if not (review/'metrics.json').exists():
            run('review_dec5_hair_path.py','--path',cameras,'--baseline',out/'both100',
                '--candidate',out/label,'--expected-frames','16','--output',review)
        rgb=out/(label+'_rgb.json')
        if not rgb.exists():run('score_dec5_rgb_proxy.py','--path',cameras,'--baseline',out/'both100','--candidate',out/label,'--output',rgb)
    atomic_json(out/'complete.json',dict(arms=arms,checkpoint_sha256=sha(n/'background_separated5k/gate_004999.pt'),
        same_field_rgb_sources_fallback_and_source_order=True,source_and_target_floor_isolated=True,
        opacity_is_not_semantic_alpha=True,uses_validation_rgb=False,uses_test_rgb=False))


if __name__=='__main__':main()
