#!/usr/bin/env python3
"""Test foreground NeRF RGB fallback against pure background on GT-warp holes.

Geometry, thresholds, source order, source IDs and supported GT colors must be
identical. The appearance metric includes transitions into unsupported pixels;
the older RGB-continuity metric deliberately excluded those transitions.
"""
from pathlib import Path
import argparse,json,subprocess,sys
import cv2,numpy as np
from dec5_hair import unproject,project
from review_dec5_hair_path import remap
from run_dec5_hair_campaign import atomic_json,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','inputs','sweep','output']:parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();n=a.root;out=a.output;out.mkdir(parents=True,exist_ok=True);scripts=Path(__file__).parent
    def run(script,*args):subprocess.run([sys.executable,str(scripts/script),*map(str,args)],check=True)
    cameras=a.inputs/'pilot_cameras.json';anchor=a.inputs/'fg_q10_anchored_counterfactual/source_anchor.json'
    rgb=out/'field_rgb'
    if not (rgb/'complete.json').exists():
        run('export_dec5_native_depth.py','--data',n/'crop_native','--training',n/'background_separated5k',
            '--checkpoint',n/'background_separated5k/gate_004999.pt','--output',rgb,
            '--camera-manifest',cameras,'--parent',a.inputs/'roi_pilot_hull_v2',
            '--extra-quantiles','.1','--include-rgb','--uint8-cpu-cache')
    fallback=out/'neural_fallback';fallback.mkdir(exist_ok=True)
    receipt=json.loads((a.inputs/'path_fg_front_v2_0/fallback/complete.json').read_text());records=[]
    for record in receipt['records']:
        name=record['camera'];value=dict(np.load(rgb/(name+'.npz')))
        base=dict(np.load(a.inputs/'path_fg_front_v2_0/fallback'/(name+'.npz')))
        np.testing.assert_array_equal(base['roi'],value['roi'])
        np.testing.assert_array_equal(value['q1000'],np.load(a.sweep/'target_0.1'/(name+'.npz'))['q1000'])
        base['rgb_display']=value['rgb_display'];target=fallback/(name+'.npz');np.savez_compressed(target,**base)
        records.append(dict(record,sha256=sha(target)))
    receipt.update(records=records,kind='premultiplied foreground NeRF + frozen background; no GT fitting',
        foreground_checkpoint_sha256=sha(n/'background_separated5k/gate_004999.pt'),
        field_export_sha256=sha(rgb/'complete.json'))
    atomic_json(fallback/'complete.json',receipt)
    prediction=out/'prediction'
    if not (prediction/'complete.json').exists():
        run('render_dec5_native_head_benchmark.py','--source-rgb-data',n/'crop_native',
            '--source-cameras',a.sweep/'source8.json','--source-depth',a.sweep/'source_0.1',
            '--semantics',n/'semantics_hd','--target-depth',a.sweep/'target_0.1',
            '--fallback',fallback,'--cameras',cameras,'--depth-key','q1000',
            '--source-reference-camera',anchor,'--output',prediction)
    frames=json.loads(cameras.read_text())['frames'];previous=None;totals={};checks=[]
    for frame in frames:
        name=frame['physical_camera'];base=dict(np.load(a.sweep/'both100'/(name+'.npz')));candidate=dict(np.load(prediction/(name+'.npz')))
        for key in ['source_id','front_depth','roi','hair_probability','support']:
            np.testing.assert_array_equal(base[key],candidate[key])
        supported=base['roi']&(base['source_id']>=0)
        np.testing.assert_array_equal(base['rgb'][supported],candidate['rgb'][supported])
        images=[cv2.imread(str(p/(name+'.png'))).astype(np.float32)/255 for p in [a.sweep/'both100',prediction]]
        if previous:
            old,old_images,camera=previous
            domain=old['roi']&(old['hair_probability']>.5)&(old['source_id']>=0)&(old['front_depth']>0)
            y,x=np.nonzero(domain);depth=old['front_depth'][y,x]
            xyz=unproject(np.stack([x+.5,y+.5],-1),depth,camera);uv,z=project(xyz,frame)
            following=remap(base['front_depth'],uv,True)
            tolerance=np.maximum(2*z/min(frame['fl_x'],frame['fl_y']),.002)
            valid=(z>0)&(uv[:,0]>=.5)&(uv[:,0]<=frame['w']-.5)&(uv[:,1]>=.5)&(uv[:,1]<=frame['h']-.5)&~((following>0)&(following<z-tolerance))
            missing=remap(base['source_id'],uv,True)<0
            for region,mask in [('all_anchored_hair',valid),('transitions_to_fallback',valid&missing)]:
                if region not in totals:totals[region]=dict(pixels=0,baseline_abs_rgb_sum=0.,neural_abs_rgb_sum=0.)
                totals[region]['pixels']+=int(mask.sum())
                for label,old_image,image in zip(['baseline','neural'],old_images,images):
                    error=np.abs(old_image[y,x]-remap(image,uv)).mean(-1)
                    totals[region][label+'_abs_rgb_sum']+=float(error[mask].sum())
        previous=(base,images,frame);checks.append(name)
    for value in totals.values():
        for label in ['baseline','neural']:value[label+'_mae']=value[label+'_abs_rgb_sum']/max(value['pixels'],1)
    atomic_json(out/'metrics.json',dict(regions=totals,identical_source_ids_geometry_supported_rgb=True,
        cameras=checks,metric='baseline-depth transported RGB error including fallback transitions; proxy, not novel-view GT',
        no_geometry_improvement_claim=True,uses_validation_rgb=False,uses_test_rgb=False))
    review=out/'review'
    if not (review/'metrics.json').exists():
        run('review_dec5_hair_path.py','--path',cameras,'--baseline',a.sweep/'both100',
            '--candidate',prediction,'--expected-frames','16','--output',review)
    atomic_json(out/'complete.json',dict(metrics_sha256=sha(out/'metrics.json'),
        source_ids_geometry_and_supported_gt_rgb_identical=True,foreground_rgb_fallback_only=True))


if __name__=='__main__':main()
