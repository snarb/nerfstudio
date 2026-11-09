#!/usr/bin/env python3
"""Same two frozen foreground geometries, one common new background fallback.

The background-only HD prediction is explicitly upsampled; native source RGB
and cached4K foreground queries retain their actual resolutions.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,subprocess,sys
import cv2
import numpy as np
from run_dec5_sequence import read,write
from dec5_sequence_worker import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    source=r/'neural_background/evaluation_019999';receipt=read(source/'prediction_complete.json')
    prediction=source/'prediction.npz'
    if receipt['uses_validation_rgb'] or receipt['uses_test_rgb'] or sha(prediction)!=receipt['prediction_sha256']:
        raise ValueError('Unverified background prediction')
    fallback=r/'fallback4k_volume_upsampled';fallback.mkdir(exist_ok=True)
    request=dict(background_prediction_sha256=sha(prediction),camera_manifest_sha256=sha(r/'target4k.json'))
    previous=read(fallback/'request.json')
    if previous is not None and previous!=request:raise ValueError('Immutable common fallback changed')
    write(fallback/'request.json',request)
    if not (fallback/'complete.json').exists():
        frames=read(r/'target4k.json')['frames']
        if len(frames)!=1:raise ValueError('Single held-out component ablation')
        f=frames[0];name=f['physical_camera'];roi=np.load(r/'roi4k'/(name+'.npz'))['roi']
        with np.load(prediction) as saved:image=cv2.resize(saved['rgb_display'],(f['w'],f['h']),interpolation=cv2.INTER_LINEAR)
        output=fallback/(name+'.npz');np.savez_compressed(output,rgb_display=image,roi=roi)
        write(fallback/'complete.json',dict(records=[dict(camera=name,sha256=sha(output))],
            camera_manifest_sha256=sha(r/'target4k.json'),checkpoint_sha256=receipt['checkpoint_sha256'],
            method='same frozen background-only volume hybrid; HD1920x960 to4K3840x1920 bilinear',
            native_background_detail_claim=False,uses_validation_rgb=False,uses_test_rgb=False))
    scripts=Path(__file__).parent
    def arm(name):
        root=r/'geometry'/name;prediction=root/'prediction_nn_fallback';metrics=root/'metrics_nn_fallback.json'
        def run(label,script,*args):
            with (root/(label+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(scripts/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
        if not (prediction/'complete.json').exists():
            if prediction.exists():raise RuntimeError('Inspect incomplete common-fallback rendering')
            run('render_nn_fallback','render_dec5_native_head_benchmark.py',
                '--source-rgb-data',r/'native_rgb/data','--source-cameras',r/'source8_hd.json',
                '--source-depth',root/'source_0100','--semantics',r/'frame/semantics',
                '--target-depth',root/'target_0100','--fallback',fallback,'--cameras',r/'target4k.json',
                '--depth-key','q1000','--expected-train-count','64','--output',prediction)
        if not metrics.exists():
            run('score_nn_fallback','score_dec5_hair.py','--root',r/'benchmark_scene',
                '--prediction',prediction,'--output',metrics,'--boundary-radius','5','--include-person-roi')
        return dict(arm=name,metrics=read(metrics)['aggregate'])
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(arm,['background_control25k','background_volume25k']))
    write(r/'common_volume_fallback_comparison.json',dict(results=results,
        source_rgb_resolution=[6144,3072],target_resolution=[3840,1920],
        common_background_evaluation_resolution=[1920,960],background_upsampling_explicit=True,
        unchanged_foreground_depths_and_source_selection=True,physical_time_stability_established=False))


if __name__=='__main__':main()
