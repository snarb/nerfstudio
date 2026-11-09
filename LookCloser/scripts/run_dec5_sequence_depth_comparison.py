#!/usr/bin/env python3
"""Matched64+1 field geometry comparison, native6K GT transfer to held-out4K."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,subprocess,sys,time
from run_dec5_sequence import read,write
from run_dec5_opacity_sweep import threshold
from dec5_sequence_worker import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--training',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--field-data',type=Path,help='Explicit training dataset for native6K fields; source RGB protocol stays fixed')
    p.add_argument('--name',default='full_recipe');a=p.parse_args();r=a.root
    if bool(a.training)!=bool(a.checkpoint):p.error('Explicit training and checkpoint must be supplied together')
    if a.field_data and not a.training:p.error('Explicit field data requires one explicit training/checkpoint pair')
    if not a.name.replace('_','').isalnum():p.error('Benchmark name must be alphanumeric with underscores')
    while not (r/'models_copied.json').exists():time.sleep(10)
    scripts=Path(__file__).parent;data=a.field_data or r/'frame/semantic_data'
    def arm(name,training=None,checkpoint=None):
        out=r/'geometry'/name;out.mkdir(parents=True,exist_ok=True)
        training=training or r/'models'/('corrected_'+name)/'frames/000901'/('training_corrected_'+name)
        checkpoint=checkpoint or training/'step-000020188.ckpt'
        request=dict(training=str(training),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            source_manifest_sha256=sha(r/'source8_hd.json'),target_manifest_sha256=sha(r/'target4k.json'),
            fallback_receipt_sha256=sha(r/'fallback4k/complete.json'),
            source_rgb_dataset_sha256=sha(r/'native_rgb/data/transforms.json'),minimum_opacity=.1,depth_key='q1000')
        if a.field_data:request['field_data_metadata_sha256']=sha(data/'transforms.json')
        previous=read(out/'request.json')
        if previous is not None and previous!=request:raise ValueError('Immutable geometry benchmark inputs changed')
        write(out/'request.json',request)
        def run(stage,script,*args):
            write(out/'progress.json',dict(phase=stage,time=time.time()))
            with (out/(stage+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(scripts/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
        common=['--data',data,'--training',training,'--checkpoint',checkpoint,'--extra-quantiles','.1',
            '--minimum-opacity','.001','--rays-per-chunk','1024','--vectorized-quantiles','--uint8-cpu-cache']
        for role,manifest,extra in [('source',r/'source8_hd.json',['--source-semantics',r/'frame/semantics','--source-roi-person']),
                                  ('target',r/'target4k.json',['--parent',r/'roi4k'])]:
            destination=out/(role+'_low')
            if not (destination/'complete.json').exists():
                if destination.exists():destination.rename(out/(role+'_interrupted_'+str(int(time.time()))))
                run('export_'+role,'export_dec5_native_depth.py',*common,'--camera-manifest',manifest,'--output',destination,*extra)
            threshold(destination,out/(role+'_0100'),.1)
            if read(destination/'complete.json')['checkpoint_sha256']!=request['checkpoint_sha256']:
                raise ValueError('Cached depth comes from a different checkpoint')
        prediction=out/'prediction'
        if not (prediction/'complete.json').exists():
            if prediction.exists():prediction.rename(out/('prediction_interrupted_'+str(int(time.time()))))
            run('render','render_dec5_native_head_benchmark.py','--source-rgb-data',r/'native_rgb/data',
                '--source-cameras',r/'source8_hd.json','--source-depth',out/'source_0100','--semantics',r/'frame/semantics',
                '--target-depth',out/'target_0100','--fallback',r/'fallback4k','--cameras',r/'target4k.json',
                '--depth-key','q1000','--expected-train-count','64','--output',prediction)
        metrics=out/'metrics.json'
        if not metrics.exists():run('score','score_dec5_hair.py','--root',r/'benchmark_scene','--prediction',prediction,
            '--output',metrics,'--boundary-radius','5','--include-person-roi')
        result=dict(arm=name,metrics=read(metrics)['aggregate']);write(out/'complete.json',result);return result
    if a.training:results=[arm(a.name,a.training,a.checkpoint)]
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(arm,['control','hair_mix']))
    result_path=r/('depth_comparison_'+a.name+'.json' if a.training else 'depth_comparison.json')
    write(result_path,dict(results=results,heldout_cameras=1,native_source_rgb=[6144,3072],
        target_rgb=[3840,1920],field_training_resolution=([read(data/'transforms.json')['w'],read(data/'transforms.json')['h']] if a.field_data else [1920,960]),source_camera_count=8,
        minimum_opacity=.1,depth_quantile=.1,same_renderer_sources_fallback_and_roi=True,
        no_depth_ground_truth=True,physical_time_stability_established=False))


if __name__=='__main__':main()
