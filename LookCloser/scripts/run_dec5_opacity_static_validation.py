#!/usr/bin/env python3
"""Independent UHD held-out check of the opacity-floor path pilot.

All three arms share a freshly exported frozen field, source cameras and RGB,
repaired receiver ROI, source selection and background. No held-out fitting.
"""
from pathlib import Path
import argparse,json,subprocess,sys
from run_dec5_opacity_sweep import threshold
from run_dec5_hair_campaign import atomic_json,sha


def main():
    p=argparse.ArgumentParser()
    for key in ['root','inputs','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();n=a.root;out=a.output;out.mkdir(parents=True,exist_ok=True);scripts=Path(__file__).parent
    def run(script,*args):subprocess.run([sys.executable,str(scripts/script),*map(str,args)],check=True)
    common=['--data',n/'crop_native','--training',n/'background_separated5k',
            '--checkpoint',n/'background_separated5k/gate_004999.pt','--extra-quantiles','.1',
            '--minimum-opacity','.001','--uint8-cpu-cache']
    source=out/'source_low';target=out/'target_low'
    if not (source/'complete.json').exists():run('export_dec5_native_depth.py',*common,
        '--source-semantics',n/'semantics_hd','--camera-manifest',n/'source_hd_cameras.json','--output',source)
    if not (target/'complete.json').exists():run('export_dec5_native_depth.py',*common,
        '--parent',a.inputs/'roi_4k_hull_v2','--camera-manifest',n/'validation_4k_cameras.json','--output',target)
    results={}
    for label,s,t in [('both100',.1,.1),('both010',.01,.01),('target010',.1,.01)]:
        threshold(source,out/f'source_{s:g}',s);threshold(target,out/f'target_{t:g}',t)
        prediction=out/label;metrics=out/(label+'_metrics.json')
        if not (prediction/'complete.json').exists():
            run('render_dec5_native_head_benchmark.py','--source-rgb-data',n/'crop_native',
                '--source-cameras',n/'source_hd_cameras.json','--source-depth',out/f'source_{s:g}',
                '--semantics',n/'semantics_hd','--target-depth',out/f'target_{t:g}',
                '--fallback',a.inputs/'ray_bg_fallback_uhd_hull_v2','--cameras',n/'validation_4k_cameras.json',
                '--depth-key','q1000','--output',prediction)
        if not metrics.exists():run('score_dec5_hair.py','--root',n/'benchmark_scene',
            '--prediction',prediction,'--output',metrics,'--boundary-radius','5')
        results[label]=json.loads(metrics.read_text())['aggregate']
        atomic_json(out/'progress.json',dict(completed=len(results),total=3,arm=label))
    atomic_json(out/'complete.json',dict(results=results,checkpoint_sha256=sha(n/'background_separated5k/gate_004999.pt'),
        same_renderer_rgb_sources_and_fallback=True,uses_validation_rgb_for_metrics_only=True,
        uses_test_rgb=False,no_independent_depth_gt=True))


if __name__=='__main__':main()
