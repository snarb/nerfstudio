#!/usr/bin/env python3
"""Matched static-time head slide; frozen fields, train-only sources, gray fallback."""
from pathlib import Path
import argparse,subprocess,sys,time,os
from run_dec5_sequence import read,write,identity
from run_dec5_opacity_sweep import threshold
from dec5_sequence_worker import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--frame-root',type=Path,required=True)
    p.add_argument('--field-data',type=Path)
    p.add_argument('--expected-frames',type=int,choices=[16,96],default=16)
    a=p.parse_args();r=a.root;f=a.frame_root;scripts=Path(__file__).parent
    if len(read(r/'path.json')['frames'])!=a.expected_frames:raise ValueError('Unexpected frozen path inventory')
    write(r/'worker.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),started=time.time()))
    def run(stage,script,*args):
        write(r/'progress.json',dict(stage=stage,time=time.time()))
        with (r/(stage+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(scripts/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
    for name in ['control','volume']:
        arm=r/name;checkpoint=arm/'inference.pt'
        common=['--data',a.field_data or f/'semantic_data','--training',arm,'--checkpoint',checkpoint,
                '--extra-quantiles','.1','--minimum-opacity','.001','--rays-per-chunk','1024',
                '--vectorized-quantiles','--uint8-cpu-cache']
        for role,manifest,extra in [('source',r/'source8_hd.json',['--source-semantics',f/'semantics','--source-roi-person']),
                                   ('target',r/'path.json',['--parent',r/'roi'])]:
            destination=arm/(role+'_low')
            if not (destination/'complete.json').exists():
                if destination.exists():destination.rename(arm/(role+'_interrupted_'+str(int(time.time()))))
                run(name+'_export_'+role,'export_dec5_native_depth.py',*common,'--camera-manifest',manifest,'--output',destination,*extra)
            if read(destination/'complete.json')['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Wrong checkpoint cache')
            threshold(destination,arm/(role+'_0100'),.1)
        if not (arm/'prediction/complete.json').exists():
            run(name+'_render','render_dec5_native_head_benchmark.py','--source-rgb-data',r/'native_rgb',
                '--source-cameras',r/'source8_hd.json','--source-depth',arm/'source_0100',
                '--semantics',f/'semantics','--target-depth',arm/'target_0100','--fallback',r/'fallback',
                '--cameras',r/'path.json','--source-reference-camera',r/'anchor.json',
                '--depth-key','q1000','--expected-train-count','64','--output',arm/'prediction')
    run('review','review_dec5_hair_path.py','--path',r/'path.json','--baseline',r/'control/prediction',
        '--candidate',r/'volume/prediction','--expected-frames',a.expected_frames,'--output',r/'review')
    run('rgb_proxy','score_dec5_rgb_proxy.py','--path',r/'path.json','--baseline',r/'control/prediction',
        '--candidate',r/'volume/prediction','--output',r/'rgb_proxy.json')
    write(r/'complete.json',dict(static_time_camera_path=True,physical_time_flicker_measured=False,
        frame_count=a.expected_frames,
        heldout_rgb_used=False,neutral_gray_diagnostic=True,review=read(r/'review/metrics.json')))


if __name__=='__main__':main()
