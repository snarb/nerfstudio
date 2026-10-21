"""Sequential, own-directory-only reproduction of v1 across frozen frames."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from luster_hybrid_common import read,write


def run(root,frames,chunk=1024,render_only=False,postprocess='none'):
    assert postprocess in ('none','contour-s2-d1')
    scripts=Path(__file__).resolve().parent
    config=scripts.parent/'experiments/luster_000540_hybrid_v1/selection_config.json'
    state_path=root/('render_state.json' if render_only else 'state.json')
    for frame in frames:
        directory=root/'frames'/f'{frame:06d}'
        context=directory/'context.json'
        assert context.exists()
        frame_lock=(directory/'transfer_pipeline.lock').open('a')
        try:
            fcntl.flock(frame_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another transfer pipeline owns frame {frame}; wait for it to finish')
        env={**os.environ,'LUSTER_HYBRID_FRAME_MANIFEST':str(context),
             'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1',
             'CUDA_VISIBLE_DEVICES':'','CUDA_HOME':str(Path(sys.executable).parents[1])}
        env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env['PATH']
        def command(stage,args,gpu=False):
            if shutil.disk_usage(root).free<768*1024**3:
                raise RuntimeError('Reserve >=768 GiB for the unrelated training campaign')
            current={**env,'CUDA_VISIBLE_DEVICES':'0' if gpu else ''}
            write(state_path,dict(frame=frame,stage=stage,started=time.time(),pid=os.getpid(),
                                        writes_to_training=False))
            print(json.dumps(dict(frame=frame,stage=stage,time=time.time())),flush=True)
            with (root/'logs'/f'{frame:06d}_{stage}.log').open('a') as log:
                result=subprocess.run([sys.executable,str(scripts/args[0]),*args[1:]],
                    env=current,stdout=log,stderr=subprocess.STDOUT,cwd=scripts.parents[1])
            if result.returncode:
                write(state_path,dict(frame=frame,stage=stage,exit_code=result.returncode,
                                            status='stopped_own_pipeline_only',time=time.time()))
                raise RuntimeError(f'{frame}/{stage} failed: {result.returncode}')
        if not (directory/'preflight/manifest.json').exists():
            command('prepare',['luster_hybrid_common.py','--root',str(directory)])
        if not render_only:
            command('protocol',['protocol_multiframe_transfer.py','--root',str(directory)])
        command('render_targets',['guarded_transfer_inference.py','--root',str(directory),
            '--cameras','7','12','95','150','--occupancy-only','--chunk',str(chunk)],gpu=True)
        if not render_only and not (directory/'evaluation/baseline_metrics.json').exists():
            command('baseline_cpu',['evaluate_luster_hybrid.py','--root',str(directory),'--device','cpu'])
        if not (directory/'preflight/sources.json').exists():
            command('sources',['transfer_luster_hybrid.py','--root',str(directory),'sources','--count','16'])
        sources=read(directory/'preflight/sources.json')['all_sources']
        pool=directory/'preflight/witness_pool.json'
        frozen_pool=dict(cameras=sorted(set(sources)-{7,12,95,150}),
                         policy='Selected source union; fixed before confidence construction; extra caches do not change witnesses')
        if pool.exists():assert read(pool)==frozen_pool
        else:write(pool,frozen_pool)
        command('render_sources',['guarded_transfer_inference.py','--root',str(directory),
            '--cameras',*map(str,sources),'--occupancy-only','--chunk',str(chunk)],gpu=True)
        command('warps',['transfer_luster_hybrid.py','--root',str(directory),'warp','--count','16'])
        command('cubic_warps',['cubic75_luster_hybrid.py','--root',str(directory),'--count','16'])
        # Ranking stage is intentionally fresh-name-only if interrupted. Never
        # silently overwrite an already complete per-camera source assignment.
        for cid in [7,12,95,150]:
            if not (directory/'candidates/v1_rank'/f'cam{cid:03d}_receipt.json').exists():
                command(f'rank_{cid:03d}',['patch_eligibility_rank_luster_hybrid.py',
                    '--root',str(directory),'--name','v1_rank','--config',str(config),'--cameras',str(cid)])
        command('resample',['cubic75_once_luster_hybrid.py','--root',str(directory),
                           '--parent','v1_rank','--name','v1_cubic'])
        command('merge',['fixed_sampling_seam_luster_hybrid.py','--root',str(directory),
            '--parent','v1_cubic','--name','v1','--filter-support','geometry','--seam-fade','3',
            '--adaptive-seam','--seam-tolerance','.02'])
        candidate='v1'
        if postprocess=='contour-s2-d1':
            candidate='contour_s2_d1'
            command('contour',['contour_guard_luster_hybrid.py','--root',str(directory),
                '--parent','v1','--name',candidate,'--sigma','2','--displacement','1'])
        write(directory/f'{candidate}_render_complete.json',dict(time=time.time(),
            candidate=candidate,cameras=[7,12,95,150],resolution='native',
            context=str(context),evaluation_performed=False,visual_acceptance='not_implied',
            outputs={str(c):str(directory/f'candidates/{candidate}/cam{c:03d}.png')
                     for c in [7,12,95,150]}))
        if render_only:
            frame_lock.close()
            continue
        if not (directory/f'evaluation/{candidate}_metrics.json').exists():
            command('hybrid_cpu',['evaluate_luster_hybrid.py','--root',str(directory),
                                 '--candidate',candidate,'--device','cpu'])
        command('review',['review_single_luster_hybrid.py','--root',str(directory),'--name',candidate])
        command('summary',['summarize_multiframe_transfer.py','--root',str(root),'--candidate',candidate])
        write(directory/f'{candidate}_complete.json',dict(time=time.time(),visual_acceptance='pending'))
        frame_lock.close()
    write(state_path,dict(status='requested_frames_complete',frames=frames,time=time.time(),
                         render_only=render_only,postprocess=postprocess))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--frames',nargs='+',type=int,required=True)
    p.add_argument('--render-only',action='store_true',help='Skip evaluation masks, GT scoring and comparison panels')
    p.add_argument('--postprocess',choices=['none','contour-s2-d1'],default='none',
                   help='Optional prediction-only contour protection; no per-frame GT-based switching')
    p.add_argument('--chunk',type=int,choices=[256,512,1024,2048,4096],default=1024);a=p.parse_args()
    run(a.root,a.frames,a.chunk,a.render_only,a.postprocess)
