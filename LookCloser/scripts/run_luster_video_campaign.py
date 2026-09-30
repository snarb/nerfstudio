"""Continue adjacent temporal frames, stopping at explicit quality/visual gates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from prepare_luster_video import write,environment,SCRIPTS
from archive_luster_checkpoint import archive,restore


def numeric_pass(selected):
    views=[v for v in selected['per_view'] if v['split']=='eval' and v.get('foreground_psnr') is not None]
    foreground=sum(v['foreground_psnr'] for v in views)/len(views) if views else 0
    return (selected['eval_all_psnr']>=30.0 and selected['eval_all_ssim']>=.947
            and selected['eval_all_lpips']<=.060 and foreground>=24.0)


def useful_improvement(previous,current):
    return (current['eval_all_psnr']-previous['eval_all_psnr']>=.07
            or previous['eval_all_lpips']-current['eval_all_lpips']>=.001)


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--start',type=int,required=True)
    p.add_argument('--end',type=int,required=True);p.add_argument('--max-step',type=int,default=12000)
    p.add_argument('--remote-root',default='/fsx/tmp/luster/lookcloser_video_000470_000529_20260930')
    args=p.parse_args();root=args.root.resolve()
    previous_frame=f'{args.start-1:06d}'
    review=root/'visual_reviews'/f'{previous_frame}.json'
    if not review.exists() or not json.loads(review.read_text()).get('accepted'):
        raise ValueError('Review the preceding batch before starting another temporal batch')
    for number in range(args.start,args.end+1):
        frame=f'{number:06d}';frame_root=root/'frames'/frame;data=frame_root/'data'
        while not (data/'frequency_complete.json').exists():
            write(root/'campaign_status.json',dict(phase='waiting_for_frequencies',frame=frame,time=time.time()))
            time.sleep(30)
        parent=json.loads((root/'snapshots'/f'{number-1:06d}.json').read_text())
        prior_run=Path(parent['run']);last_run=None
        for end_step in range(6000,args.max_step+1,2000):
            run=frame_root/'runs'/f's{end_step:06d}'
            write(root/'campaign_status.json',dict(phase='training',frame=frame,end_step=end_step,time=time.time()))
            if not (run/'complete.json').exists():
                command=[sys.executable,str(SCRIPTS/'launch_luster_video_stage.py'),str(root),frame,str(end_step),
                         '--reason','Adjacent-frame weights; continue only while quality improves or before the bounded quality gate']
                if last_run:command+=['--resume-run',str(last_run)]
                else:command+=['--warm-parent',str(prior_run),'--eval-steps','4096','6000']
                subprocess.run(command,env=environment(),check=True)
            history=json.loads((run/'history.json').read_text());selected=json.loads((run/'selection.json').read_text())
            restore(selected['checkpoint'])
            keep={selected['checkpoint'],json.loads((run/'complete.json').read_text())['latest_checkpoint']}
            for checkpoint in (frame_root/'runs').glob('*/trainer/lookcloser/seed42/nerfstudio_models/*.ckpt'):
                archive(root,checkpoint,args.remote_root,release=str(checkpoint) not in keep)
            last_run=run
            if numeric_pass(selected) and not useful_improvement(history[-2],history[-1]):break
            if len(history)>=2 and not numeric_pass(selected) and not useful_improvement(history[-2],history[-1]):
                write(root/'campaign_status.json',dict(phase='quality_review_required',frame=frame,run=str(run),reason='Quality below guardrails and no useful recent improvement'))
                raise SystemExit(2)
        if not numeric_pass(selected):
            write(root/'campaign_status.json',dict(phase='quality_review_required',frame=frame,run=str(last_run),reason='Bounded temporal budget reached below guardrails'))
            raise SystemExit(2)
        write(root/'campaign_status.json',dict(phase='exporting',frame=frame,run=str(last_run),time=time.time()))
        subprocess.run([sys.executable,str(SCRIPTS/'finish_luster_video_frame.py'),str(root),frame,str(last_run),'--remote-root',args.remote_root],env=environment(),check=True)
        # The predecessor is no longer needed in GPU startup. Its archive was
        # already byte-verified, and restore() can materialize it for rerenders.
        archive(root,parent['archived_checkpoint']['local_path'],args.remote_root,release=True)
        write(root/'campaign_status.json',dict(phase='frame_complete_pending_batch_review',frame=frame,time=time.time()))
    write(root/'campaign_status.json',dict(phase='visual_review_required',through=f'{args.end:06d}',time=time.time()))


if __name__=='__main__':main()
