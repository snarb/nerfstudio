"""Continue adjacent temporal frames, stopping at explicit quality/visual gates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from prepare_luster_video import write,environment,SCRIPTS
from archive_luster_checkpoint import archive,restore
from luster_checkpoint_retention import prune_dominated


def numeric_pass(selected, final=True):
    views=[v for v in selected['per_view'] if v['split']=='eval' and v.get('foreground_psnr') is not None]
    foreground=sum(v['foreground_psnr'] for v in views)/len(views) if views else 0
    limits=(30.0,.947,.060) if final else (29.0,.935,.085)
    return (selected['eval_all_psnr']>=limits[0] and selected['eval_all_ssim']>=limits[1]
            and selected['eval_all_lpips']<=limits[2] and foreground>=24.0)


def useful_improvement(previous,current):
    return (current['eval_all_psnr']-previous['eval_all_psnr']>=.07
            or previous['eval_all_lpips']-current['eval_all_lpips']>=.001)


def gate_action(history, selected, final_metrics=None, polished=False, at_limit=False):
    if final_metrics is not None:
        if numeric_pass(final_metrics):return 'accept'
        return 'review' if polished or at_limit else 'polish'
    improving=len(history)>=2 and useful_improvement(history[-2],history[-1])
    if improving and not at_limit:return 'continue'
    return 'export' if numeric_pass(selected,final=False) else 'review'


def export_candidate(frame_root,run,selected):
    export=frame_root/f'export_s{selected["step"]:06d}'
    if not (export/'complete.json').exists():
        with (frame_root/'logs'/f'export_s{selected["step"]:06d}.log').open('w') as log:
            subprocess.run([sys.executable,str(SCRIPTS/'export_luster_selection.py'),str(run),'--output',str(export),
                            '--hull-margin-voxels','3'],env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((export/'selection.json').read_text())
    if result['checkpoint']!=selected['checkpoint']:raise ValueError('Candidate export checkpoint differs')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--start',type=int,required=True)
    p.add_argument('--end',type=int,required=True);p.add_argument('--max-step',type=int,default=24000)
    p.add_argument('--remote-root',default='/fsx/tmp/luster/lookcloser_video_000470_000529_20260930')
    args=p.parse_args();root=args.root.resolve()
    if args.max_step<6000 or args.max_step%2000:raise ValueError('Budget must be a multiple of2000, at least6000')
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
        prior_run=Path(parent['run']);last_run=None;end_step=6000;polished=False;lr_override=None
        while True:
            run=frame_root/'runs'/f's{end_step:06d}'
            write(root/'campaign_status.json',dict(phase='training',frame=frame,end_step=end_step,time=time.time()))
            if not (run/'complete.json').exists():
                command=[sys.executable,str(SCRIPTS/'launch_luster_video_stage.py'),str(root),frame,str(end_step),
                         '--reason','Adjacent-frame weights; continue only while quality improves or before the bounded quality gate']
                if last_run:command+=['--resume-run',str(last_run)]
                else:command+=['--warm-parent',str(prior_run),'--eval-steps','4096','6000']
                if lr_override is not None:command+=['--lr-base',str(lr_override),'--eval-steps',str(end_step-2000),str(end_step)]
                subprocess.run(command,env=environment(),check=True)
            history=json.loads((run/'history.json').read_text());selected=json.loads((run/'selection.json').read_text())
            restore(selected['checkpoint'])
            prune_dominated(root,frame_root,history,json.loads((run/'complete.json').read_text())['latest_checkpoint'])
            last_run=run
            action=gate_action(history,selected,polished=polished,at_limit=end_step>=args.max_step)
            # A polish is a bounded experiment; evaluate its final rendering even
            # if its raw metrics still improve, instead of extending it blindly.
            if polished and action=='continue':action='export'
            if action=='export':
                write(root/'campaign_status.json',dict(phase='evaluating_candidate',frame=frame,run=str(run),time=time.time()))
                final=export_candidate(frame_root,run,selected)
                action=gate_action(history,selected,final,polished,end_step+4000>args.max_step)
            with (root/'decisions.jsonl').open('a') as log:
                log.write(json.dumps(dict(time=time.time(),frame=frame,stage=run.name,action=action,
                                         selected_step=selected['step'],polished=polished))+'\n')
            if action=='accept':break
            if action=='review':
                write(root/'campaign_status.json',dict(phase='quality_review_required',frame=frame,run=str(run),
                                                       reason='Poor plateau, failed final export, or bounded budget exhausted',time=time.time()))
                raise SystemExit(2)
            if action=='polish':
                base=json.loads((run/'request.json').read_text())['lr'];lr_override=base/2
                polished=True;end_step+=4000
            else:end_step+=2000
        write(root/'campaign_status.json',dict(phase='exporting',frame=frame,run=str(last_run),time=time.time()))
        subprocess.run([sys.executable,str(SCRIPTS/'finish_luster_video_frame.py'),str(root),frame,str(last_run),'--remote-root',args.remote_root],env=environment(),check=True)
        # The predecessor is no longer needed in GPU startup. Its archive was
        # already byte-verified, and restore() can materialize it for rerenders.
        for retained in parent.get('retained_checkpoints',[parent['archived_checkpoint']]):
            archive(root,retained['local_path'],args.remote_root,release=True)
        write(root/'campaign_status.json',dict(phase='frame_complete_pending_batch_review',frame=frame,time=time.time()))
    write(root/'campaign_status.json',dict(phase='visual_review_required',through=f'{args.end:06d}',time=time.time()))


if __name__=='__main__':main()
