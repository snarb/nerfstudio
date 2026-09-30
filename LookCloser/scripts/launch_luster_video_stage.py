"""Run one bounded temporal stage using the existing Trainer transfer contract."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from prepare_luster_video import write, environment, SCRIPTS, REFERENCE
from archive_luster_checkpoint import restore


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('frame');p.add_argument('end_step',type=int)
    p.add_argument('--resume-run',type=Path);p.add_argument('--warm-parent',type=Path)
    p.add_argument('--lr-base',type=float);p.add_argument('--initial-lr',type=float)
    p.add_argument('--fr',type=float);p.add_argument('--eval-steps',nargs='*',type=int)
    p.add_argument('--label');p.add_argument('--reason',required=True);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();root=args.root.resolve();frame_root=root/'frames'/args.frame;data=frame_root/'data'
    if not (root/'preparation_complete.json').exists():raise ValueError('Freeze and audit sequence bounds before training')
    if args.resume_run and args.warm_parent:raise ValueError('Choose continuation within a frame or transfer to a new frame')
    template=REFERENCE/'runs/exp_sh_fasfix_fr03_lr002_40000/request.json'
    if args.resume_run:
        request=json.loads((args.resume_run/'request.json').read_text())
        if Path(request['data']).resolve()!=data:raise ValueError('Full-state resume cannot cross frames')
        previous=json.loads((args.resume_run/'complete.json').read_text())
        if args.end_step<=previous['step']:raise ValueError('Continuation must extend the local step')
        for key in ['warm_start','warm_start_request']:request.pop(key,None)
        request.update(resume=str(restore(previous['latest_checkpoint'])),parent_history=str(args.resume_run/'history.json'))
    else:
        source=json.loads(template.read_text())
        request={k:deepcopy(source[k]) for k in ['seed','eval_stride','train_review_indices','model','background_mask_margin']}
        request['rois_by_image']=json.loads((data/'video_rois.json').read_text())
        request['data']=str(data)
        request['lr']=args.initial_lr or (.002 if args.warm_parent else .01)
        request['lr_final']=.0001;request['lr_steps']=200000
        if args.warm_parent:
            parent_request=json.loads((args.warm_parent/'request.json').read_text())
            parent_frame=Path(parent_request['data']).parent.name
            if int(parent_frame)+1!=int(args.frame):raise ValueError('Warm parent must be the previous real frame')
            selected=json.loads((args.warm_parent/'selection.json').read_text())
            request.update(warm_start=str(restore(selected['checkpoint'])),warm_start_request=str(args.warm_parent/'request.json'))
    if args.lr_base is not None:
        if not args.resume_run or args.lr_base<=0:raise ValueError('LR override requires a same-frame resume')
        request['resume_fields_lr_override']=args.lr_base
    if args.fr is not None:request['model']['feature_reweighting_strength']=args.fr
    overrides=json.loads((frame_root/'source/mask_overrides.json').read_text()) if (frame_root/'source/mask_overrides.json').exists() else {}
    request['background_mask_exclude_cameras']=sorted({164,*[int(cid) for cid in overrides]})
    label=args.label or f's{args.end_step:06d}'
    if Path(label).name!=label:raise ValueError('Stage label must be one path component')
    output=frame_root/'runs'/label
    request.update(output=str(output),end_step=args.end_step,eval_steps=args.eval_steps or [args.end_step])
    if args.dry_run:print(json.dumps(request,indent=2));return
    if output.exists():raise ValueError('Stage exists; use a new explicit continuation')
    # Source, calibration, per-image frequency receipts and the ragged map
    # contract are checked before any expensive field update.
    for script,arguments in [('audit_luster_data.py',[str(frame_root),'--require-frequencies']),('audit_luster_sampling.py',[str(frame_root)])]:
        with (frame_root/'logs'/f'{label}_{script}.log').open('w') as log:
            subprocess.run([sys.executable,str(SCRIPTS/script),*arguments],env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    request_path=frame_root/'requests'/f'{label}.json';write(request_path,request)
    logdir=frame_root/'logs'/label;job_path=frame_root/'requests'/f'{label}_job.json'
    write(job_path,dict(log_dir=str(logdir),command=[sys.executable,str(SCRIPTS/'run_luster_experiment.py'),str(request_path)],
                       progress=str(output/'progress.json'),complete=str(output/'complete.json')))
    with (root/'decisions.jsonl').open('a') as f:f.write(json.dumps(dict(frame=args.frame,stage=label,reason=args.reason,request=str(request_path)))+'\n')
    subprocess.run([sys.executable,str(SCRIPTS/'supervise_luster_job.py'),str(job_path)],env=environment(),check=True)


if __name__=='__main__':main()
