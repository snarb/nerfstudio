"""Export, render and archive one selected temporal model without hiding raw metrics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from prepare_luster_video import write,environment,SCRIPTS
from archive_luster_checkpoint import archive,restore


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('frame');p.add_argument('run',type=Path)
    p.add_argument('--remote-root',default='/fsx/tmp/luster/lookcloser_video_000470_000529_20260930');args=p.parse_args()
    root=args.root.resolve();frame_root=root/'frames'/args.frame
    selected=json.loads((args.run/'selection.json').read_text());restore(selected['checkpoint'])
    export=frame_root/f'export_s{selected["step"]:06d}'
    if not (export/'complete.json').exists():
        with (frame_root/'logs'/f'export_s{selected["step"]:06d}.log').open('w') as log:
            subprocess.run([sys.executable,str(SCRIPTS/'export_luster_selection.py'),str(args.run),'--output',str(export),'--hull-margin-voxels','3'],env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    final=json.loads((export/'selection.json').read_text())
    if final['checkpoint']!=selected['checkpoint']:raise ValueError('Export belongs to another selected checkpoint')
    receipt=root/'video_frames/receipts'/f'{args.frame}.json'
    if not receipt.exists():
        with (frame_root/'logs'/'render_temporal.log').open('w') as log:
            subprocess.run([sys.executable,str(SCRIPTS/'render_luster_video_frame.py'),str(root),args.frame,str(export/'selection.json')],env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    rendering=json.loads(receipt.read_text());complete=json.loads((export/'complete.json').read_text())
    if rendering['checkpoint_sha256']!=complete['checkpoint_sha256']:raise ValueError('Temporal render used a different model')
    archived=archive(root,selected['checkpoint'],args.remote_root)
    # Keep full-state resume checkpoints, including unselected gates, durably.
    for checkpoint in (frame_root/'runs').glob('*/trainer/lookcloser/seed42/nerfstudio_models/*.ckpt'):
        if str(checkpoint)!=selected['checkpoint']:archive(root,checkpoint,args.remote_root,release=True)
    snapshot=dict(frame=args.frame,run=str(args.run),selection=str(export/'selection.json'),
                  training_selection={k:selected[k] for k in ['step','eval_all_psnr','eval_all_ssim','eval_all_lpips']},
                  export_metrics={k:final[k] for k in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']},
                  archived_checkpoint=archived,render_receipt=str(receipt),review_status='pending_visual_review',time=time.time())
    write(root/'snapshots'/f'{args.frame}.json',snapshot)
    remote=f'ubuntu@dev3:{args.remote_root}/artifacts/frames/{args.frame}/'
    # Earlier checkpoint archives live under this same directory. No --delete:
    # released local checkpoints must remain in the durable archive.
    subprocess.run(['rsync','-a','--exclude=*.ckpt',str(frame_root)+'/',remote],check=True)
    print(json.dumps(snapshot,indent=2))


if __name__=='__main__':main()
