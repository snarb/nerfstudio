"""Read completed external jobs; write only a separate transfer campaign."""
import argparse
import csv
import json
import shutil
from pathlib import Path
from luster_hybrid_common import sha, write


def inventory(training, root):
    if root.resolve() == training.resolve() or training.resolve() in root.resolve().parents:
        raise ValueError('Transfer results must be outside the training campaign')
    for name in ['logs', 'preflight', 'frames', 'evaluation']:
        (root/name).mkdir(parents=True, exist_ok=True)
    records=[]
    for path in sorted((training/'frames').glob('*/evaluation/metrics.json')):
        data=json.loads(path.read_text())
        checkpoint=Path(data['checkpoint'])
        if not checkpoint.exists():
            continue
        assert data['step']==106316
        assert sha(checkpoint)==data['checkpoint_sha256']
        config=Path(data['config'])
        assert sha(config)==data['config_sha256']
        frame=int(data['frame'])
        assert frame==int(path.parent.parent.name)
        dest=root/'frames'/f'{frame:06d}'
        dest.mkdir(exist_ok=True)
        context=dict(frame=frame,origin=str(path.parent.parent),
                     source=f'/fsx/tmp/luster/root_8s/working/fullres/{frame:06d}',
                     checkpoint=str(checkpoint),checkpoint_sha256=data['checkpoint_sha256'],
                     config=str(config),config_sha256=data['config_sha256'],
                     original_metrics=str(path),original_metrics_sha256=sha(path))
        if frame==540:
            context['origin']='/fsx/tmp/luster_000540_body_opt_20261016'
        context_path=dest/'context.json'
        if context_path.exists():
            assert json.loads(context_path.read_text())==context
        else:
            write(context_path,context)
        for name in ['logs','preflight','evaluation','baseline','geometry','candidates']:
            (dest/name).mkdir(exist_ok=True)
        for name in ['runtime','code']:
            link=dest/name
            if not link.exists():
                link.symlink_to(root/name,target_is_directory=True)
        records.append(dict(frame=frame,checkpoint=str(checkpoint),checkpoint_sha256=data['checkpoint_sha256'],
            baseline_hd_lpips=data['heldout_mean']['lpips'],baseline_hd_psnr=data['heldout_mean']['psnr'],
            baseline_hd_ssim=data['heldout_mean']['ssim'],
            cam007_hd_lpips=next(row['lpips'] for row in data['rows'] if row['camera']==7),
            context=str(context_path),
            split='development' if frame<585 else 'locked_temporal_validation'))
    records.sort(key=lambda x:-x['baseline_hd_lpips'])
    write(root/'preflight/completed_frames.json',dict(
        selection='Only complete published evaluation markers with verified final checkpoint/config',
        training_root=str(training),frames=records,
        target='At least 30% native heldout-person LPIPS reduction on every frame; no new obvious geometric artifacts',
        protocol_note='HD training CSV only prioritizes frames; native baseline and hybrid are evaluated separately',
        gpu_policy='Own inference capped at 4 GiB torch allocator, 6 GiB process watchdog, admission free>=64 GiB; total GPU used>=24 GiB aborts ONLY own process. CPU scoring.',
        training_mutations=False))
    with (root/'evaluation/baseline_inventory.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    for src in [training/'metrics.csv',training/'distributed/metrics.csv']:
        if src.exists():
            shutil.copy2(src,root/'preflight'/('distributed_metrics_snapshot.csv' if src.parent.name=='distributed' else 'training_metrics_snapshot.csv'))
    print(json.dumps({'completed':len(records),'worst':[x['frame'] for x in records[:5]]}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();inventory(a.training,a.root)
