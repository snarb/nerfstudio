"""Collect native float metrics, selectors and paired initialization receipts."""
import argparse
import json
from pathlib import Path
from statistics import mean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = []
    for folder in sorted(args.root.iterdir()):
        if not (folder/'history.json').exists(): continue
        history = json.loads((folder/'history.json').read_text())
        selection = json.loads((folder/'selection.json').read_text())
        selected_step = int(Path(selection['path']).stem.split('_')[-1])
        identity = json.loads((folder/'identity.json').read_text())
        first_batches = json.loads((folder/'first_batches.json').read_text())
        for record in history:
            row = dict(run=folder.name,step=record['step'],selected=record['step']==selected_step,
                       complete=(folder/'complete.json').exists(),eval_stride=record['eval_stride'],
                       initial_weights=identity['initial_weights'],first_batches=first_batches,
                       full={k:record['eval_all_'+k] for k in ['psnr','ssim','lpips']})
            for split in ['train','eval']:
                views = [v for v in record['per_view'] if v['split']==split]
                row[split+'_full'] = {k:mean(v[k] for v in views) for k in ['psnr','ssim','lpips']}
                regions = sorted({r for v in views for r in v['rois']})
                row[split+'_rois'] = {r:{k:mean(v['rois'][r][k] for v in views if r in v['rois'])
                                        for k in ['psnr','ssim','lpips']} for r in regions}
                details = [v['rois'][r] for v in views for r in ['face','hair','lipstick'] if r in v['rois']]
                if details:
                    row[split+'_detail'] = {k:mean(d[k] for d in details) for k in ['psnr','ssim','lpips']}
            rows.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,indent=2)+'\n')
    print('| Run | Step | Selected | Eval PSNR | SSIM | LPIPS | Face PSNR | Train face PSNR |')
    print('| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |')
    for row in rows:
        face = row['eval_rois'].get('face',{}).get('psnr',float('nan'))
        train = row['train_rois'].get('face',{}).get('psnr',float('nan'))
        full = row['full']
        print(f"| {row['run']} | {row['step']} | {row['selected']} | {full['psnr']:.3f} | "
              f"{full['ssim']:.4f} | {full['lpips']:.4f} | {face:.3f} | {train:.3f} |")


if __name__ == '__main__': main()
