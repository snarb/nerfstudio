"""Assemble matched-stage metrics and unfiltered GT/render comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--selected',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    results=[];requests=[];identities=[];origins=[]
    for run in args.runs:
        if not (run/'complete.json').exists():raise ValueError(f'Stage is incomplete: {run}')
        results.append(json.loads((run/'selection.json').read_text()) if args.selected else json.loads((run/'history.json').read_text())[-1])
        requests.append(json.loads((run/'request.json').read_text()))
        origin=run;seen=set()
        while True:
            if origin in seen:raise ValueError('Cyclic run ancestry')
            seen.add(origin);request=json.loads((origin/'request.json').read_text())
            if not request.get('parent_history'):break
            origin=Path(request['parent_history']).parent
        origins.append(origin);identities.append(json.loads((origin/'identity.json').read_text()))
    if not args.selected and len({r['step'] for r in results})!=1:raise ValueError('Matched comparison requires equal steps')
    for request in requests[1:]:
        if request['data']!=requests[0]['data'] or request.get('rois_by_image')!=requests[0].get('rois_by_image'):raise ValueError('Comparison inputs differ')
    summary=[]
    for run,result in zip(args.runs,results):
        item=dict(run=str(run),step=result['step'],**{k:result[k] for k in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']})
        for split in ['eval','train']:
            views=[v for v in result['per_view'] if v['split']==split]
            item[f'{split}_foreground_psnr']=float(np.mean([v['foreground_psnr'] for v in views]))
            item[f'{split}_foreground_opacity']=float(np.mean([v['foreground_opacity'] for v in views]))
            for label in ['face','hair','hand','clothing']:
                for metric in ['psnr','ssim','lpips']:
                    item[f'{split}_{label}_{metric}']=float(np.mean([v['rois'][label][metric] for v in views if label in v['rois']]))
            all_rois=[roi for v in views for roi in v['rois'].values()]
            for metric in ['psnr','ssim','lpips']:
                item[f'{split}_detail_{metric}']=float(np.mean([v[metric] for v in all_rois]))
        summary.append(item)
    parity=dict(initial_weights_equal=len({i['initial_weights'] for i in identities})==1,
                first_batches_equal=len({(run/'first_batches.json').read_text() for run in origins})==1,
                transforms_equal=len({i['transforms_sha256'] for i in identities})==1,
                subject='Matched-stage comparison' if not args.selected else 'Each run metric-selected checkpoint')
    (args.output/'metrics.json').write_text(json.dumps(dict(results=summary,parity=parity),indent=2)+'\n')
    for split in ['eval','train']:
        views=[v for v in results[0]['per_view'] if v['split']==split]
        for label in ['full','face','hair','hand','clothing']:
            panels=[]
            for view in views:
                gt=Image.open(Path(requests[0]['data'])/'images'/view['image']).convert('RGB')
                images=[gt]+[Image.open(Path(result['render_dir'])/f'{split}_{view["index"]:03d}.png').convert('RGB') for result in results]
                if label!='full':
                    box=requests[0]['rois_by_image'][view['image']][label];images=[im.crop(box) for im in images]
                else:
                    for im in images:im.thumbnail((600,500))
                width=max(im.width for im in images);height=max(im.height for im in images)
                row=Image.new('RGB',(width*len(images),height+28),(35,35,35));d=ImageDraw.Draw(row)
                for i,im in enumerate(images):
                    row.paste(im,(i*width,28));name='GT' if i==0 else args.runs[i-1].name
                    d.text((i*width+4,4),f'{view["physical_camera"]} {name}',fill='white')
                panels.append(row)
            sheet=Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),(35,35,35));y=0
            for panel in panels:sheet.paste(panel,(0,y));y+=panel.height
            sheet.save(args.output/f'{split}_{label}.jpg',quality=95)
    print(json.dumps(dict(parity=parity,summary=[{k:v for k,v in s.items() if k in ['run','step','eval_all_psnr','eval_all_ssim','eval_all_lpips','eval_foreground_psnr','train_foreground_psnr','eval_detail_psnr','train_detail_psnr','eval_face_psnr','train_face_psnr']} for s in summary]),indent=2))


if __name__=='__main__':main()
