"""CPU matched-gate source/baseline/two-map RGB review, with fixed crops."""
import argparse,hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(a):
    out=a.trials/'reviews'/f'step-{a.step:09d}';out.mkdir(parents=True,exist_ok=False)
    root=a.trials.parent;ready=root/'data/ready.json';records={r['camera']:r for r in json.loads(ready.read_text())['records']}
    regions_path=root/'native_semantic_grid_20000/corrected_v2/review/region_crops.json';regions=json.loads(regions_path.read_text())
    paths={'baseline':root/'independent_occupancy/semantic_quarter_distortion_progress'/f'step-{a.step:09d}'}
    labels=['SOURCE','baseline .95 constant','map LR decay','map SSIM .93']
    if a.quarter4k:
        paths={'HD_quarter01':paths['baseline'],'old4K_coarse001':root/'independent_occupancy/semantic_4k_original_r0'/f'step-{a.step:09d}','new4K_quarter01':root/'independent_occupancy/quarter_frozen4k_progress'/f'step-{a.step:09d}'}
        labels=['SOURCE FHD','HD quarter / distortion .1','old4K coarse / distortion .01','new4K quarter / distortion .1']
    else:
        for v in ['lr_decay','threshold093']:paths[v]=root/'independent_occupancy'/('frequency_HD63_'+v+'_progress')/f'step-{a.step:09d}'
    metrics={};bindings={};details=[];actors=[];font=ImageFont.truetype('DejaVuSans.ttf',14)
    for name,path in paths.items():
        if not(path/'fetch_complete.json').exists():raise ValueError('Atomic gate fetch incomplete: '+name)
        m=json.loads((path/'renders/metrics.json').read_text());assert m['step']==a.step
        metrics[name]={k:m['metrics'][k]for k in ['psnr','ssim','lpips']}
        for f in ['archive_receipt.json','fetch_complete.json','renders/metrics.json']:bindings[str(path/f)]=sha(path/f)
    for camera in ['DC','HC','KA']:
        source=root/'data'/records[camera]['image'];files=[source]+[p/'renders'/(camera+'.png')for p in paths.values()]
        images=[Image.open(p).convert('RGB')for p in files];assert all(im.size==(1920,1012)for im in images)
        bindings.update({str(p):sha(p)for p in files})
        crops=list(records[camera]['crops'].values());actor=[max(0,min(c[0]for c in crops)-64),max(0,min(c[1]for c in crops)-64),min(1920,max(c[2]for c in crops)+64),min(1012,max(c[3]for c in crops)+64)]
        for name,box in [('actors_margin64',actor)]+list(regions[camera].items()):
            w,h=box[2]-box[0],box[3]-box[1];panel=Image.new('RGB',(4*w,h+38),'#181818');d=ImageDraw.Draw(panel)
            for i,(im,label)in enumerate(zip(images,labels)):
                panel.paste(im.crop(box),(i*w,38));d.text((i*w+3,2),camera+' '+name,font=font,fill='white');d.text((i*w+3,19),label,font=font,fill='white')
            panel.save(out/(camera+'_'+name+'.png'))
            panel.thumbnail((2000,850 if name=='actors_margin64' else 500));(actors if name=='actors_margin64' else details).append(panel)
    for name,panels in [('all_actors.jpg',actors),('all_details.jpg',details)]:
        image=Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),'#181818');y=0
        for p in panels:image.paste(p,(0,y));y+=p.height
        image.save(out/name,quality=95)
    report=dict(step=a.step,metrics=metrics,inputs_sha256=bindings,ready_sha256=sha(ready),fixed_regions_sha256=sha(regions_path),helper_sha256=sha(__file__),GPU_used=False,promotion=False)
    (out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(output=str(out),metrics=metrics)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=Path,required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--quarter4k',action='store_true');main(p.parse_args())
