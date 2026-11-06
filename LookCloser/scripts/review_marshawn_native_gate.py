"""CPU source/baseline/previous/current details for a completed native gate."""
import argparse,hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    campaign=a.production.parent;base=campaign/'native_semantic_grid_20000';ready=json.loads((campaign/'data/ready.json').read_text());records={r['camera']:r for r in ready['records']};state=json.loads((a.production/'state.json').read_text());history=sorted(state['evaluations'],key=lambda r:r['step']);current=next(r for r in history if r['step']==a.step);previous=max((r for r in history if r['step']<a.step),key=lambda r:r['step']);baseline=history[0]
    out=Path(current['evaluation']).parent/'detail_review';out.mkdir(exist_ok=False);regions=json.loads((base/'corrected_v2/review/region_crops.json').read_text());panels=[];font=ImageFont.truetype('DejaVuSans.ttf',14);sources=[]
    for camera in ['DC','HC','KA']:
        r=records[camera];paths=[campaign/'data'/r['image']]+[Path(e['renders'])/camera/(camera+'_inside.png') for e in [baseline,previous,current]]
        images=[Image.open(p).convert('RGB') for p in paths];labels=['SOURCE','native20k','previous'+str(previous['step']),'current'+str(a.step)]
        sources.extend(dict(path=str(p),sha256=sha(p)) for p in paths)
        for name,roi in regions[camera].items():
            crops=[im.crop(roi) for im in images];w,h=crops[0].size;panel=Image.new('RGB',(4*w,h+37),(20,20,20));draw=ImageDraw.Draw(panel)
            for i,(im,label) in enumerate(zip(crops,labels)):
                panel.paste(im,(i*w,37));draw.text((i*w+3,2),camera+' '+name,fill='white',font=font);draw.text((i*w+3,18),label,fill='white',font=font)
            panel.save(out/(camera+'_'+name+'.png'));panel.thumbnail((1800,450));panels.append(panel)
    sheet=Image.new('RGB',(max(x.width for x in panels),sum(x.height for x in panels)),(20,20,20));y=0
    for panel in panels:sheet.paste(panel,(0,y));y+=panel.height
    sheet.save(out/'all_details.jpg',quality=95)
    (out/'provenance.json').write_text(json.dumps(dict(step=a.step,previous_step=previous['step'],source_files=sources,regions_sha256=sha(base/'corrected_v2/review/region_crops.json'),evaluation_sha256=sha(Path(current['evaluation'])),helper_sha256=sha(Path(__file__)),no_GPU_or_model_loaded=True),indent=2)+'\n')
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--production',type=Path,required=True);p.add_argument('--step',type=int,required=True);main(p.parse_args())
