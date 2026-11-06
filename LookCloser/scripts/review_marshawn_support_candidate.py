"""Source/renders and diagnostic support histograms for one isolated candidate."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main(a):
    out=a.output/'review';out.mkdir(exist_ok=True);data=a.base.parent/'data';records={r['camera']:r for r in json.loads((data/'ready.json').read_text())['records']};regions=json.loads((a.base/'corrected_v2/review/region_crops.json').read_text());font=ImageFont.truetype('DejaVuSans.ttf',15);panels=[]
    labels=['SOURCE','clip20k','partial pad1','min2 pad1']
    for camera in ['DC','HC','KA']:
        r=records[camera];images=[Image.open(data/r['image']).convert('RGB')]+[Image.open(a.output/'renders'/camera/(camera+'_'+k+'.png')).convert('RGB') for k in ['inside','raw_semantic_gate_pad1','semantic_gate_pad1']]
        for name,roi in regions[camera].items():
            crops=[im.crop(roi) for im in images];w,h=crops[0].size;panel=Image.new('RGB',(4*w,h+35),(20,20,20));draw=ImageDraw.Draw(panel)
            for i,(im,label) in enumerate(zip(crops,labels)):panel.paste(im,(i*w,35));draw.text((i*w+3,4),camera+' '+name+' '+label,fill='white',font=font)
            panel.save(out/(camera+'_'+name+'.png'));panel.thumbnail((1800,450));panels.append(panel)
        bounds=np.asarray(list(r['crops'].values()));roi=(max(0,int(bounds[:,0].min())-25),max(0,int(bounds[:,1].min())-20),min(images[0].width,int(bounds[:,2].max())+25),min(images[0].height,int(bounds[:,3].max())+20));w,h=images[0].crop(roi).size;panel=Image.new('RGB',(4*w,h+35),(20,20,20));draw=ImageDraw.Draw(panel)
        for i,(im,label) in enumerate(zip(images,labels)):panel.paste(im.crop(roi),(i*w,35));draw.text((i*w+3,4),camera+' '+label,fill='white',font=font)
        panel.save(out/(camera+'_full.png'))
    sheet=Image.new('RGB',(max(x.width for x in panels),sum(x.height for x in panels)),(20,20,20));y=0
    for panel in panels:sheet.paste(panel,(0,y));y+=panel.height
    sheet.save(out/'all_details.jpg',quality=95)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    receipt=json.loads((a.base/'positive_support_audit128/receipt.json').read_text());fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,group in zip(axes,['gate0','pad1_nominal_floor_band_NOT_surface_label']):
        hist=receipt['histograms'][group]
        for key,label in [('W64_overlap','Any W64 overlap'),('W64_overlap_no_unknown_footprint','W64 overlap; rectangle UNKNOWN-free'),('nominal_actor_overlap_no_unknown_footprint','Nominal actor; rectangle UNKNOWN-free')]:
            y=np.asarray(hist['counts'][key]);ax.step(np.arange(64),np.cumsum(y)/hist['cells'],where='post',label=label)
        ax.set(xlabel='Positive TRAIN-view count',ylabel='Cumulative fraction of cells',title='Original gate0' if group=='gate0' else 'z=[-.01,.01] coordinate band (not floor truth)',ylim=(0,1));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'support_counts.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
