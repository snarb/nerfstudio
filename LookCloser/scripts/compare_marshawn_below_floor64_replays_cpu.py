"""Paired native64-step renders: float differences, fixed metrics and9 details."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def panel(images,labels,crop):
    tiles=[im.crop(crop)for im in images];w=max(x.width for x in tiles);h=max(x.height for x in tiles)
    out=Image.new('RGB',(w*len(tiles),h+30),'#171717');d=ImageDraw.Draw(out)
    for i,(tile,label)in enumerate(zip(tiles,labels)):out.paste(tile,(i*w,30));d.text((i*w+2,3),label,fill='white')
    return out

def main(root):
    s=json.loads((root/'comparison_spec.json').read_text());ready=json.loads(Path(s['ready']).read_text());regions=json.loads(Path(s['regions']).read_text());rows=[]
    for arm in ['control0','prior1e4','prior1e3']:
        dest=root/'comparison'/arm;dest.mkdir(parents=True,exist_ok=True)
        ca=json.loads((root/'renders'/(arm+'_resume64')/'complete.json').read_text());cb=json.loads((root/'renders'/(arm+'_full64')/'complete.json').read_text())
        for camera in ['DC','HC','KA']:
            a=np.load(root/'renders'/(arm+'_resume64')/(camera+'_float.npz'));b=np.load(root/'renders'/(arm+'_full64')/(camera+'_float.npz'))
            rgb=np.abs(a['rgb'].astype(np.float64)-b['rgb']);alpha=np.abs(a['alpha'].astype(np.float64)-b['alpha'])
            Image.fromarray((np.clip(rgb*20,0,1)*255+.5).astype(np.uint8)).save(dest/(camera+'_RGB_abs_x20.png'))
            Image.fromarray((np.clip(alpha[...,0]*20,0,1)*255+.5).astype(np.uint8)).save(dest/(camera+'_alpha_abs_x20.png'))
            row=next(x for x in ready['records']if x['camera']==camera)
            target=Image.open(Path(s['ready']).parent/row['image']).convert('RGB')
            ia=Image.fromarray((np.clip(a['rgb'],0,1)*255+.5).astype(np.uint8));ib=Image.fromarray((np.clip(b['rgb'],0,1)*255+.5).astype(np.uint8))
            for name,crop in regions[camera].items():panel([target,ia,ib],[camera+' GT',arm+'32+32',arm+'64'],crop).save(dest/(camera+'_'+name+'.png'))
            for actor,crop in row['crops'].items():panel([target,ia,ib],[camera+' GT',arm+'32+32',arm+'64'],crop).save(dest/(camera+'_'+actor+'_actor.png'))
            rows.append(dict(arm=arm,camera=camera,RGB_max=float(rgb.max()),RGB_mean=float(rgb.mean()),RGB_RMSE=float(np.sqrt(np.mean(rgb**2))),
                alpha_max=float(alpha.max()),alpha_mean=float(alpha.mean()),pixels_RGB_delta_gt001=int((rgb.max(-1)>.01).sum()),
                metric_delta={k:ca['by_view'][camera][k]-cb['by_view'][camera][k]for k in ['psnr','ssim','lpips']}))
    output=dict(passed=True,CPU_only=True,rows=rows,quality_promotion=False,interpretation='Read-only practical inference differences at identical native renderer settings; numerical cause is not established.')
    (root/'comparison/summary.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
