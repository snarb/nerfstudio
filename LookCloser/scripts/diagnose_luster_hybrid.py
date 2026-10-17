"""Evaluation-only spatial attribution of remaining native LPIPS error.

These maps use target GT/masks and must never feed source selection or transfer.
LPIPS maps have broad receptive fields; strata are diagnostic, not causal masks.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
from luster_hybrid_common import bootstrap,geometry,read,sha,write


def run(a):
    bootstrap(a.root)
    import lpips
    torch.set_num_threads(4);cv2.setNumThreads(4)
    metric=lpips.LPIPS(net='alex',spatial=True).eval().to(a.device)
    protocol=read(a.root/'evaluation/protocol.json');out=a.root/'evaluation'/f'{a.name}_spatial';out.mkdir(exist_ok=True)
    rows=[]
    for c in [7,12,95,150]:
        e=protocol['cameras'][str(c)];assert sha(e['source'])==e['source_sha256'] and sha(e['mask'])==e['mask_sha256']
        gt=np.asarray(Image.open(e['source']).convert('RGB'),np.float32)/255;mask=np.asarray(Image.open(e['mask']))>0
        base=geometry(a.root,c)['rgb'];pred=np.load(a.root/f'candidates/{a.name}/cam{c:03d}.npy',mmap_mode='r');weight=np.load(a.root/f'candidates/{a.name}/cam{c:03d}_confidence.npy',mmap_mode='r')
        yy,xx=np.where(mask);x0=max(0,int(xx.min())-16);x1=min(mask.shape[1],int(xx.max())+17);y0=max(0,int(yy.min())-16);y1=min(mask.shape[0],int(yy.max())+17);sl=np.s_[y0:y1,x0:x1];m=mask[sl]
        mt=torch.from_numpy(m.astype(np.float32))[None,None].to(a.device)
        g=torch.from_numpy(np.array(gt[sl])).permute(2,0,1)[None].to(a.device);g=g*mt+.5*(1-mt)
        maps=[]
        with torch.inference_mode():
            for source in [base,pred]:
                p=torch.from_numpy(np.array(source[sl])).permute(2,0,1)[None].to(a.device);p=p*mt+.5*(1-mt)
                maps.append(metric(p*2-1,g*2-1)[0,0].cpu().numpy())
        baseline,hybrid=maps;distance=cv2.distanceTransform(m.astype(np.uint8),cv2.DIST_L2,5);w=np.asarray(weight[sl])
        strata={'all':m,'fallback':m&(w==0),'fade':m&(w>0)&(w<.999),'full_transfer':m&(w>=.999),'silhouette16':m&(distance<=16),'interior':m&(distance>16)}
        record={'camera':c,'crop':[x0,y0,x1,y1],'strata':{}}
        total=hybrid[m].sum()
        for name,support in strata.items():
            record['strata'][name]=dict(pixels=int(support.sum()),area_fraction=float(support.sum()/m.sum()),baseline_lpips=float(baseline[support].mean()) if support.any() else None,hybrid_lpips=float(hybrid[support].mean()) if support.any() else None,remaining_error_fraction=float(hybrid[support].sum()/total))
        tile=[]
        for y in range(0,m.shape[0],128):
            for x in range(0,m.shape[1],128):
                t=np.s_[y:y+128,x:x+128];inside=m[t]
                if inside.sum()<128:continue
                tile.append(dict(box=[x+x0,y+y0,min(x+x0+128,x1),min(y+y0+128,y1)],pixels=int(inside.sum()),baseline=float(baseline[t][inside].mean()),hybrid=float(hybrid[t][inside].mean()),sum_error=float(hybrid[t][inside].sum())))
        record['highest_remaining_tiles']=sorted(tile,key=lambda v:v['sum_error'],reverse=True)[:20]
        np.savez_compressed(out/f'cam{c:03d}_lpips_maps.npz',baseline=baseline,hybrid=hybrid,mask=m,crop=np.array(record['crop']))
        panels=[]
        for values in [baseline,hybrid]:
            colors=cv2.applyColorMap(np.rint(np.clip(values/.35,0,1)*255).astype(np.uint8),cv2.COLORMAP_INFERNO)[...,::-1];colors[~m]=0;panels.append(colors)
        delta=np.zeros((*m.shape,3),np.float32);diff=hybrid-baseline;delta[...,0]=np.clip(diff/.08,0,1);delta[...,1]=np.clip(-diff/.08,0,1);delta[~m]=0;panels.append(np.rint(delta*255).astype(np.uint8))
        panel=Image.fromarray(np.concatenate(panels,axis=1));panel.thumbnail((1500,1000));im=Image.new('RGB',(panel.width,panel.height+24));im.paste(panel,(0,24));draw=ImageDraw.Draw(im)
        for i,title in enumerate(['LookCloser LPIPS (0-.35)','Hybrid LPIPS (0-.35)','Delta: green improved / red worse']):draw.text((i*panel.width//3+3,5),title,fill='white')
        im.save(out/f'cam{c:03d}_errors.png');rows.append(record);print(c,record['strata'],flush=True)
    write(out/'summary.json',dict(candidate=a.name,usage='evaluation_only_never_construction',device=a.device,code_sha256=sha(Path(__file__)),rows=rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args();run(a)
