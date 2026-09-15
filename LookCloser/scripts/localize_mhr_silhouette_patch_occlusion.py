"""Posthoc signed depth changes; never an input to fitting or admission."""
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from PIL import Image, ImageDraw
from admit_mhr_silhouette_patch import OUT
from study_multiview_face_prior import read, save, sha


def depth_statistics(base, candidate):
    common=(base>0)&(candidate>0)
    delta=np.zeros(base.shape);delta[common]=candidate[common]-base[common]
    return delta, dict(common_changed=int((abs(delta)>1e-6).sum()),
        nearer_over_003=int((delta<-.003).sum()),nearer_over_01=int((delta<-.01).sum()),
        farther_over_003=int((delta>.003).sum()),minimum_signed_delta=float(delta.min()),
        maximum_signed_delta=float(delta.max()))


def main():
    root=OUT/'occlusion_review';root.mkdir(exist_ok=False)
    records=[];bindings={};files=[]
    for view in ['old_moving','F004_E','M004_B','C004_E']:
        frames={}
        for variant in ['baseline','strict','interpolated']:
            folder=OUT/'rgb'/view/variant/'frames/001193'
            receipt=read(folder/'complete.json')
            for name,h in receipt['hashes'].items():assert sha(folder/name)==h
            bindings[str(folder/'complete.json')]=sha(folder/'complete.json')
            frames[variant]=(np.array(Image.open(folder/'frame.png')),np.rot90(np.load(folder/'target_depth.npz')['depth']))
        base,bd=frames['baseline']
        for variant in ['strict','interpolated']:
            im,d=frames[variant];delta,stats=depth_statistics(bd,d)
            components,n=label(abs(delta)>.003)
            regions=[]
            for i in range(1,n+1):
                yy,xx=np.where(components==i)
                regions.append(dict(component=i,pixels=len(xx),bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],
                    signed_min=float(delta[yy,xx].min()),signed_max=float(delta[yy,xx].max()),
                    baseline_depth_range=[float(bd[yy,xx].min()),float(bd[yy,xx].max())],
                    candidate_depth_range=[float(d[yy,xx].min()),float(d[yy,xx].max())]))
            regions.sort(key=lambda x:(x['signed_min'],-x['pixels']))
            for region in regions[:5]:
                x0,y0,x1,y1=region['bbox'];crop=(max(0,x0-70),max(0,y0-70),min(1080,x1+71),min(1920,y1+71))
                marked=im.copy();marked[components==region['component']]=[255,0,255]
                w,h=crop[2]-crop[0],crop[3]-crop[1];panel=Image.new('RGB',(3*w,h+25));draw=ImageDraw.Draw(panel)
                for col,(title,img) in enumerate([('baseline',base),(variant,im),('depth change > .003',marked)]):
                    panel.paste(Image.fromarray(img).crop(crop),(col*w,25));draw.text((col*w+3,5),title,fill='white')
                path=root/f'{view}_{variant}_{region["component"]}.png';panel.save(path)
                region['review_crop']=list(crop);region['panel']=str(path);files.append(dict(path=str(path),sha256=sha(path)))
            records.append(dict(view=view,branch=variant,statistics=stats,regions=regions))
    save(root/'result.json',dict(records=records,input_hashes=bindings,files=files,script_sha256=sha(__file__),
        sign_convention='candidate minus baseline ray depth; negative is new nearer occlusion',posthoc_only=True))
    print([(r['view'],r['branch'],r['statistics']) for r in records])


if __name__=='__main__':main()
