"""Posthoc localized dark-ridge attribution; not an anatomical hole mask."""
import cv2
import numpy as np
from PIL import Image,ImageDraw
from study_temporal_source_retention import ROOT,BASE
from joint_temporal_texture import read,sha,atomic_json
from review_jaw_repair_transfer import verified_image


def main():
    frame='001123';dest=ROOT/'nose_attribution';dest.mkdir(exist_ok=False)
    a,ar=verified_image(BASE,frame);b,br=verified_image(ROOT/frame,frame)
    roi=[885,610,945,750];light=a.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32)
    mask=np.zeros(light.shape,bool);x0,y0,x1,y1=roi;mask[y0:y1,x0:x1]=True
    mask&=light<cv2.medianBlur(light,5)-20
    yy,xx=np.nonzero(mask);records=[];bindings={}
    for name,folder,rgb,result in [('baseline',BASE/'frames'/frame,a,ar),
                                  ('relative',ROOT/frame/'frames'/frame,b,br)]:
        ids=np.rot90(np.array(Image.open(folder/'source_ids.png')))
        depth=np.rot90(np.load(folder/'target_depth.npz')['depth'])
        source,count=np.unique(ids[mask],return_counts=True)
        records.append(dict(variant=name,selected_pixels=len(xx),zero_rgb=int((rgb[mask].max(1)==0).sum()),
            no_source=int((ids[mask]==255).sum()),no_depth=int((depth[mask]<=0).sum()),
            sources={result['source_cameras'][s] if s<62 else 'missing':int(n) for s,n in zip(source,count)}))
        np.savez_compressed(dest/(name+'.npz'),portrait_xy=np.c_[xx,yy],native_xy=np.c_[1919-yy,xx],
            rgb=rgb[mask],source_ids=ids[mask],depth=depth[mask])
        for p in [folder/'complete.json',folder/'frame.png',folder/'source_ids.png',folder/'target_depth.npz',folder/'result.json']:
            bindings[str(p)]=sha(p)
    box=[825,540,1010,810];images=[a,b,a.copy()];images[-1][mask]=[255,0,255]
    out=Image.new('RGB',(3*(box[2]-box[0]),box[3]-box[1]+24));draw=ImageDraw.Draw(out)
    for i,(name,im) in enumerate(zip(['baseline','relative .5','selected diagnostic'],images)):
        off=i*(box[2]-box[0]);out.paste(Image.fromarray(im).crop(box),(off,24));draw.text((off+2,4),name,fill='white')
    out.save(dest/'nose.png')
    atomic_json(dest/'result.json',dict(frame=frame,roi=roi,selection='baseline luma < 5x5 median - 20/255',
        records=records,changed_selected_rgb=int(np.any(a[mask]!=b[mask],1).sum()),
        input_hashes=bindings,script_sha256=sha(__file__),
        outputs={p.name:sha(p) for p in dest.iterdir() if p.is_file()},
        posthoc_localization_only=True,ground_truth_used=False,
        geometric_accuracy_not_proven_by_positive_depth=True,visual_status='pending'))
    print(records,flush=True)


if __name__=='__main__':main()
