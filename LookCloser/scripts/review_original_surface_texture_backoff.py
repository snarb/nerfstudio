"""Independently verify and expose every changed pixel of the texture backoff."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from study_multiview_face_prior import read,save,sha

ROOT=Path('/mnt/data/dec5_mhr_original_surface_texture_backoff')


def main():
    r=read(ROOT/'result.json'); dest=ROOT/'review'; assert not dest.exists()
    for p,h in r['input_hashes'].items(): assert sha(p)==h,p
    for p,h in r['hashes'].items(): assert sha(ROOT/p)==h,p
    base,candidate=Path(r['baseline']),Path(r['candidate'])
    q=read(candidate.parent.parent/'request.json'); bindings={}
    for name,h in q['helpers'].items():
        path=Path(__file__).with_name(name); assert sha(path)==h,path; bindings[str(path)]=h
    mask=np.rot90(np.load(ROOT/'evidence.npz')['mask'])
    images={label:np.asarray(Image.open(path/'frame.png').convert('RGB'))
            for label,path in [('baseline',base),('raw patch',candidate),('backoff',ROOT)]}
    before,raw,fixed=images.values()
    np.testing.assert_array_equal(fixed[mask],before[mask]); np.testing.assert_array_equal(fixed[~mask],raw[~mask])
    changed=np.any(fixed!=raw,axis=2); np.testing.assert_array_equal(changed,mask)
    source=np.asarray(Image.open(ROOT/'source_ids.png')); old_source=np.asarray(Image.open(base/'source_ids.png'))
    raw_source=np.asarray(Image.open(candidate/'source_ids.png')); native=np.rot90(mask,-1)
    np.testing.assert_array_equal(source[native],old_source[native]); np.testing.assert_array_equal(source[~native],raw_source[~native])
    old_new_black=(before.max(2)>0)&(raw.max(2)==0)
    new_new_black=(before.max(2)>0)&(fixed.max(2)==0)
    assert int(old_new_black.sum())==3 and not new_new_black.any()
    dest.mkdir(); files=[]
    for index,(y,x) in enumerate(np.argwhere(mask)):
        crop=(max(0,int(x)-30),max(0,int(y)-30),min(1080,int(x)+31),min(1920,int(y)+31))
        w,h=crop[2]-crop[0],crop[3]-crop[1]; panel=Image.new('RGB',(w*3,h+24)); draw=ImageDraw.Draw(panel)
        for col,(name,im) in enumerate(images.items()):
            panel.paste(Image.fromarray(im).crop(crop),(col*w,24)); draw.text((col*w+2,4),name,fill='white')
        path=dest/f'pixel_{index}.png'; panel.save(path); files.append(str(path))
    for path in [ROOT/'result.json',ROOT/'evidence.npz',Path(__file__),base/'complete.json',candidate/'complete.json']:
        bindings[str(path)]=sha(path)
    save(dest/'result.json',dict(changed_pixels=int(mask.sum()),new_black_before=3,new_black_after=0,
        only_exact_verified_baseline_rgb_and_source_ids_reused=True,unchanged_pixels_bit_exact=True,
        no_new_geometry_textured=True,input_hashes=bindings,files={p:sha(p) for p in files},
        visual_status='pending',full_video_review_not_performed=True,production_accepted=False))
    print('Verified exactly three recovered texture pixels; no other RGB/source change',flush=True)


if __name__=='__main__':main()
