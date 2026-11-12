"""Original120fps BC photographs: animated native crop and temporal soft alpha.

No low-resolution source surrogate and no RGB frame interpolation. A variable
crop is sampled once into the fixed FullHD video; the last window is exactly1:1.
"""
import argparse,json,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np


def make(out):
    out=Path(out);source=out/'gt_sources';dest=out/'gt';dest.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    manifest=json.loads((source/'manifest.json').read_text());path=json.loads((out/'path.json').read_text())
    frames=range(1251,1270)
    rgb={f:cv2.imread(str(source/f'{f:06}_rgb.png')) for f in frames}
    alpha={f:cv2.imread(str(source/f'{f:06}_alpha.png'),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255 for f in frames}
    grey={f:cv2.cvtColor(cv2.resize(rgb[f],None,fx=.5,fy=.5),cv2.COLOR_BGR2GRAY) for f in frames}
    h,w=alpha[1251].shape;yy,xx=np.mgrid[:h,:w].astype(np.float32)
    def one(row):
        frame=row['source_frame'];candidates=[alpha[frame]];qualified=[]
        for neighbour in (frame-2,frame-1,frame+1,frame+2):
            if neighbour not in rgb:continue
            flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            forward=cv2.resize(flow.calc(grey[frame],grey[neighbour],None),(w,h))*2
            backward=cv2.resize(flow.calc(grey[neighbour],grey[frame],None),(w,h))*2
            mx,my=xx+forward[...,0],yy+forward[...,1]
            reverse=cv2.remap(backward,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
            valid=np.linalg.norm(forward+reverse,axis=-1)<2
            valid&=(mx>=0)&(mx<w-1)&(my>=0)&(my<h-1)
            other=cv2.remap(alpha[neighbour],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
            candidates.append(np.where(valid,other,alpha[frame]));qualified.append(float(valid.mean()))
        matte=np.median(np.stack(candidates),axis=0).astype(np.float32)
        x,y,cw,ch=row['native_crop'];x-=2665;y-=650
        assert 0<=x<=w-cw and 0<=y<=h-ch
        photo=rgb[frame][y:y+ch,x:x+cw];a=matte[y:y+ch,x:x+cw]
        composed=photo.astype(np.float32)*a[...,None]
        # No source pre-resize. Exactly one final crop-to-video sampling step.
        if (cw,ch)!=(1920,1080):composed=cv2.resize(composed,(1920,1080),interpolation=cv2.INTER_LANCZOS4)
        image=np.uint8(np.clip(composed+.5,0,255));cv2.imwrite(str(dest/f'{row["index"]:03}.png'),image)
        cv2.imwrite(str(dest/f'{row["index"]:03}_alpha.png'),np.uint8(np.clip(matte*255+.5,0,255)))
        return dict(index=row['index'],source_frame=frame,crop=row['native_crop'],source_rgb_sha256=hashlib.sha256((source/f'{frame:06}_rgb.png').read_bytes()).hexdigest(),
            frame_sha256=hashlib.sha256((dest/f'{row["index"]:03}.png').read_bytes()).hexdigest(),qualified_flow_fraction=qualified,
            final_rgb_sampling='identity native crop' if (cw,ch)==(1920,1080) else 'single Lanczos crop-to-FullHD',source_pre_resize=False)
    rows=[r for r in path['records'] if r['kind']=='native_gt']
    # Exact endpoint photograph at the same source time as the last neural view,
    # for boundary review only; not an inserted duplicate temporal video frame.
    endpoint=dict(rows[0],index=69,source_frame=1253)
    with ThreadPoolExecutor(6) as pool:results=list(pool.map(one,[endpoint]+rows))
    panels=[]
    for index in [69,70,76,83]:
        im=cv2.imread(str(dest/f'{index:03}.png'));im=cv2.resize(im,(960,540));cv2.putText(im,str(index),(12,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1);panels.append(im)
    cv2.imwrite(str(out/'gt_contact.jpg'),np.concatenate(panels,axis=0))
    receipt=dict(camera='BC',frames=14,endpoint_reference=69,slowmo_speed=.2,RGB_frame_interpolation=False,
        alpha_source='original8K SAM soft alpha; five-frame bidirectional-flow-qualified temporal median',AABB_used_for_GT=False,
        source_pre_resize=False,output_sampling='one sampling of animated native crop; final1920x1080 window1:1',rows=results)
    (out/'gt_receipt.json').write_text(json.dumps(receipt,indent=2));print('GT_DONE',len(results))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);make(p.parse_args().output)
