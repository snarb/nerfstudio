"""Ordered matched videos and native transition evidence for face source control.

Baseline-only flow diagnostics are not perceptual scores or ground-truth motion.
Never feeds review crops/flow/target RGB back into the render.
"""
import argparse
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from run_temporal_face_angular_control import ROOT,BASE,CLIPS,verify
from build_train_hair_semantics import read,write,sha
from review_temporal_source_retention import temporal_sources


def pair(a,b,label):
    h,w=a.shape[:2];im=Image.new('RGB',(2*w,h+24));im.paste(Image.fromarray(a),(0,24));im.paste(Image.fromarray(b),(w,24))
    draw=ImageDraw.Draw(im);draw.text((3,4),label+' baseline',fill='white');draw.text((w+3,4),'face angular',fill='white');return im


def sheets(items,dest,prefix,cols,rows):
    w,h=items[0].size
    for start in range(0,len(items),cols*rows):
        image=Image.new('RGB',(cols*w,rows*h))
        for i,im in enumerate(items[start:start+cols*rows]):image.paste(im,((i%cols)*w,(i//cols)*h))
        image.save(dest/f'{prefix}_{start//(cols*rows):02d}.png')


def tracked_appearance(previous,current):
    """Same baseline flow and valid samples for both arms; diagnostic, not GT."""
    a,b,x,y=previous;c,d,xx,yy=current
    def small(im):return cv2.resize(im,(270,480),interpolation=cv2.INTER_AREA).astype(np.float32)
    aa,bb,cc,dd=map(small,[a,b,c,d])
    ga=cv2.cvtColor(aa.astype(np.uint8),cv2.COLOR_RGB2GRAY);gc=cv2.cvtColor(cc.astype(np.uint8),cv2.COLOR_RGB2GRAY)
    flow=cv2.calcOpticalFlowFarneback(ga,gc,None,.5,3,21,4,7,1.5,0)
    back=cv2.calcOpticalFlowFarneback(gc,ga,None,.5,3,21,4,7,1.5,0)
    sy,sx=np.indices(ga.shape,dtype=np.float32);mx=sx+flow[...,0];my=sy+flow[...,1]
    reverse=cv2.remap(back,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    valid=(np.linalg.norm(flow+reverse,axis=-1)<.5)&(mx>=1)&(my>=1)&(mx<269)&(my<479)
    # Fixed head-context window, common to both arms. Does not score missing mesh.
    valid[:40]=False;valid[330:]=False
    for left,right in [(x,xx),(y,yy)]:
        left=cv2.resize(left,(270,480),interpolation=cv2.INTER_NEAREST)
        right=cv2.resize(right,(270,480),interpolation=cv2.INTER_NEAREST)
        tracked=cv2.remap(right,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
        valid&=(left<62)&(tracked<62)
    result={'common_samples':int(valid.sum())}
    for name,left,right in [('baseline',aa,cc),('candidate',bb,dd)]:
        tracked=cv2.remap(right,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        delta=np.abs(tracked-left).mean(2)[valid]
        result[name]=dict(mean_rgb_step=float(delta.mean()),p95_rgb_step=float(np.quantile(delta,.95)))
    return result


def frame_pair(record):
    frame=record['frame'];old=BASE/'frames'/frame;new=Path(record['output'])
    receipt=read(old/'complete.json');assert sha(old/'complete.json')==record['baseline_complete_sha256']
    for n,h in receipt['hashes'].items():assert sha(old/n)==h,n
    r=read(new/'result.json');assert r['request_sha256']==sha(new/'request.json')
    for n,h in r['hashes'].items():assert sha(new/n)==h,n
    assert r['baseline']==str(old) and r['depth_sha256']==sha(old/'target_depth.npz')
    if record['reuse']:assert sha(new/'result.json')==record['reused_result_sha256']
    a=np.array(Image.open(old/'frame.png'));b=np.array(Image.open(new/'frame.png'))
    x=np.rot90(np.array(Image.open(old/'source_ids.png')));y=np.rot90(np.array(Image.open(new/'source_ids.png')))
    np.testing.assert_array_equal(a[x==y],b[x==y])
    return a,b,x,y,r


def review(clip):
    cv2.setNumThreads(2);q=verify();inventory={x['frame']:x for x in q['inventory']}
    # Review a completed clip while independent frame workers keep rendering.
    for frame in CLIPS[clip]:
        if not inventory[frame]['reuse']:
            state=read(ROOT/'states'/(frame+'.json'))
            assert state['terminal'] and state['exit_code']==0
    dest=ROOT/'review'/clip;assert not dest.exists();dest.mkdir(parents=True)
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-f','rawvideo','-pixel_format','rgb24',
        '-video_size','2160x1920','-framerate','24','-i','pipe:0','-an','-c:v','libx264','-threads','4',
        '-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'matched.mp4')]
    records=[];details=[];nose=[];overviews=[];previous=None;bindings={}
    with (dest/'encode.log').open('x') as log:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for frame in CLIPS[clip]:
                inv=inventory[frame];a,b,x,y,r=frame_pair(inv);change=np.any(a!=b,2)
                rec=dict(frame=frame,source_changes=int((x!=y).sum()),rgb_changes=int(change.sum()),
                    new_black=int(((a.max(2)>0)&(b.max(2)==0)).sum()),
                    candidate_source_counts=np.bincount(y[y<62],minlength=62).tolist(),
                    baseline_source_counts=np.bincount(x[x<62],minlength=62).tolist(),
                    result_sha256=sha(Path(inv['output'])/'result.json'))
                now=(a,b,x,y)
                if previous is not None:
                    rec['motion_diagnostic']=temporal_sources(previous,now)
                    rec['appearance_diagnostic']=tracked_appearance(previous,now)
                previous=now
                delta=np.abs(a.astype(np.float32)-b.astype(np.float32)).mean(2)
                score=cv2.boxFilter(delta,cv2.CV_32F,(256,256),normalize=False);score[:128]=0;score[-128:]=0;score[:,:128]=0;score[:,-128:]=0
                yy,xx=np.unravel_index(score.argmax(),score.shape);box=[int(xx)-128,int(yy)-128,int(xx)+128,int(yy)+128]
                rec['posthoc_change_box']=box;records.append(rec)
                details.append(pair(a[box[1]:box[3],box[0]:box[2]],b[box[1]:box[3],box[0]:box[2]],frame))
                if clip=='nose_seam':nose.append(pair(a[560:880,775:1031],b[560:880,775:1031],frame))
                overviews.append(pair(np.array(Image.fromarray(a).resize((180,320))),np.array(Image.fromarray(b).resize((180,320))),frame))
                proc.stdin.write(np.concatenate([a,b],axis=1).tobytes())
                for p in [BASE/'frames'/frame/'complete.json',Path(inv['output'])/'request.json',Path(inv['output'])/'result.json']:
                    bindings[str(p)]=sha(p)
        finally:
            proc.stdin.close();code=proc.wait()
        assert code==0
    sheets(overviews,dest,'overview',4,3);sheets(details,dest,'largest_change_native',1,4)
    if nose:sheets(nose,dest,'nose_native',1,4)
    import av
    with av.open(str(dest/'matched.mp4')) as c:
        s=c.streams.video[0];assert (s.width,s.height,s.frames)==(2160,1920,24) and str(s.average_rate)=='24'
        assert sum(1 for _ in c.decode(video=0))==24
    write(dest/'result.json',dict(clip=clip,records=records,input_hashes=bindings,
        images={str(p):sha(p) for p in dest.glob('*.png')},video_sha256=sha(dest/'matched.mp4'),
        encode_command=command,script_sha256=sha(__file__),visual_status='pending',production_promoted=False,
        diagnostic_counts_not_quality_metrics=True,flow_from_baseline_only=True))
    print('review ready',clip,'24 dynamic frames',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--clip',choices=CLIPS,required=True);a=p.parse_args();review(a.clip)
