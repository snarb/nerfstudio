"""Audit matched clip geometry and expose source-switch/appearance side effects.

Optical flow is posthoc baseline-only diagnostics, never prediction input. Its
cycle-consistent switch counts are not a temporal quality score or ground truth.
"""
from pathlib import Path
import argparse
import subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from study_temporal_source_retention import ROOT,BASE,CLIPS
from joint_temporal_texture import read,sha,atomic_json
from review_jaw_repair_transfer import verified_image


def exact_geometry(base,candidate):
    for name,key in [('target_depth.npz','depth'),('face_source_labels.npy',None)]:
        a=np.load(base/name);b=np.load(candidate/name)
        np.testing.assert_array_equal(a[key] if key else a,b[key] if key else b)


def draw_pair(a,b,text):
    assert a.shape==b.shape
    h,w=a.shape[:2];out=Image.new('RGB',(2*w,h+24));out.paste(Image.fromarray(a),(0,24));out.paste(Image.fromarray(b),(w,24))
    d=ImageDraw.Draw(out);d.text((3,4),text+' original',fill='white');d.text((w+3,4),'relative .5',fill='white')
    return out


def temporal_sources(previous,current):
    """Track only with original RGB; report both arms on identical valid pixels."""
    a,_,oldids,newids=previous;b,_,old2,new2=current
    def gray(im):return cv2.cvtColor(cv2.resize(im,(270,480),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
    ga,gb=gray(a),gray(b)
    flow=cv2.calcOpticalFlowFarneback(ga,gb,None,.5,3,21,4,7,1.5,0)
    back=cv2.calcOpticalFlowFarneback(gb,ga,None,.5,3,21,4,7,1.5,0)
    y,x=np.indices(ga.shape,dtype=np.float32);mx=x+flow[...,0];my=y+flow[...,1]
    reverse=cv2.remap(back,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    valid=(np.linalg.norm(flow+reverse,axis=-1)<.5)&(mx>=1)&(my>=1)&(mx<269)&(my<479)
    # Fixed head/hand diagnostic window shared by both arms. Background/no-source
    # pixels are excluded using both source-ID buffers, not candidate RGB error.
    valid[:40]=False;valid[390:]=False
    result={};common=valid.copy();pairs=[]
    for left,right in [(oldids,old2),(newids,new2)]:
        left=cv2.resize(left,(270,480),interpolation=cv2.INTER_NEAREST)
        right=cv2.resize(right,(270,480),interpolation=cv2.INTER_NEAREST)
        tracked=cv2.remap(right,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
        common&=(left<62)&(tracked<62);pairs.append((left,tracked))
    result['common_tracked_samples']=int(common.sum())
    for name,(left,right) in zip(['baseline','candidate'],pairs):result[name+'_source_switches']=int(((left!=right)&common).sum())
    return result


def review(clip):
    cv2.setNumThreads(2)
    dest=ROOT/'review'/clip;dest.mkdir(parents=True,exist_ok=False)
    frames=CLIPS[clip];records=[];details=[];nose=[];overviews=[];previous=None;bindings={};changes=[]
    binary=imageio_ffmpeg.get_ffmpeg_exe()
    command=[binary,'-nostdin','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','2160x1920',
        '-framerate','24','-i','pipe:0','-an','-c:v','libx264','-threads','4','-crf','16','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(dest/'matched_diagnostic.mp4')]
    with (dest/'encode.log').open('x') as log:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for frame in frames:
                old=BASE/'frames'/frame;new=ROOT/frame/'frames'/frame
                a,ar=verified_image(BASE,frame);b,br=verified_image(ROOT/frame,frame)
                for k in ['camera','mesh_sha256','source_cameras','fixed_exposure']:assert ar[k]==br[k],k
                assert not br['target_rgb_read'] and not br['rgb_averaging']
                exact_geometry(old,new)
                oldids=np.rot90(np.array(Image.open(old/'source_ids.png')));newids=np.rot90(np.array(Image.open(new/'source_ids.png')))
                rgb_delta=np.abs(a.astype(np.float32)-b.astype(np.float32)).mean(2)
                changed=np.any(a!=b,2);newblack=(a.max(2)>0)&(b.max(2)==0)
                record=dict(frame=frame,changed_rgb=int(changed.sum()),new_black=int(newblack.sum()),
                    changed_source=int((oldids!=newids).sum()),target_depth_equal=True,graph_labels_equal=True,
                    elapsed_seconds=br['elapsed_seconds'])
                now=(a,b,oldids,newids)
                if previous is not None:record['temporal_diagnostic']=temporal_sources(previous,now)
                previous=now;records.append(record)
                # Show the strongest 256px integrated difference region, not only
                # selected successes. Window location is posthoc, never a render gate.
                score=cv2.boxFilter(rgb_delta,cv2.CV_32F,(256,256),normalize=False)
                score[:128]=0;score[-128:]=0;score[:,:128]=0;score[:,-128:]=0
                y,x=np.unravel_index(np.argmax(score),score.shape);box=[int(x-128),int(y-128),int(x+128),int(y+128)]
                details.append(draw_pair(a[box[1]:box[3],box[0]:box[2]],b[box[1]:box[3],box[0]:box[2]],frame))
                changes.append(dict(frame=frame,box=box))
                if clip=='late_face':nose.append(draw_pair(a[560:880,775:1031],b[560:880,775:1031],frame+' nose'))
                overviews.append(draw_pair(np.array(Image.fromarray(a).resize((180,320))),np.array(Image.fromarray(b).resize((180,320))),frame))
                proc.stdin.write(np.concatenate([a,b],axis=1).tobytes())
                for folder in [old,new]:
                    receipt=read(folder/'complete.json')
                    for n,h in receipt['hashes'].items():bindings[str(folder/n)]=h
                    bindings[str(folder/'complete.json')]=sha(folder/'complete.json')
                    bindings[str(folder.parent.parent/'request.json')]=sha(folder.parent.parent/'request.json')
                print(clip,frame,record['changed_rgb'],'changed',record['new_black'],'new black',flush=True)
        finally:
            proc.stdin.close();code=proc.wait()
        assert code==0
    def sheets(items,kind,columns,rows):
        for start in range(0,len(items),columns*rows):
            w,h=items[0].size;sheet=Image.new('RGB',(columns*w,rows*h))
            for i,im in enumerate(items[start:start+columns*rows]):sheet.paste(im,((i%columns)*w,(i//columns)*h))
            sheet.save(dest/f'{kind}_{start//(columns*rows):02d}.png')
    sheets(overviews,'overview',4,3);sheets(details,'largest_change_native',1,4)
    if nose:sheets(nose,'nose_native',1,4)
    import av
    with av.open(str(dest/'matched_diagnostic.mp4')) as container:
        s=container.streams.video[0];assert (s.width,s.height,s.frames)==(2160,1920,24)
        assert sum(1 for _ in container.decode(video=0))==24
    atomic_json(dest/'result.json',dict(clip=clip,records=records,posthoc_change_crops=changes,
        input_hashes=bindings,images={str(p):sha(p) for p in dest.glob('*.png')},
        video_sha256=sha(dest/'matched_diagnostic.mp4'),encode_command=command,
        script_sha256=sha(__file__),visual_status='pending',promoted=False,
        optical_flow_not_ground_truth=True,temporal_counts_not_quality_metrics=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--clip',choices=list(CLIPS),required=True)
    review(p.parse_args().clip)
