"""Evaluation-only native-pixel head videos; no training or checkpoint selection."""
from pathlib import Path
import os,subprocess,hashlib
import cv2,numpy as np
from run_dec5_sequence import read,write


def render(root,manifest,results,output,reference,fps=24):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
    reference=Path(reference);probability=np.load(reference)['probability']
    domain=(probability[...,0]>.5*65535)|(probability[...,1]>.5*65535)
    y,x=np.nonzero(domain)
    if len(x)<100:raise ValueError('Insufficient evaluation head reference')
    sx,sy=6144/domain.shape[1],3072/domain.shape[0]
    if abs(sx-sy)>1e-6:raise ValueError('Native head reference aspect ratio differs')
    # Constant evaluation-only crop; do not track or stabilize predictions.
    x0=max(0,(int(x.min()*sx)-128)//2*2);x1=min(6144,(int((x.max()+1)*sx)+130)//2*2)
    y0=max(0,(int(y.min()*sy)-128)//2*2);y1=min(3072,(int((y.max()+1)*sy)+130)//2*2)
    videos={}
    for mode,result in results.items():
        directory=output/mode;directory.mkdir(exist_ok=True);images=directory/'frames';images.mkdir(exist_ok=True)
        for index,row in enumerate(result['per_frame']):
            if row['frame']!=manifest['frames'][index]:raise ValueError('Head review order differs')
            data=root/'frames'/row['frame']/'data';meta=read(data/'transforms.json')
            frame=next(f for f in meta['frames'] if f['file_path'] in meta['val_filenames'])
            if frame['physical_camera']!=manifest['heldout_camera']:raise ValueError('Wrong evaluation camera')
            gt=cv2.imread(str(data/frame['file_path']))
            with np.load(Path(row['renders'])/'0.npz') as archive:rgb=archive['rgb']
            if gt.shape!=(3072,6144,3) or rgb.shape!=gt.shape:raise ValueError('Native review raster differs')
            target=np.rot90(gt[y0:y1,x0:x1]);prediction=np.rot90(np.uint8(np.clip(rgb[y0:y1,x0:x1],0,1)*255+.5)[...,::-1])
            canvas=np.zeros((target.shape[0]+32,target.shape[1]*2,3),np.uint8)
            canvas[32:,:target.shape[1]]=target;canvas[32:,target.shape[1]:]=prediction
            cv2.putText(canvas,'GT '+row['frame'],(8,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            cv2.putText(canvas,mode+' '+row['frame'],(target.shape[1]+8,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            cv2.imwrite(str(images/('%04d.jpg'%index)),canvas,[cv2.IMWRITE_JPEG_QUALITY,96])
            write(directory/'progress.json',dict(completed=index+1,total=len(result['per_frame'])))
        video=directory/'gt_vs_native_head.mp4';temporary=directory/'gt_vs_native_head.tmp.mp4'
        subprocess.run([os.environ.get('FFMPEG_BINARY','ffmpeg'),'-hide_banner','-loglevel','error','-y',
            '-framerate',str(fps),'-i',str(images/'%04d.jpg'),'-c:v','libx264','-threads','2','-crf','16',
            '-pix_fmt','yuv420p',str(temporary)],check=True)
        cap=cv2.VideoCapture(str(temporary));count=0
        while True:
            ok,image=cap.read()
            if not ok:break
            if count in {0,len(result['per_frame'])//2,len(result['per_frame'])-1}:
                cv2.imwrite(str(directory/('decoded_%04d.jpg'%count)),image)
            count+=1
        measured_fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        if count!=len(result['per_frame']) or abs(measured_fps-fps)>.01:raise ValueError('Native head video decode failed')
        temporary.replace(video)
        videos[mode]=dict(path=str(video),decoded_frames=count,display_fps=fps,
            sha256=hashlib.sha256(video.read_bytes()).hexdigest(),encoded_shape=[canvas.shape[1],canvas.shape[0]])
    result=dict(videos=videos,crop_native_xyxy=[x0,y0,x1,y1],source_resolution=[6144,3072],
        prediction_resized=False,crop_fixed_across_time=True,reference_used_for_evaluation_only=True,
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),checkpoint_selection_changed=False,
        capture_timing_calibrated=False,individual_hair_depth_ground_truth=False)
    write(output/'complete.json',result);return result
