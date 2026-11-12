"""Encode the reviewed PNG sequence as H.264, with decode and timing verification."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import cv2
import numpy as np


def encode(out):
    out=Path(out);sys.path.insert(0,str(out/'codec'))
    import imageio_ffmpeg
    path=json.loads((out/'path.json').read_text())
    target=out/'Marshawn908_1267_flying_strict_BC_slowmo_24fps.mp4'
    temp=target.with_name(target.stem+'.h264.partial.mp4')
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
        '-s','1920x1080','-r','24','-i','pipe:0','-an','-c:v','libx264','-preset','medium','-crf','16',
        '-threads','8','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
    with (out/'encode.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for row in path['records']:
                file=out/'neural'/f'{row["index"]:03}'/'strict.png' if row['kind']=='nerf' else out/'gt'/f'{row["index"]:03}.png'
                im=cv2.imread(str(file));assert im is not None and im.shape==(1080,1920,3)
                process.stdin.write(im.tobytes())
        finally:process.stdin.close()
        assert process.wait(timeout=120)==0
    cap=cv2.VideoCapture(str(temp));count=0;max_error=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(1080,1920,3);count+=1
    fps=cap.get(cv2.CAP_PROP_FPS);cap.release();assert count==84 and fps==24
    temp.replace(target)
    receipt=json.loads((out/'delivery_receipt.json').read_text())
    receipt.update(video_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),codec='H.264',pixel_format='yuv420p',crf=16,
        decoded_frames=count,fps=fps,duration_seconds=count/fps)
    (out/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('H264_VERIFIED',count,fps,receipt['video_sha256'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);encode(p.parse_args().output)
