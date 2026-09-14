"""Verify camera-only pilot hashes/inventory and decode real MP4 review panels."""
from pathlib import Path
import argparse,json,os,subprocess
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json


def audit(output):
    env=dict(os.environ,LD_PRELOAD='/lib/x86_64-linux-gnu/libmpg123.so.0');reports={}
    for name in ['old','3x3','4x4']:
        root=output/name;request=read(root/'request.json');result=read(root/'result.json');hashes=read(root/'frame_hashes.json')
        if sha(root/'video.mp4')!=result['video_sha256']:raise ValueError('Changed pilot video')
        count=len(request['path'])
        if [r['index'] for r in hashes]!=list(range(count)):raise ValueError('Pilot frame inventory mismatch')
        for item in hashes:
            if sha(root/'frames'/f'{item["index"]:05d}.png')!=item['sha256']:raise ValueError('Changed pilot frame')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(root/'video.mp4')],env=env,text=True))['streams'][0]
        if int(probe['nb_read_frames'])!=count or probe['width']!=540 or probe['height']!=960:raise ValueError('Encoded inventory mismatch')
        if abs(float(probe['duration'])-request['report']['duration_seconds'])>1e-3:raise ValueError('Encoded duration mismatch')
        decoded=root/'decoded';decoded.mkdir(exist_ok=True)
        indices=[0,count//4,count//2,3*count//4];expression='+'.join(f'eq(n\\,{i})' for i in indices)
        subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(root/'video.mp4'),'-vf',f'select={expression},crop=380:420:120:240','-vsync','vfr',str(decoded/'%03d.png')],env=env,check=True)
        paths=sorted(decoded.glob('*.png'))
        if len(paths)!=4:raise ValueError('Decoded review inventory mismatch')
        panel=Image.new('RGB',(760,888));draw=ImageDraw.Draw(panel)
        for i,p in enumerate(paths):
            x,y=i%2*380,i//2*444;panel.paste(Image.open(p),(x,y+24));draw.text((x+4,y+4),f'{name}: frame {indices[i]}',fill='white')
        panel.save(root/'encoded_detail.png')
        reports[name]={'ffprobe':probe,'video_sha256':sha(root/'video.mp4'),'verified_png_count':count,
                       'decoded_detail_sha256':sha(root/'encoded_detail.png'),'request_sha256':sha(root/'request.json')}
    atomic_json(output/'integrity_audit.json',{'status':'pass','variants':reports,'actual_source_time_count':1,'no_new_face_metrics':True,'audit_script_sha256':sha(__file__)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
