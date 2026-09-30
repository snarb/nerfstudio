"""Validate all real-time model receipts and encode exactly two seconds at30FPS."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from PIL import Image,ImageDraw
import imageio_ffmpeg
from prepare_luster_video import write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();root=args.root
    meta=json.loads((root/'manifest.json').read_text());frames=meta['frames'];fps=meta['fps']
    if len(frames)!=60 or fps!=30 or [int(f) for f in frames]!=list(range(int(frames[0]),int(frames[0])+60)):
        raise ValueError('Expected60 consecutive real frames at30FPS')
    receipts=[];rows=[]
    for frame in frames:
        receipt=json.loads((root/'video_frames/receipts'/f'{frame}.json').read_text())
        snapshot=json.loads((root/'snapshots'/f'{frame}.json').read_text())
        if receipt['frame']!=frame or receipt['checkpoint_sha256']!=snapshot['archived_checkpoint']['sha256']:raise ValueError('Frame/model identity mismatch')
        if receipt['cameras_sha256']!=sha(root/'video_cameras.json'):raise ValueError('Camera changed during sequence')
        for kind,item in receipt['images'].items():
            if sha(item['path'])!=item['sha256']:raise ValueError('Rendered PNG changed')
            with Image.open(item['path']) as im:
                if im.size!=(1080,1920):raise ValueError('Unexpected video dimensions')
                im.verify()
        receipts.append(receipt)
        result=json.loads(Path(snapshot['selection']).read_text())
        for view in result['per_view']:
            row=dict(frame=frame,split=view['split'],camera=view['physical_camera'],step=result['step'],
                     evaluation_data_revision=(result.get('data_revision') or {}).get('id','original_prepared'),
                     **{k:view[k] for k in ['psnr','ssim','lpips','foreground_psnr']})
            for label,values in view['rois'].items():
                for key in ['psnr','ssim','lpips','foreground_fraction']:row[f'{label}_{key}']=values[key]
            rows.append(row)
    if len({r['field_parameters_sha256'] for r in receipts})!=60:raise ValueError('Sequence does not contain60 distinct learned fields')
    out=root/'final_video';out.mkdir(exist_ok=True)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();videos={}
    # This host's /usr/local/lib contains an incompatible libmpg123. Restrict
    # only the system ffprobe child to the matching distribution libraries.
    probe_env=os.environ.copy();probe_env.pop('LD_PRELOAD',None)
    probe_env['LD_LIBRARY_PATH']='/usr/lib/x86_64-linux-gnu'
    for kind in ['body','detail']:
        destination=out/f'{kind}_30fps.mp4'
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-framerate','30','-start_number',str(int(frames[0])),
                        '-i',str(root/'video_frames'/kind/'%06d.png'),'-frames:v','60','-c:v','libx264','-crf','16','-threads','2',
                        '-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709','-pix_fmt','yuv420p',
                        '-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
                        '-movflags','+faststart',str(destination)],check=True)
        subprocess.run([ffmpeg,'-v','error','-i',str(destination),'-f','null','-'],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                         '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration,color_space,color_range,color_primaries,color_transfer','-of','json',str(destination)],text=True,env=probe_env))['streams'][0]
        if probe['nb_read_frames']!='60' or probe['avg_frame_rate']!='30/1' or abs(float(probe['duration'])-2)>1e-6:raise ValueError('Encoded timeline differs')
        if probe['color_space']!='bt709' or probe['color_range']!='tv':raise ValueError('Encoded color metadata differs')
        sheet=Image.new('RGB',(10*180,6*342))
        for i,frame in enumerate(frames):
            im=Image.open(root/'video_frames'/kind/f'{frame}.png');im.thumbnail((180,320));x=(i%10)*180;y=(i//10)*342
            sheet.paste(im,(x,y+22));ImageDraw.Draw(sheet).text((x+2,y+2),frame,fill='white')
        sheet.save(out/f'{kind}_contact.jpg',quality=94)
        videos[kind]=dict(path=str(destination),sha256=sha(destination),probe=probe,decoded=True)
    write(out/'manifest.json',dict(frames=frames,fps=30,duration_seconds=2,models=receipts,videos=videos,
                                  visual_review='pending; encoding validation does not certify image quality'))
    print(json.dumps(videos,indent=2))


if __name__=='__main__':main()
