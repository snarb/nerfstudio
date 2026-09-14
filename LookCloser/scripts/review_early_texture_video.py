"""Incremental native jaw/lipstick review using the exact parent flight crops."""
from pathlib import Path
import argparse
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from render_smooth_temporal_mesh_video import verify_request
from finalize_local_mesh_repair import verify_hashes
from run_early_texture_video import PARENT,OUTPUT


def sheets(output):
    request=verify_request(output);old=read(PARENT/'jaw_review/request.json')['inputs']
    if [r['frame_id'] for r in old]!=request['ordered_frame_ids']:raise ValueError('Wrong inherited crop inventory')
    root=output/'jaw_review';root.mkdir(exist_ok=True);records=[]
    for start in range(0,150,6):
        group=old[start:start+6]
        if not all((output/'frames'/r['frame_id']/'complete.json').exists() for r in group):continue
        inputs=[]
        for row in group:
            folder=output/'frames'/row['frame_id'];receipt=read(folder/'complete.json')
            if receipt['request_sha256']!=sha(output/'request.json'):raise ValueError('Wrong frame request')
            verify_hashes(folder,receipt['hashes'])
            inputs.append(dict(frame_id=row['frame_id'],path=str(folder/'frame.png'),sha256=sha(folder/'frame.png'),crop=row['crop']))
        path=root/f'jaw_{start:03d}.png';panel=Image.new('RGB',(1500,648));draw=ImageDraw.Draw(panel)
        for j,row in enumerate(inputs):
            x,y=j%3*500,j//3*324;panel.paste(Image.open(row['path']).crop(row['crop']),(x,y+24))
            draw.text((x+3,y+3),row['frame_id'],fill='white')
        panel.save(path);records.append(dict(start=start,path=str(path),sha256=sha(path),inputs=inputs))
    atomic_json(root/'sheets.json',dict(request_sha256=sha(output/'request.json'),parent_crop_sha256=sha(PARENT/'jaw_review/request.json'),
        scope='native jaw/lipstick only; no artifact-free whole-actor claim',sheets=records))
    print('ready native jaw sheets',len(records),flush=True)


def review(output,notes):
    root=output/'jaw_review';records=read(root/'sheets.json')['sheets'];written=[]
    for r in records:
        note=notes.get(str(r['start']))
        if not note:continue
        if sha(r['path'])!=r['sha256']:raise ValueError('Changed inspected sheet')
        for inp in r['inputs']:
            if sha(inp['path'])!=inp['sha256']:raise ValueError('Changed reviewed frame')
            written.append(dict(frame_id=inp['frame_id'],render_sha256=inp['sha256'],sheet_sha256=r['sha256'],notes=note,
                visual_status='reviewed_with_residual_artifacts',artifact_free=False))
    atomic_json(root/'visual_review.json',dict(records=written,reviewer='main_agent_actual_image_inspection',
        reviewed_frames=len(written),complete=len(written)==150,scope='native jaw/lipstick; overview review is separate'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['sheets','review']);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--notes',type=Path);a=p.parse_args()
    if a.action=='sheets':sheets(a.output)
    else:review(a.output,read(a.notes))
