"""Inspect a few actual-frame presentation dissolves before full-video assembly."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from compose_cinematic_train_ending import configuration,dissolve_alpha,blend_display,write_png
from joint_temporal_texture import read,sha,atomic_json
from review_jaw_repair_transfer import verified_image

INDICES=[118,122,125,126,149]


def preview(root):
    request,config=configuration(root);out=root/'train_transition_review';out.mkdir(exist_ok=False)
    assert read(root/'train_ending/request.json')==config
    sheet=Image.new('RGB',(5*324,606));draw=ImageDraw.Draw(sheet);records=[]
    for col,index in enumerate(INDICES):
        frame=request['ordered_frame_ids'][index];alpha=dissolve_alpha(index)
        path=root/'train_ending/frames'/frame/'frame.png'
        receipt=read(path.with_name('complete.json'));assert receipt['image_sha256']==sha(path)
        train=np.asarray(Image.open(path));raw=None
        if alpha<1:raw,_=verified_image(root,frame)
        image=train if alpha==1 else blend_display(raw,train,alpha)
        write_png(out/f'{frame}.png',image)
        sheet.paste(Image.fromarray(image).resize((324,576)),(324*col,30))
        draw.text((324*col+3,5),f'{frame} train opacity {alpha:.3f}',fill='white')
        records.append(dict(frame_id=frame,index=index,train_alpha=alpha,
            path=str(out/f'{frame}.png'),sha256=sha(out/f'{frame}.png')))
    sheet.save(out/'contact.png')
    atomic_json(out/'request.json',dict(records=records,raw_request_sha256=sha(root/'request.json'),
        ending_request_sha256=sha(root/'train_ending/request.json'),script_sha256=sha(__file__),
        compositor_sha256=sha(Path(__file__).with_name('compose_cinematic_train_ending.py')),
        contact_sha256=sha(out/'contact.png'),visual_status='pending',is_presentation_dissolve_not_texture_averaging=True))
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    preview(p.parse_args().root)
