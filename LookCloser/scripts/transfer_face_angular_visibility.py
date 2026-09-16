"""Same frozen skin-consensus/angular policy on three other real video times."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
import study_face_angular_visibility as pilot
from stage_face_visibility_transfer import ROOT as INPUTS,FRAMES
from study_multiview_face_prior import read,save,sha


def prepare(frame):
    sem=INPUTS/frame;sq=read(sem/'request.json');complete=read(sem/'complete.json')
    assert complete['request_sha256']==sha(sem/'request.json')
    parent=read(pilot.base.BASE/'request.json');record=next(r for r in parent['inventory'] if r['frame_id']==frame)
    expected={x['physical_camera']:x['sha256'] for x in next(r for r in parent['source_rows'] if Path(r['source_dataset']).name==frame)['source_images']}
    rows,_,_=pilot.base.cameras(frame);spec=record['source_masks'];maskroot=Path(spec['root'])
    names=read(maskroot/'cameras.json');foreground=np.load(maskroot/'masks.npz')['masks'];bindings={}
    for n,k in [('masks.npz','masks_sha256'),('cameras.json','cameras_sha256'),('complete.json','complete_sha256')]:
        assert sha(maskroot/n)==spec[k];bindings[str(maskroot/n)]=spec[k]
    out=pilot.ROOT/frame/'consensus';assert not out.exists();out.mkdir(parents=True)
    masks=[];canvas=Image.new('RGB',(8*180,8*230));draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(rows):
        name=row['physical_camera'];r=next(x for x in complete['outputs'] if x['camera']==name);s=next(x for x in sq['records'] if x['camera']==name)
        assert s['source_sha256']==expected[name]
        for p,h in [(r['path'],r['sha256']),(s['input_path'],s['input_sha256'])]:assert sha(p)==h;bindings[p]=h
        confidence=np.load(r['path'])['confidence'];strict=distance_transform_edt(confidence>=230)>=4
        full=np.zeros((1920,1080),bool);x0,y0,x1,y1=sq['crop'];full[y0:y1,x0:x1]=strict
        masks.append(np.rot90(full,-1)&(foreground[names.index(name)]>0))
        rgb=np.array(Image.open(s['input_path']));rgb[~strict]=(rgb[~strict]*.3).astype(np.uint8)
        canvas.paste(Image.fromarray(rgb).resize((180,210)),((i%8)*180,(i//8)*230+20));draw.text(((i%8)*180+2,(i//8)*230+2),name[:9],fill='white')
    canvas.save(out/'mask_overview.png');np.savez_compressed(out/'face_masks.npz',masks=np.stack(masks))
    q=read(pilot.ROOT/'001123/consensus/request.json')
    q['input_hashes'].update(bindings)
    for p in [Path(__file__).resolve(),sem/'request.json',sem/'complete.json',pilot.ROOT/'001123/consensus/request.json']:
        q['input_hashes'][str(p)]=sha(p)
    q.update(frame=frame,face_masks_sha256=sha(out/'face_masks.npz'),
        transfer_from='001123',policy_parameters_unchanged=True)
    save(out/'request.json',q);print('prepared',frame,'skin pixels',int(np.stack(masks).sum()),flush=True)


def render(frame):
    out,generated=pilot.configure('consensus',frame)
    q=read(out/'request.json');assert q['frame']==frame and q['generated_render_sha256']==generated
    pilot.base.render()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render']);p.add_argument('--frame',choices=FRAMES,required=True);a=p.parse_args()
    pilot.base.torch.set_num_threads(2)
    with pilot.base.torch.inference_mode():globals()[a.action](a.frame)
