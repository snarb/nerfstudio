"""Train-only calibrated face-landmark consistency gate; never use model z as depth.

stage/analyze/audit: reconstruction .venv; infer: isolated mediapipe_hand_env.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

OUT=Path('/mnt/data/dec5_multiview_face_prior')
FRAMES=['001193','001195']
VALIDATION_PREFIXES=['E004_B','G004_B','I004_B','K004_B','M004_B','F004_D','H004_D','L004_D']
CROP=(0,400,1080,1550)
MODEL_URL='https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
# Explicit anatomical index sets; lower face is not pooled with eyes/nose.
JAW=[234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454]
CHEEK=[50,101,205,206,207,187,123,116,117,118,119,120,100,36,126,142,203,
       280,330,425,426,427,411,352,345,346,347,348,349,329,266,355,371,423]
INTERIOR=[1,4,5,6,19,94,97,98,2,168,195,197,33,133,362,263,61,291,0,17]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.tmp')
    tmp.write_text(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def portrait_to_native(xy):
    xy=np.asarray(xy,float)
    if xy.shape[-1]!=2 or not np.isfinite(xy).all():raise ValueError('Expected finite xy')
    return np.stack((1919-xy[...,1],xy[...,0]),axis=-1)

def stage():
    from calibrated_depth_witness import load_images
    from joint_temporal_texture import cameras,HELD_CAMERAS,ROOT
    if OUT.exists():raise ValueError('Use a fresh root')
    OUT.mkdir(parents=True);records=[]
    for frame in FRAMES:
        rows,mesh,meta=cameras(frame);assert not ({r['physical_camera'] for r in rows}&HELD_CAMERAS)
        images,_,receipt=load_images(frame);dest=OUT/frame;dest.mkdir();inputs=[]
        panel=Image.new('RGB',(8*216,8*245));draw=ImageDraw.Draw(panel)
        for i,row in enumerate(rows):
            name=row['physical_camera'];rgb=images[name];assert rgb.dtype==np.uint8
            image=Image.fromarray(np.rot90(rgb));path=dest/(name+'.png');image.crop(CROP).save(path)
            panel.paste(image.crop(CROP).resize((216,220)),((i%8)*216,(i//8)*245+20));draw.text(((i%8)*216+3,(i//8)*245+3),name[:9],fill='white')
            inputs.append(dict(camera=row,path=str(path),sha256=sha(path)))
        panel.save(dest/'train_head_preview.png')
        save(dest/'input.json',dict(frame=frame,inputs=inputs,display_receipt=receipt,mesh=str(mesh),mesh_sha256=sha(mesh),
            metadata=str(meta),metadata_sha256=sha(meta),crop=CROP,portrait_rotation='np.rot90 native landscape',heldout_loaded=False))
        records.append(dict(frame=frame,input_sha256=sha(dest/'input.json')));print('staged',frame,len(inputs),'train views',flush=True)
    save(OUT/'stage.json',dict(frames=records,script_sha256=sha(__file__),parameters_path=str(ROOT/'parameters.npz')))

def infer():
    import mediapipe as mp
    import importlib.metadata
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION,FACEMESH_FACE_OVAL
    model=OUT/'face_landmarker.task'
    if (OUT/'inference.json').exists():raise ValueError('Keep frozen inference')
    request=dict(frames=FRAMES,validation_prefixes=VALIDATION_PREFIXES,crop=CROP,model_url=MODEL_URL,model_sha256=sha(model),
        mediapipe_version=mp.__version__,packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        mode='IMAGE',delegate='CPU',num_faces=1,min_detection=.5,min_presence=.5,
        image_uploads=False,heldout_used=False,model_z_or_transform_used_as_metric=False,
        groups=dict(jaw=JAW,cheek=CHEEK,interior=INTERIOR,face_without_iris=list(range(468))),
        triangulation=dict(minimum_fit_views=3,robust_scale_pixels=2.,minimum_pair_parallax_degrees=1.,
            arms=['all_fit_robust','train_consensus'],consensus_inlier_pixels=3.,maximum_candidate_pairs=256,
            minimum_consensus_fraction=.3,seed=73),
        gate=dict(minimum_validation_cameras=2,minimum_fit_views=3,maximum_fit_median_pixels=2.,
            maximum_validation_median_pixels=2.,maximum_validation_p90_pixels=4.,minimum_passing_fraction_each_lower_face_group=.8),
        scope='Correspondence agreement of predictions, not ground-truth landmark accuracy',
        stage_sha256=sha(OUT/'stage.json'),script_sha256=sha(__file__))
    save(OUT/'request.json',request)
    options=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model),delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=1,min_face_detection_confidence=.5,min_face_presence_confidence=.5,
        output_face_blendshapes=False,output_facial_transformation_matrixes=False)
    records=[];started=time.monotonic()
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for frame in FRAMES:
            spec=read(OUT/frame/'input.json');folder=OUT/frame/'landmarks';folder.mkdir()
            for item in spec['inputs']:
                assert sha(item['path'])==item['sha256'];name=item['camera']['physical_camera'];rgb=np.array(Image.open(item['path']).convert('RGB'))
                prediction=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb));entry=dict(frame=frame,camera=name,detected=len(prediction.face_landmarks),source_sha256=item['sha256'])
                image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
                if len(prediction.face_landmarks)==1:
                    landmarks=prediction.face_landmarks[0];xy=np.array([[p.x*rgb.shape[1]+CROP[0],p.y*rgb.shape[0]+CROP[1]] for p in landmarks]);entry['portrait_xy']=xy.tolist()
                    entry['model_z_unused']=[p.z for p in landmarks]
                    local=xy-np.array(CROP[:2])
                    for a,b in FACEMESH_TESSELATION:draw.line([tuple(local[a]),tuple(local[b])],fill=(35,130,35),width=1)
                    for group,color in [(JAW,'red'),(CHEEK,'yellow'),(INTERIOR,'cyan')]:
                        for k in group:
                            x,y=local[k];draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
                    facebox=(max(CROP[0],int(xy[:468,0].min())-30),max(CROP[1],int(xy[:468,1].min())-30),min(CROP[2],int(xy[:468,0].max())+31),min(CROP[3],int(xy[:468,1].max())+31))
                    entry['native_review_box']=facebox;image.crop(tuple(v-CROP[i%2] for i,v in enumerate(facebox))).save(folder/(name+'.png'))
                else:image.resize((540,550)).save(folder/(name+'.png'))
                records.append(entry)
            print(frame,'detected',sum(r['detected']==1 for r in records if r['frame']==frame),'/62',flush=True)
    save(OUT/'topology.json',dict(edges=sorted([list(e) for e in FACEMESH_TESSELATION]),oval=sorted([list(e) for e in FACEMESH_FACE_OVAL]),
        source='Installed mediapipe0.10.21 face_mesh_connections',used_as_metric_geometry=False))
    save(OUT/'inference.json',dict(records=records,request_sha256=sha(OUT/'request.json'),elapsed_seconds=time.monotonic()-started,topology_sha256=sha(OUT/'topology.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['stage','infer']);p.add_argument('--output',type=Path,default=OUT)
    a=p.parse_args();OUT=a.output;globals()[a.command]()
