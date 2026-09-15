"""Isolated MediaPipe anatomical landmark canary on immutable train observations.

Run with the separate mediapipe_hand_env, not the reconstruction environment.
Landmarks are predictions, not measured anatomy or a hand surface mesh.
"""
from pathlib import Path
import argparse,hashlib,json,os,importlib.metadata
import numpy as np
from PIL import Image,ImageDraw
import mediapipe as mp

OBS=Path('/mnt/data/dec5_wrist_observations')
OUT=Path('/mnt/data/dec5_hand_landmark_prior')
MODEL=Path('/home/brans/lookcloser_temp/mediapipe_hand_models/hand_landmarker.task')
NAMES=['G004_A005_121071','G004_B005_1210FG','G004_C005_121037','H004_A005_1210M6','H004_B005_1210EL','H004_C005_1210SZ']
TIMES=['001029','001031','001033','001035','001037'];CROP=(0,1100,640,1920)
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
       (9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def atomic(path,payload):
    tmp=path.with_name('.'+path.name+'.tmp');tmp.write_text(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    request=dict(times=TIMES,cameras=NAMES,crop=CROP,model=str(MODEL),model_sha256=sha(MODEL),
        model_url='https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
        mediapipe_version=mp.__version__,mode='IMAGE',num_hands=1,detection_confidence=.5,presence_confidence=.5,
        heldout_used=False,source_rgb_changed=False,prediction_not_measured_geometry=True,
        script_sha256=sha(__file__),packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()})
    atomic(output/'request.json',request)
    options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=1,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5)
    records=[]
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for frame in TIMES:
            folder=output/frame;folder.mkdir()
            for name in NAMES:
                path=OBS/frame/(name+'.png');receipt=json.loads((OBS/frame/'result.json').read_text())
                expected=next(r for r in receipt['records'] if r['camera']['physical_camera']==name)
                if sha(path)!=expected['image_sha256']:raise ValueError('Changed source observation')
                image=Image.open(path).convert('RGB');crop=image.crop(CROP)
                prediction=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.array(crop)))
                entry=dict(frame=frame,camera=name,source=str(path),source_sha256=sha(path),detected=len(prediction.hand_landmarks),hands=[])
                draw=ImageDraw.Draw(image)
                for hand,world,category in zip(prediction.hand_landmarks,prediction.hand_world_landmarks,prediction.handedness):
                    xy=np.array([[p.x*crop.width+CROP[0],p.y*crop.height+CROP[1]] for p in hand])
                    entry['hands'].append(dict(portrait_xy=xy.tolist(),world_xyz=[[p.x,p.y,p.z] for p in world],
                        handedness=category[0].category_name,handedness_score=float(category[0].score)))
                    for a,b in EDGES:draw.line([tuple(xy[a]),tuple(xy[b])],fill=(0,255,0),width=2)
                    for i,p in enumerate(xy):
                        draw.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),fill=(255,60,40));draw.text((p[0]+4,p[1]-5),str(i),fill='yellow')
                image.crop((0,1250,500,1920)).save(folder/(name+'_landmarks.png'));records.append(entry)
            print(frame,'detections',[r['detected'] for r in records[-6:]],flush=True)
    atomic(output/'result.json',dict(records=records,request_sha256=sha(output/'request.json'),visual_status='pending',geometry_changed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);run(p.parse_args().output)
