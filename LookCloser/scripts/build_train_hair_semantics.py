"""Train-only hair/skin semantic evidence, never an output-image mask.

stage/review use the reconstruction environment; infer uses the existing
isolated MediaPipe environment. No reconstruction environment is modified.
The models are fallible semantic priors, not opacity/depth/GT estimators.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path('/mnt/data/dec5_train_hair_semantics')
FRAMES=['001083','001123']
CROP=(0,350,1080,1600)
MODELS={
    'multiclass':('selfie_multiclass_256x256.tflite','https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'),
    'hair':('hair_segmenter.tflite','https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite'),
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def stage():
    from joint_temporal_texture import cameras,exr,display,ROOT as COLOR,HELD_CAMERAS
    log_gain=np.load(COLOR/'parameters.npz')['log_gain'];gains=np.exp(log_gain-log_gain.mean(0,keepdims=True))
    exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']
    request=dict(frames=FRAMES,crop_portrait=CROP,rotation='native landscape -> numpy.rot90',
        full_native_shape=[1080,1920],model_inputs='calibrated display-domain train RGB only',
        script_sha256=sha(__file__),profiles_sha256=sha(COLOR/'parameters.npz'),exposure_sha256=sha(COLOR/'exposure.json'),
        models={k:dict(file=str(ROOT/'models'/f),sha256=sha(ROOT/'models'/f),download_url=url) for k,(f,url) in MODELS.items()},
        heldout_used=False,mesh_or_target_view_used=False,geometry_changed=False,
        unknown_outside_input_crop=True,probabilities_are_not_opacity=True)
    # JSON conversion gives the same list representation during resume.
    request=json.loads(json.dumps(request))
    if (ROOT/'request.json').exists():assert read(ROOT/'request.json')==request
    else:write(ROOT/'request.json',request)
    for frame in FRAMES:
        folder=ROOT/frame;folder.mkdir(exist_ok=True);inputs=folder/'inputs';inputs.mkdir(exist_ok=True)
        rows,_,_=cameras(frame);assert len(rows)==62 and not ({r['physical_camera'] for r in rows}&HELD_CAMERAS)
        records=[]
        for ci,row in enumerate(rows):
            name=row['physical_camera'];path=inputs/(name+'.png')
            rgb=np.rint(display(exr(row['file_path'])*gains[ci],exposure)*255).clip(0,255).astype(np.uint8)
            if path.exists():
                np.testing.assert_array_equal(np.array(Image.open(path)),np.asarray(Image.fromarray(np.rot90(rgb)).crop(CROP)))
            else:Image.fromarray(np.rot90(rgb)).crop(CROP).save(path,compress_level=1)
            records.append(dict(camera=name,input_path=str(path),input_sha256=sha(path),source_path=row['file_path'],source_sha256=sha(row['file_path'])))
        write(folder/'stage.json',dict(request_sha256=sha(ROOT/'request.json'),records=records))
        print('staged',frame,len(records),flush=True)


def infer():
    import mediapipe as mp
    options=mp.tasks.vision.ImageSegmenterOptions
    base=mp.tasks.BaseOptions
    models={k:mp.tasks.vision.ImageSegmenter.create_from_options(options(
        base_options=base(model_asset_path=str(ROOT/'models'/f),delegate=base.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,output_category_mask=False,output_confidence_masks=True))
        for k,(f,_) in MODELS.items()}
    q=read(ROOT/'request.json');assert sha(__file__)==q['script_sha256']
    for item in q['models'].values():assert sha(item['file'])==item['sha256']
    write(ROOT/'inference_environment.json',dict(mediapipe_version=mp.__version__,model_labels={k:v.labels for k,v in models.items()},delegate='CPU'))
    try:
        for frame in FRAMES:
            folder=ROOT/frame;staged=read(folder/'stage.json');assert staged['request_sha256']==sha(ROOT/'request.json')
            dest=folder/'predictions';dest.mkdir(exist_ok=True);results=[]
            for index,item in enumerate(staged['records']):
                name=item['camera'];receipt=dest/(name+'.json');path=dest/(name+'.npz')
                if receipt.exists():
                    r=read(receipt);assert r['request_sha256']==sha(ROOT/'request.json') and r['input_sha256']==item['input_sha256'] and sha(path)==r['output_sha256']
                    results.append(r);continue
                start=time.monotonic();assert sha(item['input_path'])==item['input_sha256']
                rgb=np.array(Image.open(item['input_path']).convert('RGB'));image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                multi=models['multiclass'].segment(image).confidence_masks
                hair=models['hair'].segment(image).confidence_masks
                assert len(multi)==6 and len(hair)==2
                # Preserve native crop dimensions, not the low-resolution logits.
                hair_prob=hair[1].numpy_view().copy()
                multi_hair=multi[1].numpy_view().copy()
                protected=multi[2].numpy_view()+multi[3].numpy_view()+multi[4].numpy_view()+multi[5].numpy_view()
                values=np.stack([hair_prob,multi_hair,protected])
                assert values.shape==(3,1250,1080) and np.isfinite(values).all()
                assert values.min()>=-1e-5 and values.max()<=1.00001
                np.savez_compressed(path,confidence=np.rint(values.clip(0,1)*255).astype(np.uint8))
                r=dict(camera=name,request_sha256=sha(ROOT/'request.json'),input_sha256=item['input_sha256'],
                    output_sha256=sha(path),channels=['hair_binary_model','hair_multiclass_model','skin_cloth_accessories_multiclass'],
                    quantization='round(probability*255)',seconds=time.monotonic()-start)
                write(receipt,r);results.append(r)
                write(ROOT/'progress.json',dict(stage='inference',frame=frame,camera=name,completed=index+1,pid=os.getpid()))
                print('infer',frame,index+1,name,round(r['seconds'],3),flush=True)
            write(folder/'complete.json',dict(request_sha256=sha(ROOT/'request.json'),stage_sha256=sha(folder/'stage.json'),records=results))
        write(ROOT/'progress.json',dict(stage='inference_complete',frames=FRAMES))
    finally:
        for model in models.values():model.close()


def review():
    names=['G004_C005_121037','H004_C005_1210SZ','J004_B005_1210GR','K004_B005_1210DS']
    for frame in FRAMES:
        root=ROOT/frame;receipt=read(root/'complete.json');records={r['camera']:r for r in receipt['records']}
        for name in names:
            rgb=np.array(Image.open(root/'inputs'/(name+'.png')))
            path=root/'predictions'/(name+'.npz');assert sha(path)==records[name]['output_sha256']
            maps=np.load(path)['confidence'].astype(np.float32)/255
            tile=Image.new('RGB',(4*360,442),(25,25,25));draw=ImageDraw.Draw(tile)
            images=[rgb,*[np.rint(np.repeat(p[...,None],3,2)*255).astype(np.uint8) for p in maps]]
            for i,(label,im) in enumerate(zip(['train RGB','hair binary','hair multiclass','protected skin/cloth'],images)):
                tile.paste(Image.fromarray(im).resize((360,417),Image.Resampling.LANCZOS),(i*360,25));draw.text((i*360+4,5),label,fill='white')
            dest=ROOT/'review';dest.mkdir(exist_ok=True);tile.save(dest/(frame+'_'+name+'.png'))
    print('review sheets ready',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['stage','infer','review']);a=p.parse_args()
    {'stage':stage,'infer':infer,'review':review}[a.action]()
