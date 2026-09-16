"""Stage/infer train-only face support at three additional actual video times."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from build_train_hair_semantics import read,write,sha

ROOT=Path('/mnt/data/dec5_face_angular_transfer_inputs')
FRAMES=['001083','001119','001127']
CROP=[0,350,1080,1600]


def stage(frame):
    from joint_temporal_texture import cameras,exr,display,ROOT as COLOR,HELD_CAMERAS
    parent=read('/mnt/data/dec5_cinematic_wide_spiral_v3/wide_spiral_free/request.json')
    sources=next(x for x in parent['source_rows'] if Path(x['source_dataset']).name==frame)
    expected={x['physical_camera']:x['sha256'] for x in sources['source_images']}
    model=read('/mnt/data/dec5_train_face_support_001123/request.json')['model']
    assert sha(model['file'])==model['sha256']
    dest=ROOT/frame;assert not dest.exists();(dest/'inputs').mkdir(parents=True)
    params=np.load(COLOR/'parameters.npz')['log_gain'];gain=np.exp(params-params.mean(0,keepdims=True))
    exposure=read(COLOR/'exposure.json')['fixed_exposure_gain'];rows,_,_=cameras(frame)
    assert len(rows)==62 and not {r['physical_camera'] for r in rows}&HELD_CAMERAS
    records=[]
    for ci,row in enumerate(rows):
        camera=row['physical_camera'];source=row['file_path'];assert sha(source)==expected[camera]
        rgb=np.rint(display(exr(source)*gain[ci],exposure)*255).clip(0,255).astype(np.uint8)
        out=dest/'inputs'/(camera+'.png');Image.fromarray(np.rot90(rgb)).crop(CROP).save(out,compress_level=1)
        records.append(dict(camera=camera,input_path=str(out),input_sha256=sha(out),source_path=source,source_sha256=sha(source)))
    write(dest/'request.json',dict(frame=frame,crop=CROP,model=model,records=records,
        script_sha256=sha(__file__),profiles_sha256=sha(COLOR/'parameters.npz'),exposure_sha256=sha(COLOR/'exposure.json'),
        heldout_used=False,geometry_or_target_used=False))
    print('staged',frame,len(records),flush=True)


def infer(frame):
    import mediapipe as mp
    dest=ROOT/frame;q=read(dest/'request.json');assert q['script_sha256']==sha(__file__)
    assert sha(q['model']['file'])==q['model']['sha256']
    out=dest/'confidence';assert not out.exists();out.mkdir();outputs=[]
    options=mp.tasks.vision.ImageSegmenterOptions(base_options=mp.tasks.BaseOptions(model_asset_path=q['model']['file'],
        delegate=mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_category_mask=False,output_confidence_masks=True)
    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as model:
        assert model.labels[3]=='face-skin'
        for i,r in enumerate(q['records']):
            assert sha(r['input_path'])==r['input_sha256'];rgb=np.array(Image.open(r['input_path']))
            value=model.segment(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)).confidence_masks[3].numpy_view().copy()
            assert value.shape==rgb.shape[:2] and np.isfinite(value).all()
            path=out/(r['camera']+'.npz');np.savez_compressed(path,confidence=np.rint(value.clip(0,1)*255).astype(np.uint8))
            outputs.append(dict(camera=r['camera'],path=str(path),sha256=sha(path)))
            print('inferred',frame,i+1,flush=True)
    write(dest/'complete.json',dict(request_sha256=sha(dest/'request.json'),outputs=outputs,mediapipe_version=mp.__version__))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['stage','infer']);p.add_argument('--frame',choices=FRAMES,required=True);a=p.parse_args()
    globals()[a.action](a.frame)
