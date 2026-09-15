"""Independent float64 separable replay of every prepared actual train frame.

Does not call the compositor, its sampler, or its display function. Equivalent
ending framings are first checked pixel-for-pixel across the four variants.
No mesh is loaded and no image-quality metric is reported.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,ROOT as COLOR,CALIBRATION,HELD_CAMERAS

VARIANTS=['locked_arc','free_arc','rising_arc','soft_diagonal']


def separable(linear,source,target):
    u=(np.arange(int(target['w']),dtype=float)+.5-target['cx'])*(source['fl_x']/target['fl_x'])+source['cx']-.5
    v=(np.arange(int(target['h']),dtype=float)+.5-target['cy'])*(source['fl_y']/target['fl_y'])+source['cy']-.5
    assert u.min()>=0 and v.min()>=0 and u.max()<linear.shape[1]-1 and v.max()<linear.shape[0]-1
    x=np.floor(u).astype(int);y=np.floor(v).astype(int);fu=u-x;fv=v-y
    horizontal=linear[:,x].astype(float)*(1-fu[None,:,None])+linear[:,x+1]*fu[None,:,None]
    return horizontal[y]*(1-fv[:,None,None])+horizontal[y+1]*fv[:,None,None]


def response(linear,gain,exposure):
    light=np.maximum(linear*gain,0)*exposure
    mapped=light/(1+light)
    srgb=np.where(mapped<=.0031308,mapped*12.92,1.055*mapped**(1/2.4)-.055)
    return np.rint(srgb*255).clip(0,255).astype(np.uint8)


def audit(base):
    calibration=read(CALIBRATION);name='H004_C005_1210SZ'
    source=next(r for r in calibration['frames'] if r['physical_camera']==name)
    train=sorted(r['physical_camera'] for r in calibration['frames'] if r['physical_camera'] not in HELD_CAMERAS)
    assert len(train)==62 and name in train
    parameters=np.load(COLOR/'parameters.npz')['log_gain'].astype(float)
    gain=np.exp(parameters-parameters.mean(0,keepdims=True))[train.index(name)]
    exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']
    requests={v:read(base/v/'request.json') for v in VARIANTS}
    records=[];bindings={str(CALIBRATION):sha(CALIBRATION),str(COLOR/'parameters.npz'):sha(COLOR/'parameters.npz'),
        str(COLOR/'exposure.json'):sha(COLOR/'exposure.json'),str(Path(__file__).resolve()):sha(__file__)}
    raw_sources=[]
    for index in range(118,150):
        frame=f'{899+2*index:06d}';local={};receipts={}
        for variant in VARIANTS:
            root=base/variant;folder=root/'train_ending/frames'/frame;receipt=read(folder/'complete.json')
            assert receipt['source_physical_camera']==name and receipt['frame_id']==frame and receipt['index']==index
            config=read(root/'train_ending/request.json');q=requests[variant]
            assert config['raw_request_sha256']==sha(root/'request.json')
            assert config['fixed_profiles_sha256']==sha(COLOR/'parameters.npz') and config['fixed_exposure_sha256']==sha(COLOR/'exposure.json')
            assert receipt['request_sha256']==sha(root/'train_ending/request.json') and sha(folder/'frame.png')==receipt['image_sha256']
            for path in [root/'request.json',root/'train_ending/request.json',folder/'complete.json',folder/'frame.png']:
                bindings[str(path)]=sha(path)
            manifest=next(r for r in q['source_rows'] if Path(r['source_dataset']).name==frame)
            actual=next(r for r in manifest['source_images'] if r['physical_camera']==name)
            assert 'frame_train_' in actual['file_path']
            assert Path(receipt['source_exr'])==(Path(manifest['source_dataset'])/actual['file_path']).resolve()
            assert receipt['source_sha256']==actual['sha256']==sha(receipt['source_exr'])
            assert receipt['camera']==q['inventory'][index]['camera']
            local[variant]=np.asarray(Image.open(folder/'frame.png'));receipts[variant]=receipt
        assert len({r['source_exr'] for r in receipts.values()})==1
        np.testing.assert_array_equal(local['locked_arc'],local['rising_arc'])
        np.testing.assert_array_equal(local['free_arc'],local['soft_diagonal'])
        path=receipts['locked_arc']['source_exr'];bindings[path]=sha(path);raw_sources.append(sha(path))
        rgb=cv2.imread(path,cv2.IMREAD_UNCHANGED)[...,::-1]
        assert rgb.shape==(1080,1920,3) and np.isfinite(rgb).all()
        for variant in ['locked_arc','free_arc']:
            target=receipts[variant]['camera']
            linear=separable(rgb,source,target)
            expected=np.rot90(response(linear,gain,exposure))
            error=np.abs(expected.astype(int)-local[variant].astype(int))
            assert error.max()<=1
            records.append(dict(frame_id=frame,variant=variant,maximum_uint8_replay_error=int(error.max()),
                equivalent_variant='rising_arc' if variant=='locked_arc' else 'soft_diagonal',
                source_sha256=sha(path),exact_cross_variant_pixels=True))
    assert len(set(raw_sources))==32
    atomic_json(base/'independent_real_ending_audit.json',dict(status='all128_real_rgb_frames_verified',
        records=records,unique_native_source_times=32,independent_separable_replays=64,
        compared_prepared_images=128,maximum_uint8_replay_error=max(r['maximum_uint8_replay_error'] for r in records),
        no_mesh_used=True,heldout_used=False,quality_metrics=False,hashes=bindings))
    print('Independent float64 separable audit verified128 ending images; maxuint8error',max(r['maximum_uint8_replay_error'] for r in records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    audit(p.parse_args().base)
