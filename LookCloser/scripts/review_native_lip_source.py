"""Inspect actual native train RGB behind the single-camera lip control."""
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json, cameras, display, ROOT as COLOR

ROOT=Path('/mnt/data/dec5_coherent_lip_source_audit')
CONTROL=Path('/mnt/data/dec5_coherent_lip_texture_margin4')


def main():
    import render_cinematic_6k_output as native
    ROOT.mkdir(exist_ok=False);frame='001083';folder=ROOT/'frames'/frame;folder.mkdir(parents=True)
    q=read(CONTROL/'request.json');r=read(CONTROL/'frames'/frame/'result.json')
    assert r['request_sha256']==sha(CONTROL/'request.json')
    evidence=CONTROL/'frames'/frame/'evidence.npz';assert sha(evidence)==r['hashes']['evidence.npz']
    uv=np.load(evidence)['source_uv'];lower=np.floor(uv.min(0)).astype(int)-10;upper=np.ceil(uv.max(0)).astype(int)+11
    assert (lower>=0).all() and (upper<=[5461,3072]).all()
    yy,xx=np.mgrid[lower[1]:upper[1],lower[0]:upper[0]];coords=np.stack([xx,yy],-1).reshape(-1,2).astype(np.float32)
    rows,_,_=cameras(frame);ci=next(i for i,row in enumerate(rows) if row['physical_camera']==q['chosen_source'])
    assert sha(native.__file__)==q['native_worker_sha256']
    assert sha(COLOR/'parameters.npz')==q['profiles_sha256'] and sha(COLOR/'exposure.json')==q['exposure_sha256']
    request=dict(frame=frame,controller_sha256=sha(__file__),script_sha256=sha(native.__file__),
        decoder_sha256=q['decoder_sha256'],control_request_sha256=sha(CONTROL/'request.json'),
        camera=q['chosen_source'],native_crop_uv_box=[*lower.tolist(),*upper.tolist()],
        native_integer_pixels=True,mesh_projection_used_for_rgb=False,diagnostic_only=True)
    atomic_json(ROOT/'request.json',request)
    native.OUT=ROOT;native.REMOTE_ROOT='/fsx/tmp/lookcloser_native_lip_source_audit'
    native.call(['ssh',native.REMOTE,'mkdir','-p',native.REMOTE_ROOT])
    native.call(['scp','-q',native.__file__,Path(__file__).with_name('convert_dec5_5a3_pq16_to_exr.py'),f'{native.REMOTE}:{native.REMOTE_ROOT}/'])
    samples=native.native_samples(frame,rows,{ci:coords},folder)[ci]
    original=read(CONTROL/'frames'/frame/'source_provenance.json')['sources'][0]
    receipt=read(folder/'source_provenance.json')['sources'][0]
    assert receipt['sha256']==original['sha256'] and receipt['path']==original['path']
    profile=np.load(COLOR/'parameters.npz')['log_gain'];gain=np.exp(profile[ci]-profile.mean(0))
    exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']
    rgb=np.rint(display(samples*gain,exposure)*255).clip(0,255).astype(np.uint8).reshape(*xx.shape,3)
    path=folder/'native_train_crop.png';Image.fromarray(np.rot90(rgb)).save(path)
    atomic_json(folder/'result.json',dict(request_sha256=sha(ROOT/'request.json'),image_sha256=sha(path),
        image_path=str(path),same_raw_source_sha256=receipt['sha256'],visual_status='pending'))
    print(path,flush=True)


if __name__=='__main__':main()
