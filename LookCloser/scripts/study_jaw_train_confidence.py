"""Two frozen train-confidence arms, retaining the strict native ray veto.

Arm train_anchor changes only the sample roundtrip reference. Arm footprint
also requires all four enclosing native depth samples to corroborate a far
surface before a fractional 3D sample is rejected. The final 62-camera integer
and half-pixel ray veto is unchanged in BOTH arms.
"""
from pathlib import Path
import argparse
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from study_confidence_depth_prior import load_real,project_integer,unproject,support
from study_jaw_depth_footprint import BASE
import guard_jaw_measured_depth as guard

FOOTPRINT=Path('/mnt/data/dec5_jaw_depth_footprint')


def enclosing_samples(uv,z,depth):
    xy=np.floor(uv).astype(np.int64)
    taps=xy[:,None,:]+np.array([[0,0],[1,0],[0,1],[1,1]])[None]
    inside=(z>0)&np.isfinite(z)&(taps[:,:,0]>=0).all(1)&(taps[:,:,0]<depth.shape[1]).all(1) \
        &(taps[:,:,1]>=0).all(1)&(taps[:,:,1]<depth.shape[0]).all(1)
    observed=np.zeros((len(uv),4),float);ids=np.flatnonzero(inside)
    observed[ids]=depth[taps[ids,:,1],taps[ids,:,0]]
    far=inside&np.isfinite(observed).all(1)&(observed>0).all(1)&(observed>z[:,None]+.003).all(1)
    return taps,observed,far


def footprint_veto(points,rows,depths):
    veto=np.zeros((len(rows),len(points)),np.uint8)
    for ci,(camera,depth) in enumerate(zip(rows,depths)):
        uv,z=project_integer(camera,points);taps,observed,far=enclosing_samples(uv,z,depth);ids=np.flatnonzero(far)
        if len(ids):
            xy=taps[ids].reshape(-1,2);obs=observed[ids].ravel()
            counts,_=support(unproject(camera,xy[:,0],xy[:,1],obs),camera,rows,depths)
            veto[ci,ids]=(counts.reshape(-1,4)>=3).all(1)
        if (ci+1)%20==0:print('footprint camera',ci+1,flush=True)
    return veto


def run(output,frame,arm):
    root=output/arm;folder=root/'analysis'/frame;folder.mkdir(parents=True,exist_ok=True)
    src=BASE/'analysis'/frame;sr=read(src/'result.json');tr=read(FOOTPRINT/frame/'result.json')
    if sha(src/'evidence.npz')!=sr['evidence_sha256'] or sha(FOOTPRINT/frame/'train_reference.npz')!=tr['reference_npz_sha256']:
        raise ValueError('Changed frozen evidence')
    rows,depths,receipt=load_real(BASE/'analysis',frame)
    if receipt!=read(src/'request.json')['real_depth_receipt']:raise ValueError('Observed maps changed')
    request=dict(arm=arm,frame=frame,script_sha256=sha(__file__),guard_script_sha256=sha(Path(guard.__file__)),
        real_depth_receipt=receipt,source_evidence_sha256=sha(src/'evidence.npz'),train_reference_result_sha256=sha(FOOTPRINT/frame/'result.json'),
        sample_free_space_rule='all four enclosing depths corroborated' if arm=='footprint' else 'nearest native depth corroborated',
        final_ray_veto='unchanged nearest native observed depth, integer and half-pixel rays, 62 cameras',
        numerical_thresholds_unchanged=True,heldout_used=False,virtual_reference_used=False,production_changed=False)
    if (folder/'request.json').exists() and read(folder/'request.json')!=request:raise ValueError('Frozen arm request mismatch')
    atomic_json(folder/'request.json',request)
    if not (folder/'result.json').exists():
        arrays=dict(np.load(src/'evidence.npz'));arrays['votes']=np.load(FOOTPRINT/frame/'train_reference.npz')['votes']
        if arm=='footprint':arrays['trusted_free']=footprint_veto(arrays['samples'].reshape(-1,3),rows,depths).reshape(62,-1,10)
        np.savez_compressed(folder/'evidence.npz',**arrays)
        atomic_json(folder/'real_depth_input.json',read(src/'real_depth_input.json'))
        atomic_json(folder/'result.json',dict(request_sha256=sha(folder/'request.json'),evidence_sha256=sha(folder/'evidence.npz'),
            interpretation='Train-only confidence ablation; old diagnostic-only arrays retained for attribution',production_accepted=False))
    else:
        record=read(folder/'result.json')
        if record['request_sha256']!=sha(folder/'request.json') or record['evidence_sha256']!=sha(folder/'evidence.npz'):raise ValueError('Changed arm cache')
    guard.run(root,frame)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_jaw_train_confidence'))
    p.add_argument('--frame',required=True,choices=['001193','001195']);p.add_argument('--arm',required=True,choices=['train_anchor','footprint'])
    a=p.parse_args();run(a.output,a.frame,a.arm)
