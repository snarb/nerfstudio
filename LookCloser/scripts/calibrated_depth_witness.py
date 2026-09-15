"""Opt-in train RGB witnesses with the exact frozen renderer display response."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from joint_temporal_texture import ROOT,read,sha,exr,display,cameras
from study_confidence_depth_prior import project_integer,unproject
from study_jaw_train_confidence import enclosing_samples
from contrastive_forearm_witnesses import comparison_votes,carving_decision


def response_gains(log_gain,centered=True):
    values=np.asarray(log_gain)
    if values.ndim!=2 or values.shape[1]!=3 or not np.isfinite(values).all():
        raise ValueError('Expected finite camera-by-RGB log gains')
    return np.exp(values-values.mean(0,keepdims=True)) if centered else np.exp(values)


def load_images(frame,include_legacy=False):
    rows,_,_=cameras(frame);pars=np.load(ROOT/'parameters.npz')['log_gain']
    centered=response_gains(pars);legacy=response_gains(pars,False)
    assert centered.shape==(len(rows),3)
    profile=read(ROOT/'camera_profiles.json')
    assert profile['physical_cameras']==[r['physical_camera'] for r in rows]
    np.testing.assert_allclose(centered,profile['rgb_gain'],rtol=1e-6,atol=1e-7)
    gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    def load(pair):
        i,row=pair;rgb=exr(row['file_path'])
        a=np.rint(display(rgb*centered[i],gain)*255).clip(0,255).astype(np.uint8)
        b=np.rint(display(rgb*legacy[i],gain)*255).clip(0,255).astype(np.uint8) if include_legacy else None
        return row['physical_camera'],a,b
    with ThreadPoolExecutor(max_workers=4) as pool:values=list(pool.map(load,enumerate(rows)))
    metadata=dict(frame=frame,source_rgb_hashes={r['file_path']:sha(r['file_path']) for r in rows},
        parameters_sha256=sha(ROOT/'parameters.npz'),profiles_sha256=sha(ROOT/'camera_profiles.json'),
        exposure_sha256=sha(ROOT/'exposure.json'),mean_log_gain=pars.mean(0).tolist(),
        centered_like_renderer=True,heldout_rgb_loaded=False)
    return {n:a for n,a,b in values},({n:b for n,a,b in values} if include_legacy else None),metadata


def event_votes(points,camera,depth,rows,depths,images):
    """Compare four native observed rays with the nearer proposed depth on each.

    Call only for cached, corroborated four-tap depth vetoes. All four taps must
    remain photometrically qualified to uphold that fractional-sample veto.
    """
    uv,z=project_integer(camera,points);taps,observed,far=enclosing_samples(uv,z,depth)
    if not far.all():raise ValueError('Event is not a four-tap far observation')
    xy=taps.reshape(-1,2)
    actual=unproject(camera,xy[:,0],xy[:,1],observed.ravel())
    alternative=unproject(camera,xy[:,0],xy[:,1],np.repeat(z,4))
    votes=comparison_votes(actual,alternative,camera,rows,depths,images,.01)
    if not (votes['geometric']>=3).all():raise ValueError('Cached geometric corroboration does not replay')
    return votes,carving_decision(votes).reshape(-1,4).all(1)


def footprint_color_veto(points,rows,depths,images,depth_veto):
    previous=np.asarray(depth_veto,dtype=bool)
    if previous.shape!=(len(rows),len(points)):raise ValueError('Unexpected cached veto shape')
    qualified=np.zeros_like(previous);stats=[]
    for ci,(row,depth) in enumerate(zip(rows,depths)):
        ids=np.flatnonzero(previous[ci])
        if not len(ids):continue
        votes,keep=event_votes(points[ids],row,depth,rows,depths,images)
        qualified[ci,ids]=keep
        stats.append(dict(camera=row['physical_camera'],old_sample_vetoes=len(ids),qualified_sample_vetoes=int(keep.sum()),
            rgb_four_tap_qualified=int((votes['rgb_qualified'].reshape(-1,4)>=3).all(1).sum())))
        print('color footprint',ci+1,len(ids),int(keep.sum()),flush=True)
    return qualified,stats
