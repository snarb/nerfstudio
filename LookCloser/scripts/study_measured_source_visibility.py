"""Opt-in positive measured-depth admission for current hard-source RGB.

Strictly diagnostic: unknown depth cannot admit an RGB source. Missing all
sources becomes black, not a silently restored invalid fallback. Meshes stay
fixed within each matched pair. Both production and a prior pruned mesh are
tested to separate source visibility from geometric deletion.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from joint_temporal_texture import read,sha,atomic_json
from review_full_block_transfer import ROOT as DEPTH_ROOT
from review_measured_free_surface import baseline,VIEWS
from ablate_free_surface_near_footprint import ROOT as PRUNED
from study_confidence_depth_prior import load_real

ROOT=Path('/mnt/data/dec5_measured_source_visibility')
ARMS=('production','pruned')


class MeasuredGate:
    def __init__(self,depth,tolerance=.0015):
        if depth.ndim!=3 or tolerance<=0 or not np.isfinite(tolerance):raise ValueError('Invalid measured depth gate')
        self.depth=depth;self.tolerance=tolerance
        self.calls=0;self.samples=0;self.admitted=0

    def __call__(self,q,z):
        if q.shape!=(self.depth.shape[0],1,z.shape[1],2) or z.shape[0]!=self.depth.shape[0]:
            raise ValueError('Expected camera x point projections')
        uv=q[:,0]+.5 # RGB array coordinates -> native COLMAP integer centers.
        finite=torch.isfinite(uv).all(-1)&torch.isfinite(z)&(z>0)
        xy=torch.round(torch.where(torch.isfinite(uv),uv,torch.zeros_like(uv))).long()
        x,y=xy[...,0],xy[...,1];h,w=self.depth.shape[1:]
        inside=finite&(x>=0)&(x<w)&(y>=0)&(y<h)
        cameras=torch.arange(len(self.depth),device=z.device)[:,None]
        observed=self.depth[cameras,y.clamp(0,h-1),x.clamp(0,w-1)]
        good=inside&torch.isfinite(observed)&(observed>0)&((observed-z).abs()<=self.tolerance)
        self.calls+=1;self.samples+=good.numel();self.admitted+=int(good.sum())
        return good

    def summary(self):
        return dict(tolerance=self.tolerance,unknown_depth_admits=False,nearest_native_depth=True,
            rgb_to_native_offset=.5,calls=self.calls,camera_point_queries=self.samples,
            camera_point_admissions=self.admitted,counts_not_quality_metrics=True)


def load_gate(frame,device,output):
    rows,depths,receipt=load_real(DEPTH_ROOT,frame)
    q=read(output/'request.json');spec=q['measured_source_visibility']
    assert receipt==spec['depth_receipt']
    assert [r['physical_camera'] for r in rows]==spec['physical_cameras']
    assert q['recipe']['static_registration'] is False
    return MeasuredGate(torch.as_tensor(np.stack(depths),device=device),spec['tolerance'])


def inject(source):
    replacements={
        "parameters=np.load(ROOT/'parameters.npz');":
            "_measured_gate=_load_measured_gate(frame,images.device,output)\n    parameters=np.load(ROOT/'parameters.npz');",
        'return q,zq,valid': 'valid&=_measured_gate(q,zq)\n        return q,zq,valid',
        "atomic_json(target/'result.json',result)":
            "result['measured_source_visibility']=_measured_gate.summary()\n    atomic_json(target/'result.json',result)",
    }
    for old,new in replacements.items():
        if source.count(old)!=1:raise ValueError('Unexpected source-visibility injection point: '+old)
        source=source.replace(old,new)
    return source


def install():
    import study_native_texture_footprint as footprint
    import render_smooth_temporal_mesh_video as engine
    from run_view_consistent_dynamic_video import install as parent_install
    original=footprint.transform
    footprint.transform=lambda source:inject(original(source))
    engine._load_measured_gate=load_gate
    return parent_install()


def old_root(frame,arm,view):
    return baseline(frame,view) if arm=='production' else PRUNED/frame/'rgb'/view


def prepare(frame):
    implementation=install();rows,_,receipt=load_real(DEPTH_ROOT,frame)
    received=read(DEPTH_ROOT/frame/'received.json')
    for name,digest in received['depth_hashes'].items():assert sha(DEPTH_ROOT/frame/name)==digest
    for arm in ARMS:
        for view in VIEWS:
            before=old_root(frame,arm,view);q=deepcopy(read(before/'request.json'))
            q['inventory']=[r for r in q['inventory'] if r['frame_id']==frame]
            assert len(q['inventory'])==1
            q['source_rows']=[r for r in q['source_rows'] if Path(r['source_dataset']).name==frame]
            q['ordered_frame_ids']=[frame]
            q.update(partial_diagnostic_only=True,full_video_candidate=False,geometry_changed=False,
                artifact_free_approval=False,source_quality_implementation_sha256=implementation,
                measured_source_visibility=dict(depth_receipt=receipt,tolerance=.0015,
                    physical_cameras=[r['physical_camera'] for r in rows],
                    baseline_request_sha256=sha(before/'request.json'),baseline=str(before),
                    unknown_depth_admits=False,mesh_unchanged_within_pair=True,
                    no_invalid_fallback=True,frame=frame,arm=arm))
            for name in [Path(__file__).name,'study_confidence_depth_prior.py']:
                q['script_hashes'][name]=sha(Path(__file__).with_name(name))
            out=ROOT/frame/arm/view;(out/'frames').mkdir(parents=True,exist_ok=True)
            if (out/'request.json').exists():assert q==read(out/'request.json')
            else:atomic_json(out/'request.json',q)


def render(frame,arm,view):
    import render_smooth_temporal_mesh_video as engine
    implementation=install();engine.torch.set_num_threads(2)
    out=ROOT/frame/arm/view
    assert implementation==read(out/'request.json')['source_quality_implementation_sha256']
    engine.render(out,[frame])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render'])
    p.add_argument('--arm',choices=ARMS);p.add_argument('--view',choices=VIEWS)
    a=p.parse_args()
    if a.action=='prepare':prepare('000995')
    else:
        if not a.arm or not a.view:p.error('render requires arm and view')
        render('000995',a.arm,a.view)
