"""Matched pixel-footprint control; immutable meshes, profiles and source images."""
from pathlib import Path
from copy import deepcopy
import argparse,inspect,hashlib
from joint_temporal_texture import read,sha,atomic_json
import render_smooth_temporal_mesh_video as renderer
from study_early_texture_prior import transform_source,early_quality
from temporal_texture_view_prior import angle_weights
from wide_dynamic_camera_flight import install_source_masks
from native_texture_footprint import snap_centers,relevant_tap,sample_native

OUT=Path('/mnt/data/dec5_native_texture_footprint')
BASE=Path('/mnt/data/dec5_poisson_jaw_completion/interpolated')


def transform(source):
    source=transform_source(source)
    replacements={
        "q=torch.tensor(uv[:,None],device='cuda');zq=":"q=_snap_centers(torch.tensor(uv[:,None],device='cuda'));zq=",
        'shifted=q+bounded_warp(static,residual,q);':'shifted=_snap_centers(q+bounded_warp(static,residual,q));',
        'valid&=(tap>0)&((tap-z).abs()<.003*z)':'valid&=((tap>0)&((tap-z).abs()<.003*z))|~_relevant_tap(q,dx,dy)',
        'safe&=(tap>0)&((tap-z).abs()<.003*z)':'safe&=((tap>0)&((tap-z).abs()<.003*z))|~_relevant_tap(shifted,dx,dy)',
    }
    for old,new in replacements.items():
        if source.count(old)!=1:raise ValueError('Unexpected renderer source: '+old)
        source=source.replace(old,new)
    return source


def install():
    source=transform(inspect.getsource(renderer.render_one))
    renderer.__dict__.update(_early_quality=early_quality,angle_weights=angle_weights,
        _snap_centers=snap_centers,_relevant_tap=relevant_tap,sample=sample_native)
    exec(compile(source,__file__+':native_footprint','exec'),renderer.__dict__)
    install_source_masks(renderer)
    return hashlib.sha256(source.encode()).hexdigest()


def run(view):
    implementation=install();renderer.torch.set_num_threads(2)
    baseline=BASE/'heldout' if view=='heldout' else BASE/'rgb/001193'/view/'repaired'
    request=deepcopy(renderer.verify_request(baseline));frame='001193'
    request['inventory']=[r for r in request['inventory'] if r['frame_id']==frame]
    request.update(partial_diagnostic_only=True,full_video_candidate=False,
        native_footprint_control=dict(snap_tolerance_pixels=.001,ignore_only_exact_zero_weight_taps=True,
            exact_integer_pixel_gather=True,fractional_sampling_unchanged=True,geometry_changed=False,
            baseline_request_sha256=sha(baseline/'request.json'),implementation_sha256=implementation))
    for name in ['study_native_texture_footprint.py','native_texture_footprint.py']:
        request['script_hashes'][name]=sha(Path(__file__).with_name(name))
    output=OUT/view;output.mkdir(parents=True,exist_ok=False);(output/'frames').mkdir()
    atomic_json(output/'request.json',request);renderer.render(output,[frame])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--view',choices=['moving','F004_E005_1210FP','heldout'],required=True);run(p.parse_args().view)
