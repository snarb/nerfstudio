"""Matched mesh/pose controls for hard-source seams; never average RGB."""
import argparse
from copy import deepcopy
from pathlib import Path
import inspect
import hashlib
import numpy as np
import render_smooth_temporal_mesh_video as renderer
import study_native_texture_footprint as footprint
from joint_temporal_texture import read,sha,atomic_json
from view_consistent_source_quality import quality
from wide_dynamic_camera_flight import install_source_masks
from review_jaw_repair_transfer import verified_image,panel

ROOT=Path('/mnt/data/dec5_view_consistent_head_texture')
BASE=Path('/mnt/data/dec5_central_train_pose_transfer')
CASES=[('001193','native_G004_C005_121037'),('001123','native_K004_C005_1210BC'),('001193','moving'),('001123','moving')]
MODES=('incidence2','angular_only')


def install(mode):
    source=footprint.transform(inspect.getsource(renderer.render_one))
    pairs={
        'np.abs((directions*normal).sum(-1))**8/length.clip(.01)**2':"_view_quality((directions*normal).sum(-1),length,_quality_mode)",
        'np.abs((direction*normal[f]).sum(-1))**8/length.clip(.01)**2':"_view_quality((direction*normal[f]).sum(-1),length,_quality_mode)*angle_weights(rows,record['camera'],read(output/'request.json')['recipe']['target_angle_sigma_degrees'])[0][:,None]",
    }
    if mode=='angular_only':pairs['select_surface_sources(color,quality,t,color_weight=.5)']='select_surface_sources(color,quality,t,color_weight=0.)'
    for old,new in pairs.items():
        if source.count(old)!=1:raise ValueError('Unexpected renderer statement: '+old)
        source=source.replace(old,new)
    renderer.__dict__.update(_view_quality=quality,_quality_mode=mode,_early_quality=footprint.early_quality,
        angle_weights=footprint.angle_weights,_snap_centers=footprint.snap_centers,
        _relevant_tap=footprint.relevant_tap,sample=footprint.sample_native)
    exec(compile(source,__file__+':source_quality','exec'),renderer.__dict__)
    install_source_masks(renderer);return hashlib.sha256(source.encode()).hexdigest()


def run(mode):
    implementation=install(mode);renderer.torch.set_num_threads(2)
    out=ROOT/mode;out.mkdir(parents=True,exist_ok=False)
    for frame,view in CASES:
        baseline=BASE/frame/view;parent=renderer.verify_request(baseline);request=deepcopy(parent)
        request.update(hard_source_quality_control=mode,geometry_changed=False,
            baseline_request_sha256=sha(baseline/'request.json'),implementation_sha256=implementation,
            pixel_angle_prior=True,graph_color_weight=0. if mode=='angular_only' else .5,
            partial_diagnostic_only=True,full_video_candidate=False)
        for n in ['view_consistent_source_quality.py','study_view_consistent_head_texture.py']:
            request['script_hashes'][n]=sha(Path(__file__).with_name(n))
        dest=out/frame/view;dest.mkdir(parents=True);(dest/'frames').mkdir()
        atomic_json(dest/'request.json',request);renderer.render(dest,[frame])
        old,oldr=verified_image(baseline,frame);new,newr=verified_image(dest,frame)
        for key in ['camera','source_cameras','mesh_sha256','fixed_exposure']:assert oldr[key]==newr[key]
        oldd=np.load(baseline/'frames'/frame/'target_depth.npz')['depth'];newd=np.load(dest/'frames'/frame/'target_depth.npz')['depth']
        np.testing.assert_array_equal(oldd,newd)
        images=[old,new];names=['baseline','source quality '+mode]
        if view.startswith('native_'):
            from PIL import Image
            gt=np.array(Image.open(BASE/frame/'gt'/(view[len('native_'):]+'.png')))
            images.insert(0,gt);names.insert(0,'real train GT')
        review=dest/'review';review.mkdir()
        panel(review/'head.png',images,names,(160,450,1000,1250))
        atomic_json(review/'result.json',dict(frame=frame,view=view,mode=mode,depth_byte_equal=True,
            changed_rgb_pixels=int(np.any(old!=new,2).sum()),panel_sha256=sha(review/'head.png'),
            visual_status='pending',no_image_quality_metrics=True))
    print('Completed',mode,'four matched views',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=MODES,required=True);run(p.parse_args().mode)
