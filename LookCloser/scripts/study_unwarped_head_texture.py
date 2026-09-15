"""Isolate bounded image-registration effects after incidence2 source selection."""
from pathlib import Path
from copy import deepcopy
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
import study_view_consistent_head_texture as study
import evaluate_head_source_quality as heldout
from review_jaw_repair_transfer import verified_image,panel

ROOT=Path('/mnt/data/dec5_unwarped_head_texture')


def zero_registration(static,residual,uv):
    return uv.new_zeros(uv.shape)


def run():
    implementation=study.install('incidence2');renderer=study.renderer
    renderer.bounded_warp=zero_registration;renderer.torch.set_num_threads(2);ROOT.mkdir(exist_ok=False)
    cases=[(f,v,study.BASE/f/v,ROOT/f/v) for f,v in study.CASES]
    cases.append(('001193','heldout',heldout.ROOT/'incidence2',ROOT/'unwarped'))
    for frame,view,baseline,dest in cases:
        q=deepcopy(renderer.verify_request(baseline));q['recipe']['static_registration']=False
        q.update(static_registration_control='disabled_zero_displacement',geometry_changed=False,
            baseline_request_sha256=sha(baseline/'request.json'),implementation_sha256=implementation,
            partial_diagnostic_only=True,full_video_candidate=False,pixel_angle_prior=True,graph_color_weight=.5,
            graph_labels_not_used_for_rgb=False,hard_source_quality_control='incidence2')
        for name in ['view_consistent_source_quality.py','study_view_consistent_head_texture.py',Path(__file__).name]:q['script_hashes'][name]=sha(Path(__file__).with_name(name))
        dest.mkdir(parents=True);(dest/'frames').mkdir();atomic_json(dest/'request.json',q);renderer.render(dest,[frame])
        old,oldr=verified_image(baseline,frame);new,newr=verified_image(dest,frame)
        for key in ['camera','source_cameras','mesh_sha256','fixed_exposure']:assert oldr[key]==newr[key]
        np.testing.assert_array_equal(np.load(baseline/'frames'/frame/'target_depth.npz')['depth'],np.load(dest/'frames'/frame/'target_depth.npz')['depth'])
        images=[old,new];names=['baseline (incidence2 for heldout)','incidence2, registration off']
        if view.startswith('native_'):
            images.insert(0,np.array(Image.open(study.BASE/frame/'gt'/(view[7:]+'.png'))));names.insert(0,'real train GT')
        review=dest/'review';review.mkdir();panel(review/'head.png',images,names,(160,450,1000,1250))
        atomic_json(review/'result.json',dict(frame=frame,view=view,depth_byte_equal=True,
            changed_rgb_pixels=int(np.any(old!=new,2).sum()),panel_sha256=sha(review/'head.png'),visual_status='pending'))
    for mode in ['baseline','incidence2']:(ROOT/mode).symlink_to(heldout.ROOT/mode,target_is_directory=True)
    heldout.ROOT=ROOT;heldout.MODES=('baseline','incidence2','unwarped');heldout.score()


if __name__=='__main__':run()
