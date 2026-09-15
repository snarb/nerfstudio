"""Test pixel visibility against centroid-label occlusion in hard texturing."""
from pathlib import Path
from copy import deepcopy
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
import study_view_consistent_head_texture as study
from review_jaw_repair_transfer import verified_image,panel

ROOT=Path('/mnt/data/dec5_pixel_angular_head_texture')


def install():
    implementation=study.install('angular_only')
    renderer=study.renderer;original=renderer.gather_hard_rgb
    def gather(colors,weights,preferred):
        # The face centroid may be occluded although this target pixel is not.
        # Keep all original per-pixel geometric tests; choose exactly one source.
        return original(colors,weights,weights.argmax(0))
    renderer.gather_hard_rgb=gather
    return implementation


def run():
    implementation=install();renderer=study.renderer;renderer.torch.set_num_threads(2)
    ROOT.mkdir(exist_ok=False)
    for frame,view in study.CASES:
        baseline=study.BASE/frame/view;q=deepcopy(renderer.verify_request(baseline))
        q.update(pixel_angular_hard_selection=True,geometry_changed=False,graph_labels_not_used_for_rgb=True,
            baseline_request_sha256=sha(baseline/'request.json'),implementation_sha256=implementation,
            partial_diagnostic_only=True,full_video_candidate=False,graph_color_weight=0.,pixel_angle_prior=True)
        for name in ['view_consistent_source_quality.py','study_view_consistent_head_texture.py',Path(__file__).name]:q['script_hashes'][name]=sha(Path(__file__).with_name(name))
        dest=ROOT/frame/view;dest.mkdir(parents=True);(dest/'frames').mkdir();atomic_json(dest/'request.json',q)
        renderer.render(dest,[frame]);old,oldr=verified_image(baseline,frame);new,newr=verified_image(dest,frame)
        for key in ['camera','source_cameras','mesh_sha256','fixed_exposure']:assert oldr[key]==newr[key]
        np.testing.assert_array_equal(np.load(baseline/'frames'/frame/'target_depth.npz')['depth'],np.load(dest/'frames'/frame/'target_depth.npz')['depth'])
        images=[old,new];names=['baseline','pixel angular: one source']
        if view.startswith('native_'):
            images.insert(0,np.array(Image.open(study.BASE/frame/'gt'/(view[7:]+'.png'))));names.insert(0,'real train GT')
        review=dest/'review';review.mkdir();panel(review/'head.png',images,names,(160,450,1000,1250))
        atomic_json(review/'result.json',dict(frame=frame,view=view,depth_byte_equal=True,
            changed_rgb_pixels=int(np.any(old!=new,2).sum()),panel_sha256=sha(review/'head.png'),visual_status='pending'))
    print('Completed four pixel-angular controls',flush=True)


if __name__=='__main__':run()
