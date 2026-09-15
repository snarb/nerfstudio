"""Production-base completion: native clay and matched current-recipe CPU RGB.

No video is changed. Old moving pose is an explicitly post-hoc stress view;
train views use the current production texture policy, not the old power-8
renderer. CPU/CUDA equivalence and full-sequence acceptance are not claimed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import sys

from run_mhr_production_patch_control import ROOT, OUT, ARM, PARENT, FRAME, binding
from admit_mhr_local_patch_depth import Scene2
from study_multiview_face_prior import read, save, sha


def check_recipe(parent):
    recipe = parent['recipe']
    assert recipe['source_incidence_power'] == 2
    assert recipe['static_registration'] is False
    assert recipe['pixel_fallback_angle_prior'] is True
    assert recipe['texture_source_prior'] == 'target_angle_before_incidence_clip'


def clay():
    import diffusion_mesh_repair
    import review_mhr_depth_admitted_patches as native
    import review_mhr_admission_branch_difference as branches
    diffusion_mesh_repair.scene_for = Scene2
    native.review(OUT)
    branches.ROOT = OUT
    branches.main()
    save(ROOT/'clay_wrapper.json', dict(script_sha256=sha(__file__),
        production_base_binding=binding(), native_threads=2,
        helpers={str(Path(m.__file__)): sha(m.__file__) for m in [native, branches]},
        target_used_posthoc_only=True, production_modified=False))


def render(views):
    import render_smooth_temporal_mesh_video as renderer
    from run_view_consistent_dynamic_video import install
    parent = read(PARENT/'request.json')
    check_recipe(parent)
    implementation = install()
    assert implementation == parent['source_quality_implementation_sha256']
    torch = renderer.torch
    torch.set_num_threads(2)
    original_tensor = torch.tensor

    def cpu_tensor(*values, **kwargs):
        if str(kwargs.get('device', '')).startswith('cuda'):
            kwargs['device'] = 'cpu'
        return original_tensor(*values, **kwargs)

    torch.tensor = cpu_tensor
    torch.cuda.empty_cache = lambda: None
    renderer.ThreadPoolExecutor = lambda *a, **kw: ThreadPoolExecutor(max_workers=1)
    renderer.scene_for = Scene2
    record = next(r for r in parent['inventory'] if r['frame_id'] == FRAME)
    source = next(r for r in parent['source_rows'] if Path(r['source_dataset']).name == FRAME)
    rows, _, _ = renderer.cameras(FRAME)
    moving_path = Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json')
    moving = next(r['camera'] for r in read(moving_path)['inventory'] if r['frame_id'] == FRAME)
    proof = binding()
    assert record['mesh_sha256'] == proof['production_mesh_sha256']
    helpers = ['run_view_consistent_dynamic_video.py', 'study_view_consistent_head_texture.py',
        'view_consistent_source_quality.py', 'study_native_texture_footprint.py',
        'native_texture_footprint.py', 'study_early_texture_prior.py',
        'study_unwarped_head_texture.py', 'wide_dynamic_camera_flight.py',
        'render_smooth_temporal_mesh_video.py', 'hard_surface_texture.py',
        'joint_temporal_texture.py', 'bake_joint_temporal_mesh.py', 'admit_mhr_local_patch_depth.py']
    for view in views:
        camera = moving if view == 'old_moving' else next(r for r in rows if r['physical_camera'].startswith(view))
        for variant in ['baseline', 'strict', 'interpolated']:
            mesh = Path(record['mesh']) if variant == 'baseline' else OUT/ARM/variant/'mesh.ply'
            if variant != 'baseline':
                result = read(mesh.parent/'result.json')
                assert result['observed_guard_passed'] and result['hashes']['mesh.ply'] == sha(mesh)
            row = deepcopy(record)
            row.update(camera=deepcopy(camera), mesh=str(mesh), mesh_sha256=sha(mesh))
            request = dict(recipe=parent['recipe'], inventory=[row], source_rows=[source],
                production_base_binding=proof, parent_request_sha256=sha(PARENT/'request.json'),
                source_quality_implementation_sha256=implementation,
                moving_pose_request_sha256=sha(moving_path), view=view, variant=variant,
                profiles_sha256=sha(renderer.ROOT/'parameters.npz'),
                exposure_sha256=sha(renderer.ROOT/'exposure.json'),
                cpu_only=True, torch_threads=2, source_thread_workers=1, raycast_threads=2,
                matched_ablation_not_cpu_cuda_equivalence=True, original_texture_masks_unchanged=True,
                target_rgb_used=False, production_accepted=False, script_sha256=sha(__file__),
                helpers={n: sha(Path(__file__).with_name(n)) for n in helpers})
            for key in ['profiles_sha256', 'exposure_sha256']:
                assert request[key] == parent[key]
            dest = OUT/'rgb'/view/variant
            (dest/'frames').mkdir(parents=True, exist_ok=True)
            if (dest/'request.json').exists():
                assert read(dest/'request.json') == request
            else:
                save(dest/'request.json', request)
            renderer.render_one(dest, row, source)
            print('completed current-recipe CPU', view, variant, flush=True)


def rgb_review(views):
    import review_mhr_silhouette_patch_rgb as review
    review.OUT = OUT
    sys.argv = [__file__, '--views', *views]
    review.main()
    save(ROOT/'rgb_review_wrapper.json', dict(script_sha256=sha(__file__),
        helper_sha256=sha(review.__file__), views=views, production_base_binding=binding(),
        current_recipe=True, matched_cpu_not_cuda_equivalence=True, production_modified=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['clay', 'render', 'rgb'])
    parser.add_argument('--views', nargs='+', choices=['old_moving', 'F004_E', 'M004_B', 'C004_E'],
                        default=['old_moving', 'F004_E', 'M004_B', 'C004_E'])
    args = parser.parse_args()
    assert read(ROOT/'request.json') == binding()
    if args.stage == 'clay':
        clay()
    elif args.stage == 'render':
        render(args.views)
    else:
        rgb_review(args.views)
