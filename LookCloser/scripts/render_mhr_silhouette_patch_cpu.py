"""Matched native RGB with frozen texture wrappers and CPU-only execution."""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import argparse
import numpy as np
from admit_mhr_silhouette_patch import OUT,ARM
from admit_mhr_local_patch_depth import Scene2
from study_multiview_face_prior import read,save,sha

PARENT=Path('/mnt/data/dec5_elevated_camera_dynamic_150')


def main():
    import render_smooth_temporal_mesh_video as renderer
    from temporal_texture_view_prior import install
    from wide_dynamic_camera_flight import install_source_masks
    parser=argparse.ArgumentParser();parser.add_argument('--views',nargs='+',default=['old_moving','F004_E','M004_B','C004_E']);args=parser.parse_args()
    torch=renderer.torch;torch.set_num_threads(2)
    real_tensor=torch.tensor
    def cpu_tensor(*values,**kwargs):
        if str(kwargs.get('device','')).startswith('cuda'):kwargs['device']='cpu'
        return real_tensor(*values,**kwargs)
    torch.tensor=cpu_tensor;torch.cuda.empty_cache=lambda:None
    renderer.ThreadPoolExecutor=lambda *a,**kw:ThreadPoolExecutor(max_workers=1)
    renderer.scene_for=Scene2
    install(renderer);install_source_masks(renderer)
    parent=read(PARENT/'request.json');source=next(r for r in parent['source_rows'] if Path(r['source_dataset']).name=='001193')
    record=next(r for r in parent['inventory'] if r['frame_id']=='001193');rows,_,_=renderer.cameras('001193')
    source_mesh=Path(read(OUT/'request.json')['inputs']['actual_source_mesh'])
    scripts=['render_smooth_temporal_mesh_video.py','temporal_texture_view_prior.py','wide_dynamic_camera_flight.py',
             'hard_surface_texture.py','joint_temporal_texture.py','bake_joint_temporal_mesh.py','admit_mhr_local_patch_depth.py']
    for view in args.views:
        camera=record['camera'] if view=='old_moving' else next(r for r in rows if r['physical_camera'].startswith(view))
        for variant in ['baseline','strict','interpolated']:
            dest=OUT/'rgb'/view/variant;dest.mkdir(parents=True,exist_ok=True);(dest/'frames').mkdir(exist_ok=True)
            mesh=source_mesh if variant=='baseline' else OUT/ARM/variant/'mesh.ply'
            r=deepcopy(record);r.update(mesh=str(mesh),mesh_sha256=sha(mesh),camera=deepcopy(camera))
            request=dict(recipe=parent['recipe'],inventory=[r],source_rows=[source],
                parent_request_sha256=sha(PARENT/'request.json'),admission_request_sha256=sha(OUT/'request.json'),
                cpu_only=True,torch_threads=2,source_thread_workers=1,raycast_threads=2,
                device_only_torch_tensor_shim=True,original_texture_masks_not_geometry_override=True,
                matched_ablation_not_cpu_cuda_equivalence=True,view=view,variant=variant,
                source_profiles_sha256=sha(renderer.ROOT/'parameters.npz'),exposure_sha256=sha(renderer.ROOT/'exposure.json'),
                script_sha256=sha(__file__),helpers={n:sha(Path(__file__).with_name(n)) for n in scripts},
                target_rgb_used=False,production_accepted=False)
            if (dest/'request.json').exists():assert read(dest/'request.json')==request
            else:save(dest/'request.json',request)
            renderer.render_one(dest,r,source)
            print('finished',view,variant,flush=True)


if __name__=='__main__':main()
