"""Six-worker dynamic replay: incidence2, fixed color, zero image registration.

This is a full-sequence texture evaluation with known geometry failures, not an
artifact-free approval. Actor times, camera travel and all meshes are unchanged.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import os
from datetime import datetime,timezone
import run_dynamic_grid_workers as lifecycle
from joint_temporal_texture import read,sha,atomic_json
from study_view_consistent_head_texture import install as source_install,CASES,BASE
from study_unwarped_head_texture import zero_registration

PARENT=Path('/mnt/data/dec5_phase30_early_texture_dynamic_150')
OUTPUT=Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150')


def install():
    implementation=source_install('incidence2')
    lifecycle.renderer.bounded_warp=zero_registration
    return implementation


def initialize(output):
    request=deepcopy(lifecycle.renderer.verify_request(PARENT));implementation=install()
    metrics=Path('/mnt/data/dec5_unwarped_head_texture/metrics.json');score=read(metrics)
    selected=next(r for r in score['rows'] if r['mode']=='unwarped');baseline=next(r for r in score['rows'] if r['mode']=='baseline')
    assert selected['face_psnr']>baseline['face_psnr'] and selected['face_ssim']>baseline['face_ssim'] and selected['face_lpips']<baseline['face_lpips']
    # Explicitly retain the model-selection limitation: one held-out benchmark
    # is not independent evidence of improvement across all 150 times.
    request['recipe'].update(static_registration=False,source_incidence_power=2,
        texture_source_prior='target_angle_before_incidence_clip',pixel_fallback_angle_prior=True)
    images=[]
    roots=[Path('/mnt/data/dec5_view_consistent_head_texture')/m for m in ['incidence2','angular_only']]
    roots += [Path('/mnt/data/dec5_pixel_angular_head_texture'),Path('/mnt/data/dec5_unwarped_head_texture')]
    for root in roots:images += [root/frame/view/'review/head.png' for frame,view in CASES]
    images += [Path('/mnt/data/dec5_head_source_quality_heldout')/(m+'_heldout.png') for m in ['incidence2','angular_only','pixel_angular']]
    images.append(Path('/mnt/data/dec5_unwarped_head_texture/unwarped_heldout.png'))
    request.update(source_quality_implementation_sha256=implementation,matched_texture_parent_sha256=sha(PARENT/'request.json'),
        initial_gate=dict(status='proceed_to_full_sequence_evaluation_with_known_geometry_failures',
            heldout_metrics=str(metrics),heldout_metrics_sha256=sha(metrics),
            reviewed_images={str(p):sha(p) for p in images},
            notes='Softer incidence improves face fidelity; pure angular policies worsen LPIPS. Registration-off gives small additional gain. Crown/hand/jaw geometry defects remain.'),
        required_initial_rgb_gate=[],geometry_changed_from_texture_parent=False,source_admission_order_only=False,
        pixel_fallback_changed=True,artifact_free_approval=False,partial_diagnostic_only=False,
        source_quality_validation_scope='Two actor instants / four train-or-moving views; one held-out face benchmark, used for variant selection.',
        inherited_camera_and_actor_inventory_unchanged=True)
    for n in [Path(__file__).name,'view_consistent_source_quality.py','study_view_consistent_head_texture.py',
              'study_native_texture_footprint.py','native_texture_footprint.py','study_unwarped_head_texture.py','run_dynamic_grid_workers.py']:
        request['script_hashes'][n]=sha(Path(__file__).with_name(n))
    output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Frozen video request mismatch')
    atomic_json(output/'request.json',request)


def worker(output,index,frames):
    renderer=lifecycle.renderer;request=renderer.verify_request(output)
    if request['recipe'].get('train_foreground_guard'):raise ValueError('Unexpected geometry-changing wrapper')
    if install()!=request['source_quality_implementation_sha256']:raise ValueError('Changed source-quality execution')
    original=renderer.atomic_json
    def write(path,payload):
        if Path(path)==output/'progress.json':path=output/'workers'/f'{index}.json'
        original(path,payload)
    renderer.atomic_json=write;render_one=renderer.render_one
    def notified(root,record,manifest):
        write(root/'progress.json',dict(pid=os.getpid(),frame_id=record['frame_id'],index=record['index'],stage='starting_frame',
                                      utc=datetime.now(timezone.utc).isoformat()))
        return render_one(root,record,manifest)
    renderer.render_one=notified;renderer.render(output,frames)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','supervise','worker'])
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--worker-index',type=int);p.add_argument('--frames',nargs='+');a=p.parse_args()
    lifecycle.renderer.torch.set_num_threads(2)
    if a.action=='init':initialize(a.output)
    elif a.action=='worker':worker(a.output,a.worker_index,a.frames)
    else:
        lifecycle.__file__=__file__;lifecycle.supervise(a.output,a.workers)
