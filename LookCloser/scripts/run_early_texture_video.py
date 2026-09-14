"""Supervised 150-time phase+30 replay with early target-angle source admission."""
from pathlib import Path
from copy import deepcopy
import argparse
import os
from datetime import datetime,timezone
import run_dynamic_grid_workers as lifecycle
from joint_temporal_texture import read,sha,atomic_json
from study_early_texture_prior import install

PARENT=Path('/mnt/data/dec5_phase30_dynamic_150')
OUTPUT=Path('/mnt/data/dec5_phase30_early_texture_dynamic_150')
GATE=Path('/mnt/data/dec5_early_texture_prior_gate.json')


def initialize(output):
    request=deepcopy(lifecycle.renderer.verify_request(PARENT));implementation=install();gate=read(GATE)
    if gate['status']!='proceed_to_full_sequence_evaluation_with_known_geometry_failures':raise ValueError('Missing initial visual gate')
    request['recipe']['texture_source_prior']='target_angle_before_incidence_clip'
    request.update(admission_transform_sha256=implementation,matched_phase_parent_sha256=sha(PARENT/'request.json'),
        initial_gate=dict(path=str(GATE),sha256=sha(GATE),image_hashes={p:sha(p) for p in gate['reviewed_images']},
                          heldout_metrics_sha256=sha(gate['heldout_metrics'])),
        required_initial_rgb_gate=[],geometry_changed_from_phase_parent=False,
        source_admission_order_only=True,pixel_fallback_changed=False,artifact_free_approval=False)
    for n in [Path(__file__).name,'study_early_texture_prior.py','run_dynamic_grid_workers.py']:
        request['script_hashes'][n]=sha(Path(__file__).with_name(n))
    output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Frozen video request mismatch')
    atomic_json(output/'request.json',request)


def worker(output,index,frames):
    renderer=lifecycle.renderer;request=renderer.verify_request(output)
    if request['recipe'].get('train_foreground_guard'):raise ValueError('Expected only the preserved source-mask wrapper')
    if install()!=request['admission_transform_sha256']:raise ValueError('Changed early-admission execution')
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
