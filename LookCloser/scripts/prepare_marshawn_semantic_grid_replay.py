"""Prepare a sealed executable proposal; never initializes CUDA or launches jobs."""
import argparse
import json
from pathlib import Path
import shutil
from run_marshawn_semantic_grid_replay import sha,write,validate


def prepare(campaign,output):
    C=Path(campaign).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);code=out/'code';code.mkdir()
    p3=C/'frozen_LC_semantic_physical_plan_v3';src=Path(__file__).resolve().parent
    inherited=['marshawn_LC_frozen_source.py','marshawn_LC_semantic_physical_path.py','marshawn_stable_packed_density_reference.py',
        'marshawn_frozen_semantic_head.py','marshawn_frozen_semantic_selectors.py','prepare_marshawn_frozen_semantic_cpu.py','marshawn_tuning_metrics.py','marshawn_gpu_isolation.py']
    fresh=['marshawn_semantic_grid_replay.py','run_marshawn_semantic_grid_replay.py','launch_marshawn_semantic_grid_replay.py',
        'evaluate_marshawn_semantic_grid_replay_cpu.py','prepare_marshawn_semantic_grid_replay.py',
        'export_marshawn_semantic_cell_candidates_cpu.py','marshawn_semantic_node_cell_bridge.py']
    for name in inherited:shutil.copy2(p3/'code'/name,code/name)
    for name in fresh:shutil.copy2(src/name,code/name)
    files=json.loads((p3/'root_release.json').read_text())['bound_files'].copy()
    def bind(p):
        p=Path(p).resolve();value=sha(p)
        if str(p)in files and files[str(p)]!=value:raise ValueError('Inherited immutable source changed '+str(p))
        files[str(p)]=value;return str(p)
    for p in code.glob('*.py'):bind(p)
    for p in [p3/'root_release.json',p3/'specs/prior001_render.json']:bind(p)
    archive=C/'frozen_LC_semantic_head_archives/recovery1_20261107_v1'
    head=archive/'runs/prior001_fresh2064/head.pt';bind(head)
    candidate=C/'semantic_cell_candidates_cpu_v1/prior001_fresh2064/hard_theta01'
    for name in ['candidate.npz','metadata.json','witnesses.json']:bind(candidate/name)
    import numpy as np
    from export_marshawn_semantic_cell_candidates_cpu import load_candidate
    from marshawn_semantic_node_cell_bridge import fresh_dilation
    with np.load(candidate/'candidate.npz',allow_pickle=False) as z:
        if [int(z[k].sum())for k in ['original_active','active_r0','active_r1']]!=[32046,23730,38319]:raise ValueError('Exact actual candidate counts differ')
        for radius in [0,1]:
            expected=np.stack([fresh_dilation(level,radius)for level in z['source_core']])&z['original_admission']&z['semantic_envelope']
            if not np.array_equal(expected,z['active_r'+str(radius)]):raise ValueError('Fresh noncumulative dilation policy differs')
    grid=C/'independent_occupancy/semantic_quarter_distortion_final_geometry/quarter_distortion_rawmin/grid.npz';bind(grid);bind(grid.with_suffix('.json'))
    q=C/'independent_occupancy/quarter_distortion_sourcepoint_counterfactual_v1/queries'
    queries={k:bind(q/(k+'.json'))for k in ['clothing103','shoe6','DC12']}
    queries['contour9']=bind(C/'independent_occupancy/sourceW32_continuation_final_geometry_v1/queries/contour9.json')
    reference=p3/'prior001/renders'
    for cam in ['DC','HC','KA']:
        for name in ['physical_and_partition.npz','path_coverage.npz']:bind(reference/cam/name)
    ready=C/'data/ready.json';bind(ready)
    for row in json.loads(ready.read_text())['records']:
        if row['camera']in ['DC','HC','KA']:
            bind(C/'data'/row['image']);bind(C/'data/semantics'/(row['camera']+'.npz'))
    regions=C/'native_semantic_grid_20000/corrected_v2/review/region_crops.json'
    bind(regions);bind(reference/'complete.json');bind(p3/'prior001/visual_gate/metrics.json')
    spec=json.loads((p3/'specs/prior001_render.json').read_text())
    plan=dict(kind='FROZEN_LC_P01_THETA01_GRID_R0_R1_V1',root_release=str(out/'release.json'),helpers=str(code),controller=str(out/'controller'),output=str(out/'replay'),
        python='/home/pavlo/.conda/envs/nerfstudio/bin/python',gpu=0,gpu_uuid='GPU-7ed25c4d-10d1-cbe3-d0b2-76c21211bd46',
        allocator_cap_GiB=8,process_cap_MiB=10240,max_seconds=3600,resource_poll_seconds=5,
        source_checkpoint_sha256='340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641',
        head_sha256='bdb2cc0d8138e4c22c61b4d411227b272b95ceac9e8295d474eecc4a84aa98bc',head=str(head),source_spec=spec,
        candidate=str(candidate),candidate_metadata_sha256=sha(candidate/'metadata.json'),source_grid_sha256=sha(grid),
        selector='hard_theta01',phases=['source_grid','semantic_r0','semantic_r1'],cell_counts=[32046,23730,38319],
        radius_policy='Fresh dilation(source learned core,r) AND original admission AND conservative semantic node-support envelope; never cumulative',
        only_model_mutation='Temporary occupancy_grid.binaries override; restore before all seven-state/RNG/LUT/FR source assertions',
        physical_density='Original native sigma multiplied by SAME frozen p01 hard theta.1 at actual samples, with transmittance recomputed',
        source_label_regime='two_reviewed_UNKNOWN_40470_40622',native_cap=1024,chunk=128,uniform_samples=[8192,16384],queries=queries,
        P3_reference=str(reference),P3_reference_completion=str(reference/'complete.json'),P3_reference_metrics=str(p3/'prior001/visual_gate/metrics.json'),
        data=str(C/'data'),ready=str(ready),regions=str(regions),bound_files=files,
        native_phase_cap_and_tail_changes_expected=True,CPU_candidate_support_is_not_physical_free_space=True,
        source_grid_may_omit_nonzero_raw_density=True,no_source_masks_or_grids_modified=True,optimizer_updates=0,training_authorized=False,
        manual_full3FHD_body_hair_hands_feet_review_required=True,automatic_quality_promotion=False)
    for radius in [0,1]:
        load_candidate(candidate,expected_metadata_sha256=plan['candidate_metadata_sha256'],expected_head_sha256=plan['head_sha256'],expected_source_grid_sha256=plan['source_grid_sha256'],expected_selector=plan['selector'],radius=radius)
    path=out/'plan.json';write(path,plan)
    write(out/'release.template.json',dict(reviewer='root',readonly_grid_comparison_authorized=False,plan_sha256=sha(path),
        scope='Exclusive local UUID GPU0 fixed twoUNKNOWN p01 theta.1 SOURCE/R0/R1 physical replay; no fitting, optimizer, source mutations or promotion',
        source_checkpoint_sha256=plan['source_checkpoint_sha256'],head_sha256=plan['head_sha256'],no_training=True))
    validate(path,False)
    write(out/'CPU_preparation.json',dict(passed=True,plan_sha256=sha(path),bound_files=len(files),
        inherited_helpers={n:sha(code/n)for n in inherited},new_helpers={n:sha(code/n)for n in fresh},
        GPU_started=False,release_authorized=False,
        launch_command=f"{plan['python']} {code/'launch_marshawn_semantic_grid_replay.py'} --plan {path} --execute"))
    print(json.dumps(dict(plan=str(path),sha256=sha(path),bound_files=len(files),GPU_started=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.campaign,a.output)
