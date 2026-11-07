"""CPU-only false-release package for matched FOUR-UNKNOWN128 versus256.

Copies the sealed worker dependencies and the opt-in resolution worker; never
uploads, launches, grants permission or changes the original FOUR-UNKNOWN arm.
"""
import argparse
import copy
import difflib
import json
from pathlib import Path
import shutil

from prepare_marshawn_frozen_semantic_cpu import sha, write
from marshawn_LC_semantic_resolution_contract import FAMILY, validate_qualification, validate_matched_baseline

SC = Path('/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106')
C = Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106')
SEALED_WORKER_SHA = '6f6bca9b69c97f0375d943a37f208a17407fcb64f046e920556e101382ed149d'
FOUR_LABEL_SHA = '44e7734ed93a2c10742de6dc55886978d7c2097ee62779915dada93feda94df4'
FOUR_SPEC_SHA = 'ccfdd894d29d06ef17b8dffd8fa66b7db06b4a8383d7691246632d8fff7fba93'


def prepare(campaign, baseline_root, output):
    if output.exists(): raise FileExistsError('Use a new immutable resolution proposal namespace')
    remote = SC/output.relative_to(campaign)
    baseline_remote = SC/baseline_root.relative_to(campaign)
    baseline_path = baseline_root/'spec.json'; label_path = baseline_root/'label_qualification.json'
    if sha(baseline_path) != FOUR_SPEC_SHA or sha(label_path) != FOUR_LABEL_SHA:
        raise ValueError('Use the exact root-qualified FOUR-UNKNOWN128 comparator')
    baseline = json.loads(baseline_path.read_text())
    labels = json.loads(label_path.read_text()); validate_qualification(labels)
    old_release = json.loads((baseline_root/'root_review_TEMPLATE.json').read_text())
    bound = dict(old_release['bound_files'])

    local_archive_aliases = {}

    def local(remote_path):
        p = campaign/Path(remote_path).relative_to(SC)
        if p.is_file(): return p
        control_head = SC/'frozen_LC_semantic_head_silver_alias_v1/runs/prior001_fresh2064/head.pt'
        if Path(remote_path) == control_head:
            archived = campaign/'frozen_LC_semantic_head_archives/recovery1_20261107_v1/runs/prior001_fresh2064/head.pt'
            local_archive_aliases[str(remote_path)] = str(archived)
            return archived
        return p

    # Verify the inherited complete source closure, without changing any bytes.
    for path, digest in bound.items():
        if sha(local(path)) != digest: raise ValueError('Inherited immutable input changed: '+path)
    old_code = campaign/'frozen_LC_semantic_head_plan_v1/code'
    old_worker = old_code/'probe_marshawn_LC_frozen_semantic_head.py'
    if sha(old_worker) != SEALED_WORKER_SHA: raise ValueError('Wrong sealed128 baseline worker')
    output.mkdir(parents=True); (output/'code').mkdir(); (output/'specs').mkdir(); (output/'controllers').mkdir()
    for p in old_code.glob('*.py'):
        target = output/'code'/p.name; shutil.copyfile(p, target)
        bound[str(remote/'code'/p.name)] = sha(target)
    for name in ['probe_marshawn_LC_semantic_resolution_head.py', 'marshawn_LC_semantic_resolution_contract.py']:
        target = output/'code'/name; shutil.copyfile(Path(__file__).with_name(name), target)
        bound[str(remote/'code'/name)] = sha(target)
    bound[str(baseline_remote/'spec.json')] = sha(baseline_path)
    # Reuse the SAME fit-label bytes and SAME source contract, not a derived copy.
    assert baseline['label_qualification'] == str(baseline_remote/'label_qualification.json')
    spec_hashes = {}; specs = {}
    for name, updates, resume in [('technical32', 32, None), ('technical_resume32', 32, 'technical32'),
                                  ('technical64', 64, None), ('prior001_fresh2064', 2064, None)]:
        technical = name.startswith('technical')
        spec = dict(baseline, phase='direct_LC_resolution256_four_UNKNOWN_'+name,
                    root_release=str(remote/'root_release.json'), output=str(remote/'runs'/name),
                    authorization_key='LC_resolution_technical_head_authorized' if technical else 'LC_resolution_first2064_head_authorized',
                    updates=updates, grid_resolution=256, resolution_experiment_family=FAMILY,
                    packet_history_reference=baseline['camera_history_reference'],
                    camera_history_reference=None if technical else baseline['camera_history_reference'],
                    matched_four_UNKNOWN128_spec=str(baseline_remote/'spec.json'),
                    controlled_change='Independent XYZ logit node resolution128->256 only; same FOUR-UNKNOWN labels and fresh camera packets.',
                    technical_gate_required_before_fresh2064=True)
        if resume: spec.update(resume=str(remote/'runs'/resume/'head.pt'), resume_initial_iteration=32)
        validate_matched_baseline(spec, baseline)
        p = output/'specs'/(name+'.json'); write(p, spec)
        specs[name] = spec; spec_hashes[str(remote/'specs'/p.name)] = sha(p)
    source_plan = json.loads((baseline_root/'controller_plan.json').read_text())
    host_name, host = next(iter(source_plan['hosts'].items())); host = dict(host)
    job_template = source_plan['jobs'][0]

    def job(name, dependencies):
        value = copy.deepcopy(job_template)
        value.update(name='LC_resolution256_four_UNKNOWN_'+name,
                     command=[host['python'], str(remote/'code/probe_marshawn_LC_semantic_resolution_head.py'),
                              '--spec', str(remote/'specs'/(name+'.json')), '--execute'],
                     cwd=str(remote/'code'), dependencies=dependencies,
                     progress_files=[str(remote/'runs'/name/'progress.json')],
                     metrics_file=str(remote/'runs'/name/'complete.json'))
        return value

    stages = {
        'technical': [job('technical32', []),
                      job('technical_resume32', ['LC_resolution256_four_UNKNOWN_technical32']),
                      job('technical64', ['LC_resolution256_four_UNKNOWN_technical_resume32'])],
        'first2064_after_root_technical_gate': [job('prior001_fresh2064', [])],
    }
    controllers = {}
    for stage, jobs in stages.items():
        settings = dict(host, root=str(remote/'queue'/stage))
        p = output/'controllers'/(stage+'.json')
        write(p, dict(schema_version=1, poll_seconds=15, hosts={host_name: settings}, jobs=jobs))
        controllers[str(remote/'controllers'/p.name)] = sha(p)
    new_worker = output/'code/probe_marshawn_LC_semantic_resolution_head.py'
    (output/'head_worker_vs_sealed128.diff').write_text(''.join(difflib.unified_diff(
        old_worker.read_text().splitlines(True), new_worker.read_text().splitlines(True),
        fromfile='sealed128/probe_marshawn_LC_frozen_semantic_head.py', tofile=new_worker.name)))
    write(output/'root_review_TEMPLATE.json', dict(reviewer='root',
        LC_resolution_technical_head_authorized=False, LC_resolution_first2064_head_authorized=False,
        LC_resolution_controller_technical_authorized=False, LC_resolution_controller_first2064_authorized=False,
        LC_physical_render_authorized=False, production_after_visualgate=False,
        label_qualification_sha256=sha(label_path), bound_files=bound, spec_sha256=spec_hashes,
        controller_plan_sha256=controllers,
        technical_root_review_required_before_fresh2064=True,
        technical_gate_required=['actual25632+32/64', 'exact_resume_START_head_Adam_RNG_sampler',
                                 'reported_CUDA_atomic_trajectory_drift', 'measured_peak_allocator_below8GiB_no_OOM',
                                 'unchanged_native_cache_and_source', 'root_explicit_numeric_and_resource_acceptance'],
        physical_gate='Separate constructor patch and bound render release required; no physical integration prepared here.'))
    write(output/'CPU_readiness.json', dict(CPU_only=True, CUDA_started=False, optimizer_created=False,
        remote_root=str(remote), source_worker_sha256=sha(old_worker), baseline_four_UNKNOWN128_spec_sha256=sha(baseline_path),
        shared_label_qualification_sha256=sha(label_path), source_cache_UNKNOWN_count=2, fit_overlay_UNKNOWN_count=4,
        baseline128_unchanged=True, original_two_UNKNOWN_baseline_unchanged=True,
        grids_allowed=[128, 256], selected_grid_resolution=256, initial_probability=.01, seed=42, learning_rate=.03,
        fresh_updates=2064, same2064_camera_reference=baseline['camera_history_reference'],
        source_bound_files_verified=len(old_release['bound_files']), new_execution_bound_files=len(bound),
        local_archive_evidence_aliases=local_archive_aliases,
        all_permissions_false=True, controller_started=False, stage_order=list(stages),
        dense_FP32_grid_bytes=256**3*4, dense_Adam_minimum_bytes=256**3*16,
        GPU_allocator_limit_bytes=8*2**30, actual_memory_fit_not_yet_measured=True,
        physical_constructor_status='proposal-only diff in repository; existing renderer untouched',
        next_gate='Root separately authorizes256 technical only, reviews actual results, then decides fresh2064.'))
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=C)
    parser.add_argument('--four-unknown-plan', type=Path, default=C/'frozen_LC_semantic_LA_two_UNKNOWN_candidate_v1')
    parser.add_argument('--output', type=Path, default=C/'frozen_LC_semantic_resolution256_fourUNK_plan_v1')
    args = parser.parse_args(); prepare(args.campaign, args.four_unknown_plan, args.output)
