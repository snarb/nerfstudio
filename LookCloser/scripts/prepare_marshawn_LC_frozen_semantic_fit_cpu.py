"""Bind completed direct-LC visibility to a new independent head-only proposal.

This CPU materializer does not launch workers or create root authorization.
Derived generic receipts are explicitly adapters, not new source captures.
"""
import argparse
import difflib
import json
from pathlib import Path
import shutil
import numpy as np
from prepare_marshawn_frozen_semantic_cpu import sha, write


def qualify_capture(plan_path, receipt_path):
    m, receipt = [json.loads(p.read_text()) for p in [plan_path, receipt_path]]
    if not receipt.get('passed') or not all(receipt['unchanged'].values()):
        raise ValueError('Actual completed immutable direct-LC capture required')
    if receipt['source_step'] != 126316 or receipt['source_FR'] != .3 or receipt['source_checkpoint_sha256'] != '340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641':
        raise ValueError('Wrong frozen LC source')
    if m['world_to_box'] != np.eye(4).tolist() or set(m['train_cameras']) & {'DC', 'HC', 'KA'} or len(m['train_cameras']) != 63:
        raise ValueError('LC TRAIN63/identity-floor contract differs')
    records, count, changes, model_digest = [], 0, [], None
    sources = {r['camera']: r for r in m['records']}
    hashes = {str(plan_path): sha(plan_path), str(receipt_path): sha(receipt_path)}
    initial = json.loads((receipt_path.parent/'initial.json').read_text())
    if initial['manifest_sha256'] != sha(plan_path):
        raise ValueError('Capture initial state belongs to another plan')
    hashes[str(receipt_path.parent/'initial.json')] = sha(receipt_path.parent/'initial.json')
    for row in receipt['records']:
        if not row['rays']:
            records.append(dict(row)); continue
        p, q = Path(row['archive']), Path(sources[row['camera']]['archive'])
        if sha(p) != row['archive_sha256'] or sha(q) != sources[row['camera']]['archive_sha256']:
            raise ValueError('Native LC or original fixed query archive changed')
        hashes[str(p)], hashes[str(q)] = sha(p), sha(q)
        with np.load(p, allow_pickle=False) as z, np.load(q, allow_pickle=False) as original:
            digest = str(z['model_digest'].item())
            if model_digest is not None and digest != model_digest: raise ValueError('Mixed source pipeline digests')
            model_digest = digest
            if not np.array_equal(z['world_to_box'], np.eye(4)) or not np.array_equal(z['pixel_xy'], original['pixel_xy']) or not np.array_equal(z['semantic_target'], original['semantic_target']):
                raise ValueError('Do not change floor coordinates, pixel packets or semantic targets')
            delta = np.flatnonzero(z['label_weight'] != original['label_weight'])
            for index in delta:
                if z['semantic_target'][index] != 1 or z['label_weight'][index] != 0 or original['label_weight'][index] != 1:
                    raise ValueError('Only positive1 -> UNKNOWN confidence0 is qualified')
                changes.append((row['camera'], count+int(index), tuple(z['pixel_xy'][index])))
            if z['xyz'].shape != (row['samples'], 3) or z['weight'].shape != (row['samples'],):
                raise ValueError('Actual native visibility dimensions differ')
            if np.any(z['cap_hit'] & (z['num_samples_per_ray'] < 1024)):
                raise ValueError('Native cap diagnostic inconsistent')
        records.append(dict(row)); count += row['rays']
    expected = [('IA', 40470, (1013, 562)), ('IA', 40622, (1019, 567))]
    if sorted(changes) != expected or count != 69888 or sum(r['rays'] > 0 for r in records) != 60 or model_digest != initial['pipeline_sha256']:
        raise ValueError('Exact camera/query/UNKNOWN/pipeline accounting differs')
    if receipt['qualified_UNKNOWN_queries'] != [40470, 40622] and sorted(receipt['qualified_UNKNOWN_queries']) != [40470, 40622]:
        raise ValueError('Capture UNKNOWN accounting differs')
    return m, receipt, records, hashes, model_digest


def prepare(plan_path, receipt_path, output):
    if output.exists(): raise FileExistsError('Use a new sealed proposal namespace')
    m, source, records, hashes, digest = qualify_capture(plan_path, receipt_path)
    campaign = Path(m['campaign']); source_qualification = campaign/'frozen_native_semantic_head_technical_plan_v1/label_qualification.json'
    qualification = json.loads(source_qualification.read_text())
    if sha(source_qualification) != '944d88dbbfd24a0021c1166dbfe1fa949047b8d929b9b0cad65e6c84f07ff4f0' or not qualification['root_review_complete']:
        raise ValueError('Exact root two-query source qualification required')
    output.mkdir(parents=True); (output/'code').mkdir(); (output/'specs').mkdir()
    helper_names = ['probe_marshawn_LC_frozen_semantic_head.py', 'marshawn_frozen_semantic_head.py',
        'prepare_marshawn_frozen_semantic_cpu.py', 'run_marshawn_frozen_semantic_probe.py',
        'compare_marshawn_frozen_semantic_roundtrip_cpu.py', 'marshawn_frozen_semantic_selectors.py']
    for name in helper_names:
        p = output/'code'/name; shutil.copyfile(Path(__file__).with_name(name), p); hashes[str(p)] = sha(p)
    for name in ['marshawn_gpu_isolation.py']:
        p = output/'code'/name; shutil.copyfile(Path(m['helpers'])/name, p); hashes[str(p)] = sha(p)
    for key, value in m['bound_inputs'].items():
        if sha(key) != value: raise ValueError('Source binding changed: '+key)
        hashes[key] = value
    hashes[str(source_qualification)] = sha(source_qualification)
    for unknown in qualification['unknown_queries']:
        leaf=Path(unknown['qualification'])
        if sha(leaf)!=unknown['qualification_sha256']:
            raise ValueError('Transitive root-qualified UNKNOWN evidence changed')
        hashes[str(leaf)]=unknown['qualification_sha256']
    checkpoint = Path(m['local_checkpoint']); hashes[str(checkpoint)] = sha(checkpoint)
    head_manifest = dict(kind='DERIVED_DIRECT_LC_HEAD_INPUT_CONTRACT', source_kind='DIRECT_FROZEN_LC_NATIVE_VISIBILITY',
        train_cameras=m['train_cameras'], bounds=m['bounds'], native_world_to_box=np.eye(4).tolist(),
        native_source=dict(checkpoint=str(checkpoint), checkpoint_sha256=source['source_checkpoint_sha256']),
        LC_capture_plan=str(plan_path), LC_capture_plan_sha256=sha(plan_path),
        LC_capture_receipt=str(receipt_path), LC_capture_receipt_sha256=sha(receipt_path),
        source_pipeline_digest=digest, source_qualified_unknown_indices=[40470,40622],
        actual_native_ARM_cap=1024, interpretation='Independent LC head. No NGP head transfer. Native visibility cache is immutable; target semantics are not optical alphaGT.')
    manifest_path = output/'head_input_contract.json'; write(manifest_path, head_manifest)
    adapter_path = output/'capture_adapter_receipt.json'
    write(adapter_path, dict(kind='DERIVED_LC_NATIVE_VISIBILITY_SCHEMA_ADAPTER', original_receipt=str(receipt_path),
        original_receipt_sha256=sha(receipt_path), model_digest=digest, manifest_sha256=sha(manifest_path),
        native_model_unchanged=all(source['unchanged'].values()), records=records,
        disclaimer='Aliases existing full-pipeline immutability into generic head schema. This is not a second capture or a geometry/fullvisibility qualification.'))
    qualification_path = output/'label_qualification.json'
    write(qualification_path, dict(qualification, capture_receipt_sha256=sha(adapter_path),
        root_original_qualification=str(source_qualification), root_original_qualification_sha256=sha(source_qualification),
        note='Exact two root-reviewed TRAIN queries are already UNKNOWN in immutable LC cache. Override is idempotent; no new source label is qualified.'))
    history = campaign/'frozen_native_semantic_prior001_candidate_v1/camera_history_reference.json'
    reference = json.loads(history.read_text())
    sampler = np.random.default_rng(42); pool = [r['camera'] for r in records if r['rays']]
    actual = [pool[int(sampler.integers(60))] for _ in range(2064)]
    if actual != reference['camera_history']: raise ValueError('Fresh LC packet trajectory would differ from original prior pair')
    hashes[str(history)] = sha(history)
    spec_paths = []
    for name, prior, updates, resume in [('technical32',.99,32,None), ('technical_resume32',.99,32,'technical32'),
            ('technical64',.99,64,None), ('prior099_fresh2064',.99,2064,None), ('prior001_fresh2064',.01,2064,None)]:
        technical = name.startswith('technical')
        spec = dict(phase='direct_LC_'+name, root_release=str(output/'root_release.json'), manifest=str(manifest_path),
            capture_receipt=str(adapter_path), label_qualification=str(qualification_path),
            capacity_policy='KEEP_GENUINE_LABELS_AND_REPORT_IRREDUCIBLE_BOUND',
            authorization_key='LC_technical_head_authorized' if technical else 'LC_first2064_head_authorized',
            initial_probability=prior, updates=updates, torch_version=m['torch_version'], cuda_version=m['cuda_version'],
            output=str(output/'runs'/name), seed=42, grid_resolution=128, learning_rate=.03,
            world_to_box=np.eye(4).tolist(), native_model_updates=0,
            no_NGP_head_transfer=True, qualified_UNKNOWN_count=2,
            camera_history_reference=None if technical else str(history))
        if resume: spec.update(resume=str(output/'runs'/resume/'head.pt'), resume_initial_iteration=32)
        path = output/'specs'/(name+'.json'); write(path,spec); spec_paths.append(path)
    for p in [manifest_path, adapter_path, qualification_path]: hashes[str(p)] = sha(p)
    write(output/'release_TEMPLATE.json', dict(reviewer='root', LC_technical_head_authorized=False,
        LC_first2064_head_authorized=False, LC_physical_render_authorized=False,
        LC_uncapped_qualification_authorized=False, production_after_visualgate=False,
        label_qualification_sha256=sha(qualification_path), bound_files=hashes,
        spec_sha256={str(p):sha(p) for p in spec_paths}))
    old = Path(__file__).with_name('probe_marshawn_frozen_semantic_roundtrip.py').read_text().splitlines(True)
    new = (output/'code/probe_marshawn_LC_frozen_semantic_head.py').read_text().splitlines(True)
    (output/'head_worker_vs_native.diff').write_text(''.join(difflib.unified_diff(old,new,fromfile='NGP_head_worker',tofile='direct_LC_head_worker')))
    write(output/'CPU_readiness.json', dict(CPU_only=True, CUDA_started=False, optimizer_started=False,
        source_capture_actual=True, source_pipeline_digest=digest, capture_sha256=sha(receipt_path),
        bound_files=len(hashes), source_queries=69888, declared_train=63, usable_train=60,
        unknown_override_count=2, all_capacity_cases_retained=True, world_to_box_is_identity=True,
        exact2064_camera_packet_history_matches_native_pair=True,
        technical_resume='Exact head/Adam/RNG/sampler START; independent CUDA trajectory compared numerically, not bitwise.',
        release_template_all_permissions_false=True,
        physical_path_status='Head preparation only. Separate source-loader/physical package required; native ARM1024 tail limitations remain.'))
    print(output)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--capture-plan',type=Path,required=True)
    parser.add_argument('--capture-receipt',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); prepare(args.capture_plan,args.capture_receipt,args.output)
