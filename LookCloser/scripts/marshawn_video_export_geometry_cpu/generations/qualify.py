"""Finite CPU fixtures for routing and unchanged shader call interception."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import types

import entry
import policy
import route
import verify


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def run():
    results = {}
    original = {name: getattr(policy, name) for name in ['LOCAL', 'GEN1', 'SILVER', 'SGEN1', 'SEAL']}
    with tempfile.TemporaryDirectory(prefix='marshawn_render_generation_') as temporary:
        temporary = Path(temporary)
        policy.LOCAL, policy.GEN1 = temporary / 'C', temporary / 'G1'
        policy.SILVER, policy.SGEN1 = temporary / 'SGC', temporary / 'SGG1'
        root = policy.GEN1
        directory = root / 'frames/001695'
        claim = dict(host='olive', host_root=str(root), frame=1695,
                     model_generation=1, source_generation=1, source_root='/sourceG1',
                     assignment_nonce='one-immutable-claim', input_manifest_sha256='inputSHA')
        generation = dict(model_generation=1, source_generation=1, source_campaign='/sourceG1',
                          model_campaign_by_host={name: str(policy.expected_campaign(1695, name))
                                                  for name in ['olive', 'silver', 'dev3']})
        publication = dict(model_generation=1, source_generation=1, host_root=str(root),
                           assignment_nonce=claim['assignment_nonce'], assignment_sha256='claimSHA',
                           input_manifest_sha256='inputSHA', no_model_or_learned_state_staged=True,
                           regular_owned_sameframe_DATA_materialized=True,
                           frozen_source_support_verified={'verified': True},
                           dense_all66_prerequisite_passed=True)
        save(root / 'code_seal.json', {'fixture': True})
        policy.SEAL = policy.sha(root / 'code_seal.json')
        save(directory / 'data/ready.json', {'fixture': 'sourceG1'})
        save(directory / 'data/transforms.json', {'fixture': 'same-current-cameras'})
        publication['source_ready_sha256'] = policy.sha(directory / 'data/ready.json')
        save(directory / 'dispatch_ready.json', publication)
        save(directory / 'train/cold_setup_audit.json', dict(cold=True, resume=None,
             trainer_start_step=0, occupied_cells=0, nonzero_ema_cells=0, optimizer_state_sizes={'fields': 0}))
        save(directory / 'train/cold_provenance.json', {'fixture': 'fresh cold setup'})
        cp = directory / 'train/checkpoints/step-000139999.ckpt'
        cp.parent.mkdir(parents=True)
        cp.write_bytes(b'fixture checkpoint, not a torch model')
        completion = dict(technical=False, code=str(root / 'code'), final_step=139999,
                          cold_provenance_sha256=policy.sha(directory / 'train/cold_provenance.json'),
                          selectors={'actor_lpips': dict(step=139999, checkpoint=str(cp),
                                                        checkpoint_sha256=policy.sha(cp))})
        save(directory / 'frame_completion.json', completion)
        save(directory / 'train/result.json', dict(technical_smoke=False, hard_max_iterations=140000,
             completed_iterations=140000, final_step=139999))
        inputs = {str(p.relative_to(directory)): policy.sha(p) for p in directory.rglob('*.json')}
        row = dict(index=7, frame_id='001695', assignment=claim, generation=generation,
                   native_campaign=str(root), native_inputs_sha256=inputs,
                   checkpoint_sha256=policy.sha(cp), cold_provenance_sha256=completion['cold_provenance_sha256'],
                   native_dispatch=publication, generation_source_binding=publication,
                   assignment_sha256='claimSHA', local_source_inputs_sha256={},
                   local_data_template=str(root / 'frames/{frame_id}/data'))
        assert policy.validate_native(row) == directory
        results['valid_generation1_cold_native_proof'] = True
        overcap = copy.deepcopy(row)
        terminal = policy.read(directory / 'train/result.json')
        save(directory / 'train/result.json', dict(terminal, completed_iterations=140001, final_step=140000))
        overcap['native_inputs_sha256']['train/result.json'] = policy.sha(directory / 'train/result.json')
        try:
            policy.validate_native(overcap)
        except ValueError:
            results['overcap_actual_training_completion_rejected'] = True
        else:
            raise AssertionError('Overcap generation accepted')
        save(directory / 'train/result.json', terminal)
        bad = copy.deepcopy(claim)
        bad['host_root'] = str(policy.LOCAL)
        try:
            policy.validate_claim(1695, bad, generation)
        except ValueError:
            results['retired_1695_root_rejected_before_model_load'] = True
        else:
            raise AssertionError('Retired generation accepted')
        bad = copy.deepcopy(generation)
        bad['model_generation'] = 0
        try:
            policy.validate_claim(1695, claim, bad)
        except ValueError:
            results['generation_downgrade_rejected'] = True
        else:
            raise AssertionError('Generation downgrade accepted')
        audit = read_audit = policy.read(directory / 'train/cold_setup_audit.json')
        save(directory / 'train/cold_setup_audit.json', dict(audit, occupied_cells=1))
        warm = copy.deepcopy(row)
        warm['native_inputs_sha256']['train/cold_setup_audit.json'] = policy.sha(directory / 'train/cold_setup_audit.json')
        try:
            policy.validate_native(warm)
        except ValueError:
            results['generation1_nonzero_initial_grid_rejected'] = True
        else:
            raise AssertionError('Warm replacement accepted')
        save(directory / 'train/cold_setup_audit.json', read_audit)
        for name, evidence in [('missing', None), ('empty', {}), ('nonzero', {'fields': 1})]:
            modified = copy.deepcopy(read_audit)
            if evidence is None:
                modified.pop('optimizer_state_sizes')
            else:
                modified['optimizer_state_sizes'] = evidence
            save(directory / 'train/cold_setup_audit.json', modified)
            bad = copy.deepcopy(row)
            bad['native_inputs_sha256']['train/cold_setup_audit.json'] = policy.sha(directory / 'train/cold_setup_audit.json')
            try:
                policy.validate_native(bad)
            except ValueError:
                results['generation1_%s_optimizer_evidence_rejected' % name] = True
            else:
                raise AssertionError('Unproven zero optimizer accepted: ' + name)
        save(directory / 'train/cold_setup_audit.json', read_audit)
        manifest = dict(kind='FRAME_GENERATION_NATIVE_RENDER_MANIFEST_V1', records={'7': row})
        manifest_path = temporary / 'manifest.json'
        save(manifest_path, manifest)
        spec_path = temporary / 'spec.json'
        save(spec_path, dict(generation_manifest=str(manifest_path),
             generation_manifest_sha256=policy.sha(manifest_path),
             data_template='/__generation_entry_required__/{frame_id}/data'))
        shader = original['LOCAL'] / 'operations/video_geometry_cpu_params_v2/worker_params.py'
        before = policy.sha(spec_path)
        calls = []
        fake = types.SimpleNamespace(read=policy.read)
        fake.cpu = lambda a: calls.append(dict(effective=fake.read(a.spec), spec_path=str(a.spec), index=a.index))
        old_module, old_argv = entry.module, sys.argv
        entry.module = lambda path, name: fake
        sys.argv = ['entry.py', '--manifest', str(manifest_path), '--worker', str(shader), '--',
                    'cpu', '--code', str(original['LOCAL'] / 'code'), '--spec', str(spec_path),
                    '--path', str(temporary / 'path.json'), '--index', '7']
        # CPU fixture has no real assignment file to rehash. Give it an honest
        # fixture claim path/SHA, exactly as a built production manifest does.
        claim_path = temporary / 'claim.json'
        save(claim_path, claim)
        row['assignment_path'], row['assignment_sha256'] = str(claim_path), policy.sha(claim_path)
        save(manifest_path, manifest)
        spec = policy.read(spec_path)
        spec['generation_manifest_sha256'] = policy.sha(manifest_path)
        save(spec_path, spec)
        before = policy.sha(spec_path)
        try:
            entry.main()
        finally:
            entry.module, sys.argv = old_module, old_argv
        assert calls[0]['effective']['data_template'] == str(root / 'frames/{frame_id}/data')
        assert calls[0]['spec_path'] == str(spec_path) and policy.sha(spec_path) == before
        results['CPU_I_O_resolution_preserves_single_global_spec_bytes'] = True
        # GPU launch interception checks the actual shader entry and its native
        # code argument without importing Torch or executing a GPU query.
        row['assignment_sha256'] = publication['assignment_sha256']
        save(manifest_path, manifest)
        spec['generation_manifest_sha256'] = policy.sha(manifest_path)
        save(spec_path, spec)
        previous_cap = sys.modules.get('cap_loader')
        fake_cap = types.SimpleNamespace(ROOTS=set(), load_pipeline=lambda *a, **kw: ('fixture', {}))
        sys.modules['cap_loader'] = fake_cap
        previous_run, previous_argv = entry.runpy.run_path, sys.argv
        gpu_calls = []
        entry.runpy.run_path = lambda path, **kw: gpu_calls.append((str(path), list(sys.argv)))
        sys.argv = ['entry.py', '--manifest', str(manifest_path), '--worker', str(shader.parent / 'worker.py'), '--',
                    'gpu', '--code', str(root / 'code'), '--spec', str(spec_path),
                    '--path', str(temporary / 'path.json'), '--index', '7']
        try:
            entry.main()
        finally:
            entry.runpy.run_path, sys.argv = previous_run, previous_argv
            if previous_cap is None:
                del sys.modules['cap_loader']
            else:
                sys.modules['cap_loader'] = previous_cap
        assert fake_cap.ROOTS == {root} and len(gpu_calls) == 1
        assert gpu_calls[0][0] == str(shader.parent / 'worker.py')
        results['GPU_call_preserves_original_shader_and_native_campaign_code'] = True
        # Actual-GT launch must not touch manifest neural rows or2020 DATA.
        old_run, old_argv = entry.runpy.run_path, sys.argv
        captures = []
        entry.runpy.run_path = lambda path, **kw: captures.append((str(path), list(sys.argv)))
        for index in [72, 119]:
            sys.argv = ['entry.py', '--manifest', str(manifest_path), '--worker', str(shader.parent / 'one.py'), '--',
                        'gt', '--code', str(original['LOCAL'] / 'code'), '--spec', str(spec_path),
                        '--path', str(temporary / 'path.json'), '--index', str(index)]
            entry.main()
        entry.runpy.run_path, sys.argv = old_run, old_argv
        assert len(captures) == 2 and all('--cpu' in argv for _, argv in captures)
        results['GT72_and119_call_original_one_without_neural_DATA'] = True
        # A successful source-bound RPC proving missing/incomplete geometry is a
        # visible review failure. It is not indistinguishable from lost SSH.
        path = temporary / 'path.json'
        save(path, dict(records=[{}] * 8))
        missing = verify.verify(temporary / 'missing_geometry', manifest_path, path, 7)
        expected = dict(index=7, manifest_sha=policy.sha(manifest_path), path_sha=policy.sha(path),
                        verifier_sha=policy.sha(verify.__file__), policy_sha=policy.sha(policy.__file__))
        item = dict(status='failed', worker_alive=False, supervisor_alive=False)
        route.artifact_outcome(item, missing, **expected)
        assert item['status'] == 'blocked' and item['review_required'] is True
        results['proven_dead_missing_artifact_is_explicit_review_failure'] = True
        complete = dict(missing, verified=True)
        complete.pop('review_required')
        route.artifact_outcome(item, complete, **expected)
        assert item['status'] == 'complete'
        results['same_source_complete_artifact_can_be_adopted'] = True
        wrong = dict(complete, manifest_sha256='another-source')
        try:
            route.artifact_outcome(item, wrong, **expected)
        except ValueError:
            results['foreign_artifact_RPC_rejected'] = True
        else:
            raise AssertionError('Foreign artifact RPC accepted')
    for name, value in original.items():
        setattr(policy, name, value)
    return dict(kind='GENERATION_RENDER_CPU_CONTRACT_QUALIFICATION_V1', passed=True,
                tests=results, GPU_jobs_started=0, scientific_inputs_modified=False)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
