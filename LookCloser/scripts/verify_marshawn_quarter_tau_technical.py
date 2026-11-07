"""Independent CPU verification of frozen quarter91128 tau technical segments.

Loads tensors on CPU and replays only the existing private packet sampler.
No model, optimizer, GPU context, training or source mutation is performed.
"""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


KEYS = ('pipeline', 'optimizers', 'schedulers', 'scalers', 'rng_state', 'tuning_runtime', 'actual_occupancy_capture')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def absolute(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise ValueError('Frozen helper import mismatch')
    return module


def check_segment(checks, first, last, before, after):
    if [r['step'] for r in checks] != list(range(first, last + 1)):
        raise ValueError('Missing or reordered attempted updates')
    successful = sum(bool(r['Adam_updated']) for r in checks)
    if not successful or any(r['Adam_updated'] and not r['combined_gradients_finite'] for r in checks):
        raise ValueError('Actual finite Adam updates required')
    for name in before:
        if not before[name] or not after[name] or set(after[name]) != {v + successful for v in before[name]}:
            raise ValueError('Adam counters disagree with recorded successful updates')
    return dict(attempts=len(checks), successful_updates=successful, skipped=len(checks)-successful,
                source_Adam=before, final_Adam=after)


def retention_proof(worker_path):
    source = worker_path.read_text()
    tree = ast.parse(source)
    destructive = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in ('unlink', 'rmdir', 'rmtree')]
    gates = (500, 1000, 2000, 4000, 8000, 12000)
    tuples = [tuple(v.value for v in n.elts) for n in ast.walk(tree)
              if isinstance(n, ast.Tuple) and all(isinstance(v, ast.Constant) for v in n.elts)]
    if destructive or gates not in tuples or 'if gate or partial or terminal or stop: save(step)' not in source:
        raise ValueError('Expected all-six-gate checkpoint retention contract absent')
    return dict(diagnostic_relative_gates=list(gates), absolute_gates=[91128+x for x in gates],
                save_at_every_gate=True, pruning_calls_absent=True,
                limitation='Static frozen-worker contract; future production existence still requires runtime archive checks.')


def main(a):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if a.output.exists():
        raise ValueError('Preserve previous verification generation')
    code = a.root / 'code'
    scripts = code / 'LookCloser/scripts'
    inventory = read(a.root / 'source_inventory.json')['files']
    for name, digest in inventory.items():
        if sha(code / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    sys.path[:0] = [str(scripts), str(code)]
    worker_path = scripts / 'train_marshawn_quarter_clear_tau.py'
    worker = absolute('_frozen_quarter_tau_worker', worker_path)
    aux_module = absolute('_frozen_quarter_aux', scripts / 'marshawn_quarter_clear_tau.py')
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise ValueError('CPU verifier initialized CUDA')
    targets_path = a.root / 'targets_10481.json'
    targets = read(targets_path)
    anchor = read(a.root / 'anchor_audit.json')
    arms = {}
    summaries = {}
    source_payload = None
    for name in ('control', 'old7369', 'seven10481'):
        spec_path = a.root / 'specs' / ('technical_' + name) / 'spec.json'
        spec = read(spec_path)
        if sha(targets_path) != spec['targets_sha256']:
            raise ValueError('Private packet target source differs')
        if source_payload is None:
            if sha(spec['parent_checkpoint']) != spec['parent_checkpoint_sha256']:
                raise ValueError('Source checkpoint hash differs')
            source_payload = torch.load(spec['parent_checkpoint'], map_location='cpu', weights_only=False)
            source_hashes = {k: worker.tree_digest(source_payload[k]) for k in KEYS}
            if source_payload['step'] != 91128 or source_hashes != anchor['fullstate_hashes']:
                raise ValueError('Source seven-state anchor mismatch')
            source_adam = worker.adam_steps(source_payload['optimizers'])
            if any(v != [91087.] for v in source_adam.values()):
                raise ValueError('Unexpected source actual Adam state')
            source_frequency = worker.tree_digest(source_payload['pipeline']['_model.freq_grid.grid'])
            del source_payload
            source_payload = True
            gc.collect()
        run = a.root / 'runs' / ('technical_' + name)
        cp1, cp2 = [run / 'checkpoints' / f'step-{step:09d}.ckpt' for step in (91160, 91192)]
        first = torch.load(cp1, map_location='cpu', weights_only=False)
        final = torch.load(cp2, map_location='cpu', weights_only=False)
        if (first['step'], final['step']) != (91160, 91192):
            raise ValueError('Incorrect pause/resume checkpoints')
        resume = read(run / 'resume_proof_91160.json')
        first_hashes = {k: worker.tree_digest(first[k]) for k in KEYS}
        final_hashes = {k: worker.tree_digest(final[k]) for k in KEYS}
        if any(resume[k] != first_hashes[k] for k in KEYS):
            raise ValueError('Actual resumed seven states differ from paused checkpoint')
        one, two = first['fork_runtime'], final['fork_runtime']
        if worker.tree_digest(one['auxiliary_state']) != resume['auxiliary_state_sha256']:
            raise ValueError('Private auxiliary state not restored exactly')
        actual = [worker.adam_steps(x['optimizers']) for x in (first, final)]
        segments = [check_segment(one['gradient_checks'], 91129, 91160, source_adam, actual[0]),
                    check_segment(two['gradient_checks'], 91161, 91192, actual[0], actual[1])]
        if first_hashes['pipeline'] == source_hashes['pipeline'] or final_hashes['pipeline'] == first_hashes['pipeline']:
            raise ValueError('No actual pipeline change across a segment')
        events = two['frequency_events']
        if len(events) != 1 or events[0]['step'] != 91136 or not events[0]['finite'] or events != one['frequency_events']:
            raise ValueError('Actual original frequency update91136 missing or changed on resume')
        event = events[0]
        if event['grid_before'] != source_frequency:
            raise ValueError('Frequency refresh did not start from source grid')
        for payload in (first, final):
            if worker.tree_digest(payload['pipeline']['_model.freq_grid.grid']) != event['grid_after']:
                raise ValueError('Saved actual frequency grid differs from observed update')
        replay = aux_module.QualifiedClearTau(SimpleNamespace(device=torch.device('cpu')), targets,
                                             spec['targets_sha256'], spec['negative_tau_coefficient'], spec['target_population'])
        global_before = torch.get_rng_state().clone()
        for _ in range(32):
            replay.draw()
        if worker.tree_digest(replay.state_dict()) != worker.tree_digest(one['auxiliary_state']):
            raise ValueError('First32 private packet replay differs')
        # Invoke the real frozen restore helper, then replay the resumed32 packets.
        replay.load_state_dict(one['auxiliary_state'])
        for _ in range(32):
            replay.draw()
        if worker.tree_digest(replay.state_dict()) != worker.tree_digest(two['auxiliary_state']):
            raise ValueError('Private resumed packet stream differs')
        if not torch.equal(global_before, torch.get_rng_state()):
            raise ValueError('Private CPU packet replay changed global RNG')
        complete = read(run / 'complete.json')
        if complete['final_step'] != 91192 or complete['actual_adam_steps'] != actual[1]:
            raise ValueError('Completion metadata differs from actual checkpoint')
        saved_aux = two['auxiliary_state']
        summaries[name] = dict(first_main=one['sampling_segment'], resumed_main=two['sampling_segment'],
            rng_first=first_hashes['rng_state'], rng_final=final_hashes['rng_state'],
            generator=worker.tree_digest(saved_aux['generator_state']), jitter_chain=saved_aux['jitter_chain'],
            ids_chain=saved_aux['ids_chain'], packet_state_after=saved_aux['packet_state_after'])
        arms[name] = dict(segments=segments, checkpoint_sha256={str(cp1):sha(cp1),str(cp2):sha(cp2)},
            seven_state_pause=first_hashes, seven_state_final=final_hashes,
            actual_resume_seven_and_aux_equal=True, private_packet64_replay_equal=True,
            frequency_event=event, source_pipeline_changed=True, resume_pipeline_changed=True,
            private_global_CPU_RNG_unchanged=True, source_spec_sha256=sha(spec_path))
        del first, final, replay
        gc.collect()
    initial = [s['first_main']['first_preupdate_ray_indices_sha256'] for s in summaries.values()]
    if len(set(initial)) != 1:
        raise ValueError('First pre-update main rays differ across arms')
    old, new = summaries['old7369'], summaries['seven10481']
    if any(old[k] != new[k] for k in ('generator', 'jitter_chain', 'packet_state_after')) or old['ids_chain'] == new['ids_chain']:
        raise ValueError('Expected matched private jitter/state and distinct population IDs not observed')
    if torch.cuda.is_initialized():
        raise ValueError('CPU verifier initialized CUDA')
    result = dict(passed=True, arms=arms, streams=summaries, first_preupdate_main_indices_equal=True,
        private_generator_and_jitter_equal=True, population_ID_chains_intentionally_different=True,
        first32_main_segment_equal=len({s['first_main']['ray_indices_sha256'] for s in summaries.values()})==1,
        resumed32_main_segment_equal=len({s['resumed_main']['ray_indices_sha256'] for s in summaries.values()})==1,
        global_rng_first_equal=len({s['rng_first'] for s in summaries.values()})==1,
        global_rng_final_equal=len({s['rng_final'] for s in summaries.values()})==1,
        RNG_interpretation='Exact source/resume and independent private packets are proven. Later main/global streams are observed, not forced equal or reseeded: changed learning can alter native adaptive callback/sample draws. This verifier does not claim to locate the first causal CUDA RNG divergence.',
        checkpoint_retention=retention_proof(worker_path), source_hashes=source_hashes,
        worker_sha256=sha(worker_path), verifier_sha256=sha(__file__), source_inventory_sha256=sha(a.root/'source_inventory.json'),
        model_instantiated=False, optimizer_instantiated=False, GPU_initialized=False, production_authorized=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ('passed','first32_main_segment_equal','resumed32_main_segment_equal','global_rng_first_equal','global_rng_final_equal')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
