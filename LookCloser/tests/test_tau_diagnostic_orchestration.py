"""Only new diagnostic authorization/retention behavior; no GPU/core guard replay."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock


def module(name):
    adjacent = {'guard_marshawn_tau_diagnostic_resources': 'guardian.py',
                'retain_marshawn_tau_diagnostic_gates': 'retention.py'}
    path = Path(__file__).resolve().parent / adjacent[name]
    if not path.exists(): path = Path(__file__).resolve().parents[1] / 'scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result


guard = module('guard_marshawn_tau_diagnostic_resources')
retain = module('retain_marshawn_tau_diagnostic_gates')


class Authorization(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        (self.root / 'proof.json').write_text(json.dumps({'passed': True}))
        self.policy = dict(phase_plans={'diagnostic': 'external_guard_v3/diagnostic_plan.json'},
                           technical_verification_sha256=guard.sha(self.root / 'proof.json'), technical_verification_relative='proof.json',
                           deadline_seconds_per_invocation=3600, process_cap_MiB=16384, worker_allocator_GiB=10)
        self.release = dict(authorized_phases=['diagnostic'], technical_verification_sha256=self.policy['technical_verification_sha256'])

    def test_exact_proof(self): guard.require_phase_proof(self.policy, self.root, 'diagnostic', self.release)
    def test_wrong_root_proof(self):
        with self.assertRaises(ValueError): guard.require_phase_proof(self.policy, self.root, 'diagnostic', dict(self.release, technical_verification_sha256='wrong'))
    def test_mutated_proof(self):
        (self.root / 'proof.json').write_text('{"passed":false}')
        with self.assertRaises(ValueError): guard.require_phase_proof(self.policy, self.root, 'diagnostic', self.release)
    def test_unreleased_phase(self):
        with self.assertRaises(ValueError): guard.require_phase_proof(self.policy, self.root, 'technical', self.release)
    def test_resource_drift(self):
        with self.assertRaises(ValueError): guard.require_phase_proof(dict(self.policy, deadline_seconds_per_invocation=3601), self.root, 'diagnostic', self.release)


class Retention(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name) / 'remote'; self.root.mkdir()
        self.out = Path(self.tmp.name) / 'local'; self.step = retain.GATES[0]; self.run = self.root / 'runs/diagnostic_control'; self.run.mkdir(parents=True)

    def fixture(self):
        spec = self.root / 'specs/diagnostic_control/spec.json'; spec.parent.mkdir(parents=True)
        spec.write_text(json.dumps(dict(mode='diagnostic', output=str(self.run), parent_checkpoint_sha256='source', worker_sha256='worker')))
        for name in ['source_inventory.json', 'anchor_audit.json']: (self.root / name).write_text('{}')
        cp = self.run / 'checkpoints' / f'step-{self.step:09d}.ckpt'; cp.parent.mkdir(); cp.write_bytes(b'original full state')
        archive = cp.with_name(cp.stem + '_grid.npz'); archive.write_bytes(b'grid')
        cp.with_name(cp.stem + '_grid.json').write_text(json.dumps(dict(checkpoint_step=self.step, checkpoint=str(cp), checkpoint_sha256=retain.sha(cp), grid_archive_sha256=retain.sha(archive))))
        render = self.run / 'renders' / f'step-{self.step:09d}'; render.mkdir(parents=True)
        for name in ['DC.png','HC.png','KA.png','DC_alpha.png','HC_alpha.png','KA_alpha.png']: (render / name).write_bytes(name.encode())
        (render / 'metrics.json').write_text(json.dumps(dict(step=self.step)))
        clear = self.run / 'clear_targets' / f'step-{self.step:09d}.json'; clear.parent.mkdir(); clear.write_text('{}')
        (self.run / 'latest.json').write_text(json.dumps(dict(step=self.step, checkpoint=str(cp.relative_to(self.run)), sha256=retain.sha(cp))))
        return cp

    def test_uncommitted_gate_not_copied(self):
        (self.run / 'latest.json').write_text(json.dumps(dict(step=self.step-1)))
        self.assertIsNone(retain.gate_record(self.root, 'control', self.step))

    def test_atomic_review_only_commit(self):
        cp = self.fixture(); record = retain.gate_record(self.root, 'control', self.step)
        retain.commit_gate(record, self.out, shutil.copyfile)
        self.assertTrue(cp.exists()); self.assertEqual(list(self.out.rglob('*.ckpt')), [])
        dest = self.out / 'control' / f'step-{self.step:09d}'
        self.assertTrue((dest / 'retention.json').exists()); self.assertEqual(len(record['files_sha256']), 10)
        retain.commit_gate(record, self.out, shutil.copyfile)

    def test_checkpoint_hash_failure(self):
        cp = self.fixture(); cp.write_bytes(b'changed')
        with self.assertRaises(ValueError): retain.gate_record(self.root, 'control', self.step)

    def test_copy_failure_never_publishes(self):
        self.fixture(); record = retain.gate_record(self.root, 'control', self.step)
        def corrupt(source, target): target.write_bytes(b'bad')
        with self.assertRaises(ValueError): retain.commit_gate(record, self.out, corrupt)
        self.assertFalse((self.out / 'control' / f'step-{self.step:09d}').exists())
        retain.commit_gate(record, self.out, shutil.copyfile)
        self.assertTrue((self.out / 'control' / f'step-{self.step:09d}' / 'retention.json').exists())

    def test_terminal_incomplete_is_bounded(self):
        package = self.root / 'external_guard_v3'; (package / 'diagnostic').mkdir(parents=True)
        (package / 'diagnostic/local_done.json').write_text('{"passed":false}')
        policy = package / 'policy.json'; policy.write_text('{}')
        with mock.patch.object(retain, 'policy_at', return_value={'local_root': str(self.root)}), \
             mock.patch.object(retain, 'run_once', return_value={'retained_gates': 0}) as run, \
             mock.patch.object(retain.time, 'sleep'):
            with self.assertRaises(RuntimeError): retain.watch(policy)
        self.assertEqual(run.call_count, 3)
        self.assertTrue((package / 'retained_gates/failure.json').exists())


if __name__ == '__main__': unittest.main()
