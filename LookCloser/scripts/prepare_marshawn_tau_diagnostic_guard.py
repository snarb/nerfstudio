"""Stage orchestration-only v3; never launch or change the frozen scientific fork."""
import argparse
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

TECHNICAL = '08d4055a816311e4642f80d65e43c5e16175fe9d14a3315fffe9864ac0464309'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, v): Path(p).write_text(json.dumps(v, indent=2) + '\n')


def main(root):
    root = root.resolve(); out = root / 'external_guard_v3'
    if out.exists(): raise ValueError('Preserve the existing generation')
    old = root / 'external_guard_v2'; policy = read(old / 'policy.json')
    assert policy['version'] == 2 and sha(old / 'guardian.py') == policy['guardian_sha256']
    proof = root / 'CPU_technical_verification_v2/verification.json'
    assert sha(proof) == TECHNICAL and read(proof)['passed']
    for rel, h in policy['prepared_bindings'].items(): assert sha(root / rel) == h, rel
    repo = Path(__file__).resolve().parents[1]
    old_code = (old / 'guardian.py').read_text(); new_code = (repo / 'scripts/guard_marshawn_tau_diagnostic_resources.py').read_text()
    old_funcs = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(old_code).body if isinstance(n, ast.FunctionDef)}
    new_funcs = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(new_code).body if isinstance(n, ast.FunctionDef)}
    unchanged = ['identity','same','authorized_worker','event','fail','check_limits','bounded_error','clear_errors','phase_status','signal_due','final_record','ssh_script']
    assert all(old_funcs[k] == new_funcs[k] for k in unchanged)
    plan = read(root / 'diagnostic_plan.json'); adjusted = copy.deepcopy(plan)
    adjusted['hosts']['silver']['gpus'] = [0, 1, 2, 3, 5, 6]
    assert adjusted['jobs'] == plan['jobs'] and len(adjusted['jobs']) == 3
    out.mkdir()
    shutil.copy2(repo / 'scripts/guard_marshawn_tau_diagnostic_resources.py', out / 'guardian.py')
    shutil.copy2(repo / 'scripts/retain_marshawn_tau_diagnostic_gates.py', out / 'retention.py')
    shutil.copy2(repo / 'tests/test_tau_diagnostic_orchestration.py', out / 'test_tau_diagnostic_orchestration.py')
    shutil.copy2(__file__, out / 'prepare.py')
    write(out / 'diagnostic_plan.json', adjusted)
    (out / 'guardian.diff').write_text(''.join(difflib.unified_diff(old_code.splitlines(True), new_code.splitlines(True), fromfile='external_guard_v2/guardian.py', tofile='external_guard_v3/guardian.py')))
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(repo / 'tests'), '-p', 'test_tau_diagnostic_orchestration.py', '-v'], text=True, capture_output=True)
    (out / 'CPU_tests.log').write_text(result.stdout + result.stderr)
    if result.returncode: raise RuntimeError('New orchestration tests failed; preserve staging evidence')
    write(out / 'CPU_tests.json', dict(passed=True, new_tests=9, prior_guard_core_tests_not_repeated=True,
                                      unchanged_guard_functions=unchanged, comparison='AST exact excluding line offsets', GPU_used=False))
    write(out / 'preservation.json', dict(base_guardian_sha256=sha(old / 'guardian.py'), base_policy_sha256=sha(old / 'policy.json'),
                                         base_guard_release_sha256=sha(old / 'guard_release.json'),
                                         original_diagnostic_plan_sha256=sha(root / 'diagnostic_plan.json'),
                                         source_bindings=len(policy['prepared_bindings']), scientific_specs_changed=False,
                                         diagnostic_worker_sha256=read(proof)['worker_sha256'], technical_verification_sha256=TECHNICAL,
                                         original_parent_checkpoint_sha256='81a226cdc96a437b221d9891ab8eac5bb9aa4ca5e5179ee03f92ceb16002c9a9',
                                         full_checkpoints_retained_remotely=True, pruning=False))
    (out / 'README.md').write_text('''# External diagnostic guard v3 — preparation only

Three fresh 12000-attempt forks from original quarter.01 step91128; technical checkpoints are excluded ancestry. This tests qualified clear-ray target population on that older parent, not the current best recipe.

Science/specs/worker remain unchanged. Only orchestration changes: diagnostic authorization, exact passing technical-proof binding, v3 paths, worker deadline3600s, and dispatcher GPU allowlist0/1/2/3/5/6 excluding frequency-fit4/7. Process16GiB and allocator10GiB remain. PID/start/boot/argv, UUID, sticky failure, heartbeat90s, TERM25s escalation and failed-reaper logic are retained. Twelve guard functions are AST-identical; nine new CPU tests cover proof authorization and atomic artifact retention. The prior27 core tests are not repeated.

The new guard release must bind policy.json and technical verification SHA. All three unchanged specs also need their own diagnostic_authorized root releases. Before dispatch, existing diagnostic outputs or remote job claims are rejected. No launch has occurred through this preparation.

Use local-origin guardian.py launch --policy THIS/policy.json --phase diagnostic, then retention.py launch --policy THIS/policy.json. The guard and CPU fetcher are separate detached processes; fetch failures never alter or signal scientific workers. Both require the exact root release. Copy the reviewed package, proof and releases to the same relative remote paths before launch. Guard state is THIS/diagnostic; local dispatcher state THIS/diagnostic_controller; CPU artifacts THIS/retained_gates.

All six worker checkpoints per arm (91628,92128,93128,95128,99128,103128) stay remote with no pruning. A CPU fetcher polls every30s and only copies a gate after atomic latest.step has reached it. It verifies actual checkpoint/grid hashes and raster metric step; copies six PNGs, metrics, clear-target diagnostic and grid archive/receipt into an atomic local directory; records source/spec/worker hashes and the full checkpoint's remote path/hash/size. It never copies or loads the full checkpoint. Later polls check older checkpoint existence/size and receipt digest again. A partially transferred gate is not published; failed staging remains for audit. After three fetch errors it records failure for explicit recovery, while all remote checkpoints remain preserved. Final validation must account for all18 retained gates.

No architecture, mask, optimizer, prior, sampler, frequency map, source RGB, accepted box, or scientific stop budget is changed. Root review and exact releases are required before these prepared commands can run.
''')
    policy.update(version=3, guardian_sha256=sha(out / 'guardian.py'), retention_sha256=sha(out / 'retention.py'),
                  base_v2_policy_sha256=sha(old / 'policy.json'), deadline_seconds_per_invocation=3600,
                  phase_plans={'diagnostic': 'external_guard_v3/diagnostic_plan.json'},
                  technical_verification_relative='CPU_technical_verification_v2/verification.json', technical_verification_sha256=TECHNICAL)
    for rel in ['CPU_technical_verification_v2/verification.json', 'CPU_technical_verification_v2/verifier.py']:
        policy['prepared_bindings'][rel] = sha(root / rel)
    for file in out.iterdir():
        if file.is_file(): policy['prepared_bindings'][str(file.relative_to(root))] = sha(file)
    write(out / 'policy.json', policy)
    write(out / 'guard_release.template.json', dict(reviewer='root', external_guard_authorized=False, policy_sha256=sha(out / 'policy.json'),
                                                   authorized_phases=['diagnostic'], technical_verification_sha256=TECHNICAL, phase_releases_also_required=True))
    print(json.dumps(dict(policy_sha256=sha(out / 'policy.json'), guardian_sha256=sha(out / 'guardian.py'),
                          retention_sha256=sha(out / 'retention.py'), bindings=len(policy['prepared_bindings']), launched=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
