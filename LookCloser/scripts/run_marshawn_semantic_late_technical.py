#!/usr/bin/env python3
"""Compare two zero-state proofs, then exercise real 32+32-attempt resume."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def audit_pair(control, treatment):
    if control['first32']['sampling_segment']['first_preupdate_ray_indices_sha256'] != treatment['first32']['sampling_segment']['first_preupdate_ray_indices_sha256']:
        raise ValueError('Initial paired pre-update rays differ')
    later = {}
    for label, a, b in [('first32',control['first32']['sampling_segment'],treatment['first32']['sampling_segment']),
                         ('resumed32',control['final']['sampling_segment'],treatment['final']['sampling_segment'])]:
        later[label] = a == b
    if not control['real_resume_nonempty_Adam'] or not treatment['real_resume_nonempty_Adam']:
        raise ValueError('Both actual Adam resume checks required')
    return {'initial_preupdate_rays_equal':True,'later_segment_equality':later,'later_divergence_requires_interpretation':not all(later.values()),'sampling_policy_changed':False,'both_actual_Adam_resumes_nonempty':True,'historical_cache_order_observed':False,'production_authorized':False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('worker', 'spec', 'zero-control', 'zero-treatment'):
        p.add_argument('--'+key, type=Path)
    p.add_argument('--audit-pair',nargs=2,type=Path)
    a = p.parse_args()
    if a.audit_pair:
        print(json.dumps(audit_pair(*[json.loads(x.read_text()) for x in a.audit_pair]))); return
    if any(x is None for x in (a.worker,a.spec,a.zero_control,a.zero_treatment)):
        p.error('Worker/spec and both zero proofs are required')
    control, treatment = [json.loads(x.read_text()) for x in (a.zero_control, a.zero_treatment)]
    if control['status'] != 'zero_update_state_equal' or treatment['status'] != 'zero_update_state_equal':
        raise ValueError('Both actual zero-update gates must pass first')
    for key in ('pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture','semantic_support','cache_sha256','cache_sources','sampler','next_step','Adam','FR'):
        if control['proof'][key] != treatment['proof'][key]: raise ValueError('Paired zero-state mismatch: '+key)
    spec = json.loads(a.spec.read_text())
    if spec['mode'] != 'technical': raise ValueError('Technical sequence cannot launch diagnostic pair')
    output = Path(spec['output'])
    latest = json.loads((output/'latest.json').read_text()) if (output/'latest.json').exists() else None
    if latest is None or latest['step'] < 30408:
        remaining = 30408-(latest['step'] if latest else 30376)
        subprocess.run([sys.executable, str(a.worker), '--spec', str(a.spec), '--stop-after-attempts', str(remaining)],check=True)
    import torch
    checkpoint = torch.load(output/'checkpoints/step-000030408.ckpt',map_location='cpu',weights_only=False)
    first = {'step':checkpoint['step'],'actual_adam_steps':{k:sorted({float(s['step']) for s in v['state'].values()}) for k,v in checkpoint['optimizers'].items()},'sampling_segment':checkpoint['fork_runtime']['sampling_segment'],'gradient_checks':checkpoint['fork_runtime']['gradient_checks']}
    del checkpoint
    first_record = output/'technical_first32.json'
    if not first_record.exists(): first_record.write_text(json.dumps(first,indent=2)+'\n')
    if not (output/'complete.json').exists():
        subprocess.run([sys.executable, str(a.worker), '--spec', str(a.spec)],check=True)
    result = json.loads((output/'complete.json').read_text())
    first = json.loads(first_record.read_text())
    for name, steps in first['actual_adam_steps'].items():
        if not steps or min(steps) <= min(control['proof']['Adam'][name]): raise ValueError('First32 attempts had no actual Adam updates')
        if min(result['actual_adam_steps'][name]) <= min(steps): raise ValueError('Resume had no additional Adam updates')
    for phase in (first,result):
        checks=phase['gradient_checks']
        if len(checks)!=32 or not any(c['Adam_updated'] for c in checks):raise ValueError('Each segment needs32 attempts and actual Adam updates')
        if any(c['Adam_updated'] and not c['combined_gradients_finite'] for c in checks):raise ValueError('Nonfinite updated combined gradient')
    (output/'technical_sequence.json').write_text(json.dumps({'zero_states_equal':True,'first32':first,'final':result,'real_resume_nonempty_Adam':True},indent=2)+'\n')


if __name__ == '__main__': main()
