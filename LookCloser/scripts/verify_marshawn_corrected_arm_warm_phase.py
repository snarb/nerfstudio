"""Verify old-flag ZERO and strict corrected ARM actual32+32 future receipts."""
import argparse
import json
import os
from pathlib import Path
import sys


def main(root, phase):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path[:0] = [str(root / 'code/LookCloser/scripts'), str(root / 'code')]
    import torch
    from marshawn_corrected_arm_warm_factory import ARMS, STATE_KEYS
    from train_marshawn_semantic_late_distortion import tree_digest, adam_steps
    read = lambda p: json.loads(Path(p).read_text())
    anchor = read(root / 'anchor_audit.json')
    reports = {}
    for arm, corrected in ARMS.items():
        out = root / 'runs' / (phase + '_' + arm)
        complete, proof = read(out / 'complete.json'), read(out / 'resume_proof_126316.json')
        assert complete['passed'] and proof['source_sampler_verified_before_treatment']
        assert proof['source_sampler'] == dict(corrected_arm_allocator=False, max_steps_per_ray=1024)
        expected = dict(corrected_arm_allocator=corrected, max_steps_per_ray=1024)
        assert proof['requested_sampler'] == expected and proof['distortion'] == .1 and proof['FR'] == .3
        for key in STATE_KEYS:
            assert proof[key] == anchor['fullstate_hashes'][key]
        if phase == 'zero':
            assert complete['status'] == 'zero_update_state_equal' and complete['render_state_unchanged']
            reports[arm] = dict(old_flag_source_parity=complete['render_parity'], source_seven_states_exact=True,
                                requested_sampler=expected, treatment_applied=False)
            continue
        first = torch.load(out / 'checkpoints/step-000126348.ckpt', map_location='cpu', weights_only=False)
        final = torch.load(out / 'checkpoints/step-000126380.ckpt', map_location='cpu', weights_only=False)
        resume = read(out / 'resume_proof_126348.json')
        for key in STATE_KEYS:
            assert resume[key] == tree_digest(first[key])
        middle, after = adam_steps(first['optimizers']), adam_steps(final['optimizers'])
        assert all(min(proof['Adam'][k]) < min(middle[k]) < min(after[k]) for k in after)
        assert len(complete['gradient_checks']) == 64 and complete['attempts'] == 64
        assert len(complete['source_control_checks']) == 2
        for payload in (first, final):
            runtime = payload['corrected_arm_fork_runtime']
            assert runtime['effective_sampler'] == expected and runtime['effective_distortion'] == .1
            assert all(r['normalizer_error'] < 1e-12 for r in runtime['sampling_segment']['packets'])
        assert resume['confidence_state_sha256'] == tree_digest(first['corrected_arm_fork_runtime']['confidence_state'])
        assert resume['model_cache_sha256'] == tree_digest(first['corrected_arm_fork_runtime']['model_cache'])
        assert all(r['actual_FR'] == .3 and r['actual_distortion'] == .1 and
                   (not r['Adam_updated'] or r['combined_gradients_finite']) for r in complete['gradient_checks'])
        assert all(r['effective_sampler'] == expected and r['source_W64_exact'] and
                   0 < r['packed_max_samples_per_ray'] <= 1024 for r in complete['source_control_checks'])
        reports[arm] = dict(source_seven_states_exact=True, child_seven_states_exact=True,
            effective_sampler=expected, first_Adam=middle, final_Adam=after,
            first32_indices=first['corrected_arm_fork_runtime']['sampling_segment']['indices_sha256'],
            resume32_indices=final['corrected_arm_fork_runtime']['sampling_segment']['indices_sha256'])
        del first, final
    assert not torch.cuda.is_initialized()
    (root / (phase + '_verification.json')).write_text(json.dumps(dict(passed=True, phase=phase,
        arms=reports, GPU_quality_approved=False, production_authorized=False), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['zero', 'technical'], required=True)
    args = parser.parse_args()
    main(args.root, args.phase)
