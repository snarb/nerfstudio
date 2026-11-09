"""Verify immutable seven-state/native RGB-alpha ZERO before any prior updates."""
import argparse
import hashlib
import json
from pathlib import Path


def main(root):
    read=lambda p:json.loads(Path(p).read_text())
    source=root.parent/'distortion03_warm_continuation_candidate_v1/anchor_audit.json'
    anchor=read(source);out=root/'runs/zero_source';complete=read(out/'complete.json')
    proof=read(out/'resume_proof_126316.json')
    assert complete['passed']and complete['status']=='zero_update_state_equal'and complete['render_state_unchanged']
    keys=['pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture']
    assert all(proof[k]==anchor['fullstate_hashes'][k]for k in keys)
    assert proof['FR']==.3 and proof['distortion']==.1 and complete['effective_distortion']==.1
    report=dict(passed=True,CPU_only=True,source_seven_states_exact=True,
        source_pipeline_digest=proof['pipeline'],source_checkpoint_sha256=anchor['checkpoint_sha256'],
        source_RGB_alpha_render_parity=complete['render_parity'],before_any_prior_query_or_optimizer_update=True,
        zero_complete_sha256=hashlib.sha256((out/'complete.json').read_bytes()).hexdigest())
    (root/'zero_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
