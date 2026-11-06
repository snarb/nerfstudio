"""Actual native Adam/resume proof plus corrected-confidence checkpoint binding."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch
from verify_marshawn_semantic_smoke import main as verify_semantic
from marshawn_bag_loss_correction import sha,validate_manifest

def main(a):
    manifest=validate_manifest(a.manifest,sha(a.manifest))
    base=a.output.with_name(a.output.stem+'_semantic_base.json')
    verify_semantic(SimpleNamespace(before=a.before,after=a.after,run=a.run,output=base))
    report=json.loads(base.read_text());application=json.loads((a.run/'bag_loss_application.json').read_text());bindings=[]
    for path in [a.before,a.after]:
        payload=torch.load(path,map_location='cpu',weights_only=False);runtime=payload['tuning_runtime'];binding=runtime['sampling_state']['bag_loss_correction']
        if binding['manifest_sha256']!=sha(a.manifest)or binding['changed_pixels']!=6522 or len(binding['effective_weights_sha256'])!=63:raise ValueError('Saved corrected-confidence state mismatch')
        if runtime['bag_loss_application']!=application or application['state_binding']!=binding:raise ValueError('Runtime application differs across resume')
        if runtime['input_sha256']['bag_loss_manifest']!=sha(a.manifest):raise ValueError('Bag source missing from durable input identity')
        for name,digest in manifest['files_sha256'].items():
            if runtime['input_sha256']['bag_loss/'+name]!=digest:raise ValueError('Saved bag source digest differs')
        bindings.append(binding);del payload
    if bindings[0]!=bindings[1]:raise ValueError('Effective confidence changed on resume')
    before,after=report['before'],report['after']
    if after['sampling_state']['sample_count']-before['sampling_state']['sample_count']!=after['attempted_step']-before['attempted_step']:raise ValueError('Actual sampler count did not resume exactly')
    report.update(kind='SEMANTIC_BAG_LOSS_NATIVE_ADAM_RESUME_TECHNICAL',bag_loss_manifest_sha256=sha(a.manifest),corrected_confidence_state_binding=bindings[0],application_receipt_sha256=sha(a.run/'bag_loss_application.json'),corrected_RGB_confidence_pixels=6522,raw_masks_proposal_pools_fixed_admission_unchanged=True,new_objective_actual_Adam_before_and_after=True,quality_claim=False,verifier_sha256=sha(__file__))
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,adam_before=before['adam_steps'],adam_after=after['adam_steps'],corrected_pixels=6522)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['before','after','run','output','manifest']:p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
