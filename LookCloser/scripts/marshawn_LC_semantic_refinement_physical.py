"""Explicit typed refinement-head inference constructor; no native field edits."""
import json
from pathlib import Path
from prepare_marshawn_frozen_semantic_cpu import sha
from marshawn_LC_semantic_refinement_contract import BASE_SHA,LABEL_SHA,FAMILY,construct_head,validate_spec


def physical_refinement_head(trained, completion, spec, bounds):
    import torch
    from marshawn_frozen_semantic_head import assert_exact_restoration,tensor_digest
    training_path=Path(spec['refinement_training_spec'])
    if sha(training_path)!=spec['head_training_spec_sha256']or trained['spec_sha256']!=sha(training_path):
        raise ValueError('Typed physical head requires the exact refinement training spec')
    training=json.loads(training_path.read_text());validate_spec(training)
    if training['head_type']!=spec['expected_head_type']:
        raise ValueError('Physical constructor type differs from declared refinement arm')
    for state in [trained,completion]:
        if (state.get('head_type')!=spec['expected_head_type']or state.get('head_schema_version')!=1 or
            state.get('refinement_family')!=FAMILY or state.get('base_checkpoint_sha256')!=BASE_SHA or
            state.get('iteration')!=2064 or state.get('qualified_UNKNOWN_count')!=4):
            raise ValueError('New typed2064 refinement state/completion required; never reinterpret a FrozenActorGrid archive')
    if completion.get('actual_head_updates')!=2064 or trained.get('label_qualification_sha256')!=LABEL_SHA:
        raise ValueError('Completed matched2064 refinement with exact FOUR-UNKNOWN labels required')
    bp=Path(spec['refinement_base_checkpoint'])
    if sha(bp)!=BASE_SHA:
        raise ValueError('Exact immutable learned coarse base required for physical inference')
    base=torch.load(bp,map_location='cpu',weights_only=False)
    head=construct_head(base,training,bounds)
    head.load_state_dict(trained['head'],strict=True);assert_exact_restoration(head,trained['head'])
    if tensor_digest(head)!=trained['head_digest']or not all(torch.isfinite(v).all()for v in head.state_dict().values()):
        raise ValueError('Refinement head digest or finite-state check failed')
    if training['arm']=='residual256'and not torch.equal(head.coarse_logits,base['head']['logits']):
        raise ValueError('Physical residual head must preserve the exact frozen learned128 base')
    head.eval();head.requires_grad_(False)
    return head
