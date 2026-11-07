"""Explicit semantic-only refinement stage; imports never initialize CUDA.

Both arms restart Adam and sampling. The old2064 optimizer is deliberately not
restored: the matched factor is refinement parameterization at one basefunction.
"""
from prepare_marshawn_frozen_semantic_cpu import sha

FAMILY = 'DIRECT_LC_FOUR_UNKNOWN_MATCHED_REFINEMENT_V1'
BASE_SHA = '7937ab3df744254ea4b4087f8efda9384888e3bccbf833719bc45945b8a81beb'
LABEL_SHA = '44e7734ed93a2c10742de6dc55886978d7c2097ee62779915dada93feda94df4'
PIPELINE_SHA = 'c83bc7dbb1504063c35f0a2b14ae3fb329ff27ed2eb6b3661eb89b4918d46636'
CAPTURE_SHA = 'fb8b875d8c7f3676c3dfbd187b4c76c3ee40ec8a3b62a940fdf33bb97d99073d'
ARMS = {'coarse128': 'trainable_coarse128_v1', 'residual256': 'frozen_coarse128_plus_residual256_v1'}


def validate_spec(spec):
    if (spec.get('refinement_family') != FAMILY or spec.get('arm') not in ARMS or
            spec.get('head_type') != ARMS[spec['arm']] or spec.get('head_schema_version') != 1 or
            spec.get('seed') != 42 or spec.get('learning_rate') != .03 or
            spec.get('base_checkpoint_sha256') != BASE_SHA or
            spec.get('label_qualification_sha256') != LABEL_SHA or
            spec.get('native_model_updates') != 0 or
            spec.get('stage_optimizer_policy') != 'EXPLICIT_NEW_STAGE_RESET_ADAM0_AND_SAMPLER_SEED42' or
            spec.get('capacity_policy') != 'KEEP_GENUINE_LABELS_AND_REPORT_IRREDUCIBLE_BOUND' or
            spec.get('qualified_UNKNOWN_count') != 4):
        raise ValueError('Matched refinement contract changed')
    if spec.get('updates') not in (32, 64, 2064):
        raise ValueError('Only bounded technical or2064 refinement stages supported')
    if spec.get('resume') and (spec['updates'] != 32 or spec.get('resume_initial_iteration') != 32):
        raise ValueError('Only exact technical32 continuation allowed')
    if not spec.get('packet_history_reference'):
        raise ValueError('Fixed2064 camera history binding required')
    return spec['head_type']


def checkpoint_metadata(spec):
    validate_spec(spec)
    return dict(head_type=spec['head_type'], head_schema_version=1, refinement_family=FAMILY,
                arm=spec['arm'], base_checkpoint_sha256=BASE_SHA,
                stage_optimizer_policy=spec['stage_optimizer_policy'], initial_seed=42, learning_rate=.03,
                qualified_UNKNOWN_count=4, source_manifest_sha256=sha(spec['manifest']),
                capture_adapter_sha256=sha(spec['capture_receipt']),
                packet_history_reference_sha256=sha(spec['packet_history_reference']))


def construct_head(base, spec, bounds):
    """Validate the immutable learned base, then create a distinct typed head."""
    import torch
    from marshawn_frozen_semantic_head import FrozenActorGrid, tensor_digest
    from marshawn_semantic_residual_head import FrozenResidualActorGrid
    validate_spec(spec)
    if (base.get('iteration') != 2064 or base.get('model_digest') != PIPELINE_SHA or
            base.get('label_qualification_sha256') != LABEL_SHA or
            base.get('LC_capture_receipt_sha256') != CAPTURE_SHA or
            base.get('source_kind') != 'DIRECT_FROZEN_LC_NATIVE_VISIBILITY' or
            base.get('world_to_box_is_identity') is not True):
        raise ValueError('Learned coarse2064 source provenance differs')
    logits = base.get('head', {}).get('logits')
    if not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32 or logits.shape != (1, 1, 128, 128, 128):
        raise ValueError('Base checkpoint must contain explicit FP32 coarse128 logits')
    coarse = FrozenActorGrid(bounds, torch.eye(4), resolution=128, initial_probability=.01)
    coarse.load_state_dict(base['head'], strict=True)
    if (coarse.logits.dtype != torch.float32 or not torch.isfinite(coarse.logits).all() or
            tensor_digest(coarse) != base['head_digest'] or
            not torch.equal(coarse.bounds, torch.tensor(bounds, dtype=torch.float32)) or
            not torch.equal(coarse.world_to_box, torch.eye(4))):
        raise ValueError('Learned coarse state/frame/digest differs')
    return coarse if spec['arm'] == 'coarse128' else FrozenResidualActorGrid(coarse, resolution=256)


def validate_resume(parent, spec, bounds):
    expected = checkpoint_metadata(spec)
    if any(parent.get(k) != value for k, value in expected.items()):
        raise ValueError('Refinement continuation head/schema/source bindings differ')
    groups = parent.get('optimizer', {}).get('param_groups', [])
    if len(groups) != 1 or groups[0].get('lr') != .03:
        raise ValueError('Actual refinement Adam LR differs')
    if parent.get('iteration') != 32 or len(parent.get('camera_history', [])) != 32:
        raise ValueError('Technical continuation must restore actual stage32')
    # strict load on the correctly constructed typed head checks every state key/shape.
