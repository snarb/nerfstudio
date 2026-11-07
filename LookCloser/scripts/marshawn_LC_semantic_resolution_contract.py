"""Explicit contracts for an independent FOUR-UNKNOWN 128/256 node ablation.

No field, optimizer or CUDA initialization occurs on import. The existing
FrozenActorGrid remains the sole class-grid implementation.
"""
from numbers import Integral

from prepare_marshawn_frozen_semantic_cpu import sha

FAMILY = 'DIRECT_LC_FOUR_UNKNOWN_INDEPENDENT_NODE_RESOLUTION_V1'
IDENTITY = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
UNKNOWN = {
    ('IA', 40470, (1013, 562)), ('IA', 40622, (1019, 567)),
    ('LA', 54411, (875, 564)), ('LA', 54594, (871, 558)),
}


def node_resolution(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (128, 256):
        raise ValueError('Only explicit integer128/256 node resolutions are prepared')
    return int(value)


def validate_spec(spec):
    resolution = node_resolution(spec['grid_resolution'])
    if (spec.get('resolution_experiment_family') != FAMILY or spec['seed'] != 42 or
            spec['learning_rate'] != .03 or spec['initial_probability'] != .01 or
            spec['world_to_box'] != IDENTITY or spec['qualified_UNKNOWN_count'] != 4 or
            spec['native_model_updates'] != 0 or not spec['no_NGP_head_transfer']):
        raise ValueError('Independent resolution ablation must retain p.01/seed42/LR.03/floor/FOUR-UNKNOWN/source policy')
    if not spec.get('packet_history_reference'):
        raise ValueError('The same fixed2064 camera-packet reference is required in every phase')
    return resolution


def validate_qualification(labels):
    rows = labels['unknown_queries']
    actual = {(r['camera'], r['global_query_index'], tuple(r['pixel_xy'])) for r in rows}
    if len(rows) != 4 or actual != UNKNOWN or any(r['label_weight'] != 0 for r in rows):
        raise ValueError('Exactly the two reviewed IA and two reviewed LA UNKNOWN queries are required')
    # Older IA qualification schema omits semantic_target; the actual immutable
    # cache target is checked to equal1 at the worker's override site.
    if any(r.get('semantic_target', 1) != 1 for r in rows):
        raise ValueError('UNKNOWN preserves every positive target; no new negative label')


def checkpoint_metadata(spec):
    """Resolution and data/packet bindings saved in every NEW worker checkpoint."""
    resolution = validate_spec(spec)
    return dict(grid_resolution=resolution, initial_seed=42, learning_rate=.03,
                qualified_UNKNOWN_count=4, resolution_experiment_family=FAMILY,
                source_manifest_sha256=sha(spec['manifest']),
                capture_adapter_sha256=sha(spec['capture_receipt']),
                packet_history_reference_sha256=sha(spec['packet_history_reference']))


def validate_saved_state(state, resolution, bounds):
    import torch
    resolution = node_resolution(resolution)
    logits = state.get('logits')
    if (not isinstance(logits, torch.Tensor) or logits.shape != (1, 1, resolution, resolution, resolution) or
            logits.dtype != torch.float32 or not torch.isfinite(logits).all()):
        raise ValueError('Saved node tensor must be finite FP32 with the declared XYZ resolution')
    frame = state.get('world_to_box')
    saved_bounds = state.get('bounds')
    if not isinstance(frame, torch.Tensor) or not torch.equal(frame, torch.tensor(IDENTITY, dtype=torch.float32)):
        raise ValueError('Saved floor transform must be identity')
    if not isinstance(saved_bounds, torch.Tensor) or not torch.equal(saved_bounds, torch.tensor(bounds, dtype=torch.float32)):
        raise ValueError('Saved bounds must exactly equal the accepted source bounds')


def validate_resume(parent, spec, bounds):
    expected = checkpoint_metadata(spec)
    if any(parent.get(k) != value for k, value in expected.items()):
        raise ValueError('Resolution, FOUR-UNKNOWN, optimizer or immutable data/packet resume provenance differs')
    groups = parent.get('optimizer', {}).get('param_groups', [])
    if len(groups) != 1 or groups[0].get('lr') != .03:
        raise ValueError('Resume must retain the actual Adam learning rate.03')
    validate_saved_state(parent['head'], expected['grid_resolution'], bounds)


def validate_matched_baseline(spec, baseline):
    """Only node resolution may differ between the fresh2064 scientific arms."""
    validate_spec(spec)
    fixed = ['seed', 'learning_rate', 'initial_probability', 'world_to_box', 'qualified_UNKNOWN_count',
             'native_model_updates', 'no_NGP_head_transfer', 'manifest', 'capture_receipt',
             'label_qualification', 'capacity_policy', 'torch_version', 'cuda_version']
    if baseline['grid_resolution'] != 128 or any(spec[k] != baseline[k] for k in fixed):
        raise ValueError('Comparator must be matched FOUR-UNKNOWN128, not the original TWO-UNKNOWN arm')
    if (baseline['updates'] != 2064 or baseline.get('resume') or
            baseline.get('camera_history_reference') != spec['packet_history_reference']):
        raise ValueError('Comparator must be fresh2064 with the identical camera-packet trajectory')


def physical_node_resolution(trained, completion, spec):
    """Constructor guard for the proposed renderer patch; no render integration.

    Historical128 worker checkpoints omit grid_resolution; accept that only
    when both the explicit render spec and completion say128 and shape agrees.
    New256 checkpoints must carry the new worker's full resolution identity.
    Existing source/label/spec/digest checks in the physical worker remain.
    """
    resolution = node_resolution(spec['expected_grid_resolution'])
    if completion.get('grid_resolution') != resolution or trained.get('grid_resolution', 128) != resolution:
        raise ValueError('Physical constructor resolution must match spec, completion and saved head')
    if resolution == 256 and (trained.get('resolution_experiment_family') != FAMILY or trained.get('qualified_UNKNOWN_count') != 4):
        raise ValueError('256 renderer requires the explicit new FOUR-UNKNOWN worker checkpoint')
    validate_saved_state(trained['head'], resolution, spec['expected_head_bounds'])
    return resolution
