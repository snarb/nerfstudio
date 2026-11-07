"""Strict configuration-only factory for reviewed focused15188 geometry forks."""
from pathlib import Path
from marshawn_focused_clothing_geometry import ARMS

PARENT_SHA256 = 'a29f83e34be9406729d1a05c3dc40d52adceb934028b68fd147be29cfe4eb837'
RECIPE = dict(objective='actor', sampling='body80', width=64, radius=0, coarse_step=.00625,
              distortion_final=.01, seed=42, frequency_threshold=.95, correct_sh_directions=True,
              density_activation='trunc_exp', frequency_projection_mode='legacy', lr_decay_steps=200000,
              followup_variant='actor')
FOCUS = dict(budget=2048, core_count=1229, band_count=410, uniform_count=409, private_seed=4201024)
STATE_KEYS = {'pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture'}


def validate_anchor(parent, anchor, arm):
    if arm not in ARMS:
        raise ValueError('Only declared four-arm geometry comparison')
    if anchor.get('step') != 15188 or anchor.get('checkpoint_sha256') != PARENT_SHA256:
        raise ValueError('Exact immutable focused15188 source required')
    if anchor.get('producer_continues') is not False or anchor.get('anchor_status') != 'COMPLETED_FOCUSED15188':
        raise ValueError('Live or unqualified parent prohibited')
    if set(anchor.get('fullstate_hashes', {})) != STATE_KEYS:
        raise ValueError('All seven source state hashes required')
    for key, value in RECIPE.items():
        if type(parent.get(key)) is not type(value) or parent[key] != value:
            raise ValueError('Source recipe differs: ' + key)
    if parent.get('frequency_focus') != FOCUS:
        raise ValueError('Inherited focused frequency sampler must remain exact')
    admission = parent.get('semantic_admission', {})
    if admission.get('identity_control') is not False or not admission.get('manifest_sha256'):
        raise ValueError('Actual frozen semantic admission required')
    if anchor.get('FR') != 1. or anchor.get('frequency_switch_applied') is not False:
        raise ValueError('15188 requires pre-switch FR1')
    if not anchor.get('focus_state_sha256') or not anchor.get('projected84_source_bound'):
        raise ValueError('Private focus state and qualified source bindings required')
    if not all(anchor.get('semantic_grid_checks', {}).get(k) is True for k in ('core_capture','active_capture','active_pipeline','EMA_pipeline','learned_core_pipeline')):
        raise ValueError('Actual semantic grid state proof required')
    return parent


def build_config(parent, anchor, checkpoint, output, arm):
    import train_marshawn_semantic_admission as base
    validate_anchor(parent, anchor, arm)
    base.validate_spec(parent)
    if Path(output).resolve() == Path(parent['output']).resolve():
        raise ValueError('Fork cannot overwrite producer')
    cfg = base.build_config(parent, Path(checkpoint))
    cfg.output_dir = Path(output)
    if cfg.pipeline.model.__class__.__name__ != 'SemanticAdmissionModelConfig' or cfg.pipeline.model._target.__name__ != 'SemanticAdmissionModel':
        raise ValueError('Generic model reload loses semantic selector')
    if cfg.pipeline.model.semantic_identity_control or cfg.pipeline.model.semantic_manifest_sha256 != parent['semantic_admission']['manifest_sha256']:
        raise ValueError('Semantic source differs')
    if not cfg.load_optimizers or not cfg.load_scheduler or cfg.checkpoint_load_mode != 'resume' or cfg.resume_fields_lr_override is not None or cfg.resume_reset_occupancy_grid or cfg.resume_reset_frequency_grid:
        raise ValueError('Exact state restoration without LR/grid reset required')
    if cfg.pipeline.feature_reweighting_switch_step != 75941 or cfg.pipeline.feature_reweighting_after_switch != .3:
        raise ValueError('Inherited future FR schedule must remain unchanged')
    if cfg.pipeline.grid_update_interval != 1024 or cfg.pipeline.grid_update_batch_size != 2048:
        raise ValueError('Focused frequency update policy changed')
    if cfg.pipeline.model.distortion_loss_mult != .01:
        raise ValueError('Geometry arms cannot change distortion')
    return cfg
