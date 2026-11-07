#!/usr/bin/env python3
"""Configuration factory for a separately reviewed quarter75940 FR fork; no training."""
from pathlib import Path

RECIPE=dict(objective='actor',sampling='body80',width=64,radius=0,coarse_step=.0015625,
            distortion_final=.01,seed=42,frequency_threshold=.95,correct_sh_directions=True,
            density_activation='trunc_exp',frequency_projection_mode='legacy',lr_decay_steps=200000,
            followup_variant='actor')
STATE_KEYS={'pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture'}

def validate_anchor(parent,anchor,after_switch):
    if type(after_switch)is not float or after_switch not in (.3,1.):raise ValueError('Only originalFR.3 or holdFR1 pair')
    if anchor.get('anchor_status')!='EVALUATED_INTERMEDIATE'or anchor.get('step')!=75940 or anchor.get('producer_continues')is not True:
        raise ValueError('Evaluated intermediate75940 required, not a completed parent')
    if set(anchor.get('fullstate_hashes',{}))!=STATE_KEYS:raise ValueError('Full serialized-state hashes required')
    for key,value in RECIPE.items():
        actual=parent.get(key)
        if type(actual)is not type(value)or actual!=value:raise ValueError('Wrong semantic source recipe: '+key)
    admission=parent.get('semantic_admission',{})
    if admission.get('identity_control')is not False or not admission.get('manifest_sha256'):raise ValueError('Real sealed semantic admission required')
    if not all(anchor.get('semantic_grid_checks',{}).get(k)is True for k in ('core_capture','active_capture','active_pipeline','EMA_pipeline','learned_core_pipeline')):
        raise ValueError('Actual captured semantic core/EMA/active proof required')
    capture=anchor.get('semantic_capture_metadata',{})
    if capture.get('radius')!=0 or capture.get('binary_warmup')is not False or capture.get('semantic_receipt',{}).get('manifest_sha256')!=admission['manifest_sha256']:
        raise ValueError('Semantic source policy differs')
    if not 0<=75940-capture.get('step',-100)<16:raise ValueError('Stale semantic capture')
    if anchor.get('source_eval',{}).get('step')!=75940:raise ValueError('Anchor lacks its exact evaluated gate')
    return parent

def build_config(parent,anchor,checkpoint,output,after_switch):
    import train_marshawn_semantic_admission as base
    validate_anchor(parent,anchor,after_switch);base.validate_spec(parent)
    if Path(output).resolve()==Path(parent['output']).resolve():raise ValueError('Fork cannot overwrite producer output')
    cfg=base.build_config(parent,Path(checkpoint));cfg.output_dir=Path(output)
    if cfg.pipeline.model.__class__.__name__!='SemanticAdmissionModelConfig'or cfg.pipeline.model._target.__name__!='SemanticAdmissionModel':
        raise ValueError('Generic LookCloser reload would omit semantic field admission')
    if cfg.pipeline.model.semantic_identity_control or cfg.pipeline.model.semantic_manifest_sha256!=parent['semantic_admission']['manifest_sha256']:
        raise ValueError('Semantic selector source differs')
    if not cfg.load_optimizers or not cfg.load_scheduler or cfg.checkpoint_load_mode!='resume' or cfg.resume_fields_lr_override is not None or cfg.resume_reset_occupancy_grid or cfg.resume_reset_frequency_grid:
        raise ValueError('Full-state semantic resume required')
    if cfg.pipeline.feature_reweighting_switch_step!=75941 or cfg.pipeline.feature_reweighting_after_switch!=.3:
        raise ValueError('Original FR schedule required')
    if cfg.pipeline.model.max_steps_per_ray!=1024 or cfg.pipeline.model.corrected_arm_allocator:
        raise ValueError('Original quarter legacy1024 cap required')
    cfg.pipeline.feature_reweighting_after_switch=after_switch
    return cfg


def verify_FR_runtime(pipe, step, after_switch):
    expected_applied = step >= 75941
    expected = after_switch if expected_applied else 1.
    model = pipe.model
    if bool(pipe._feature_reweighting_switch_applied) != expected_applied:
        raise ValueError('Runtime FR switch flag disagrees with absolute checkpoint step')
    if model.config.feature_reweighting_strength != expected or model.field.feature_reweighting_strength != expected:
        raise ValueError('Runtime FR model/field strength differs')
    lut = model.field._feature_weight_lut
    cached = model.field._feature_weight_lut_strength
    if lut.numel() and cached != expected:
        raise ValueError('Stale nonpersistent feature LUT strength')
    return dict(step=step, switch_applied=expected_applied, model_strength=expected,
                field_strength=expected, LUT_values=lut.numel(), LUT_strength=cached,
                cache_policy='empty after reload is allowed; next query must rebuild for actual strength')
