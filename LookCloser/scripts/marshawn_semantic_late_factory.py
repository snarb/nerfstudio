#!/usr/bin/env python3
"""Configuration factory for a separately reviewed semantic30376 fork; no training."""
from pathlib import Path

RECIPE=dict(objective='actor',sampling='body80',width=64,radius=0,coarse_step=.00625,
            distortion_final=.01,seed=42,frequency_threshold=.95,correct_sh_directions=True,
            density_activation='trunc_exp',frequency_projection_mode='legacy',lr_decay_steps=200000,
            followup_variant='actor')
STATE_KEYS={'pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture'}

def validate_anchor(parent,anchor,coefficient):
    if type(coefficient)is not float or coefficient not in (.01,1.):raise ValueError('Only reviewed .01/1 semantic pair')
    if anchor.get('anchor_status')!='EVALUATED_INTERMEDIATE'or anchor.get('step')!=30376 or anchor.get('producer_continues')is not True:
        raise ValueError('Evaluated intermediate30376 required, not a completed parent')
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
    if not 0<=30376-capture.get('step',-100)<16:raise ValueError('Stale semantic capture')
    if anchor.get('source_eval',{}).get('step')!=30376:raise ValueError('Anchor lacks its exact evaluated gate')
    return parent

def build_config(parent,anchor,checkpoint,output,coefficient):
    import train_marshawn_semantic_admission as base
    validate_anchor(parent,anchor,coefficient);base.validate_spec(parent)
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
    cfg.pipeline.model.distortion_loss_mult=coefficient
    return cfg
