"""Configuration-only matched distortion.3 continuation from exact quarter+.1 raw126316."""
from pathlib import Path

SOURCE_STEP=126316
SOURCE_SHA='340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641'
ARMS={'control01':.1,'distortion03':.3}
STATE_KEYS={'pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture'}
RECIPE=dict(objective='actor',sampling='body80',width=64,radius=0,coarse_step=.0015625,distortion_final=.1,seed=42,frequency_threshold=.95,correct_sh_directions=True,density_activation='trunc_exp',frequency_projection_mode='legacy',lr_decay_steps=200000,followup_variant='actor')


def validate_anchor(parent,anchor):
    if anchor['step']!=SOURCE_STEP or anchor['checkpoint_sha256']!=SOURCE_SHA or set(anchor['fullstate_hashes'])!=STATE_KEYS:raise ValueError('Exact seven-state raw126316 source required')
    for key,value in RECIPE.items():
        if type(parent.get(key))is not type(value)or parent[key]!=value:raise ValueError('Source scientific recipe differs: '+key)
    if parent['semantic_admission']['identity_control']or anchor['semantic_capture_metadata']['radius']!=0:raise ValueError('Original mapped semantic R0 source required')
    if anchor['semantic_capture_metadata']['semantic_receipt']['manifest_sha256']!=parent['semantic_admission']['manifest_sha256']:raise ValueError('Source fixed density admission differs')
    if not all(anchor['semantic_grid_checks'].values()):raise ValueError('Source core/active/EMA/capture checks failed')
    if anchor['source_eval']['step']!=SOURCE_STEP:raise ValueError('Exact source FHD evaluation required')


def build_config(parent,anchor,checkpoint,output):
    import train_marshawn_semantic_admission as base
    validate_anchor(parent,anchor);base.validate_spec(parent);cfg=base.build_config(parent,Path(checkpoint));cfg.output_dir=Path(output)
    if cfg.checkpoint_load_mode!='resume'or not cfg.load_optimizers or not cfg.load_scheduler or cfg.resume_fields_lr_override is not None or cfg.resume_reset_occupancy_grid or cfg.resume_reset_frequency_grid:raise ValueError('Full original optimizer/scheduler/grid resume required')
    if cfg.pipeline.model.__class__.__name__!='SemanticAdmissionModelConfig'or cfg.pipeline.model.semantic_identity_control:raise ValueError('Original mapped field admission must remain installed')
    if cfg.pipeline.model.adaptive_coarse_step_size!=.0015625 or cfg.pipeline.model.max_steps_per_ray!=1024 or cfg.pipeline.model.corrected_arm_allocator:raise ValueError('Original quarter native sampler required')
    if cfg.pipeline.feature_reweighting_after_switch!=.3 or cfg.pipeline.feature_reweighting_switch_step!=75941:raise ValueError('Original FR.3 schedule required')
    # The cold archived config displays .01; its original absolute-step loop had
    # already reached actual .1. All four branches preserve that source runtime.
    cfg.pipeline.model.distortion_loss_mult=.1
    return cfg


def build_effective_config(parent,anchor,checkpoint,output,arm):
    if arm not in ARMS:raise ValueError('Exact .1/.3 distortion arm required')
    cfg=build_config(parent,anchor,checkpoint,output)
    cfg.pipeline.model.distortion_loss_mult=ARMS[arm]
    return cfg
