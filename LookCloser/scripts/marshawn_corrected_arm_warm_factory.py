"""One-factor corrected ARM continuation; cap1024 remains fixed in both arms."""
from marshawn_distortion_warm_factory import SOURCE_STEP, SOURCE_SHA, STATE_KEYS, validate_anchor, build_config

ARMS = {'legacy1024': False, 'corrected1024': True}


def build_effective_config(parent, anchor, checkpoint, output, arm):
    if arm not in ARMS:
        raise ValueError('Exact legacy1024/corrected1024 arm required')
    cfg = build_config(parent, anchor, checkpoint, output)
    cfg.pipeline.model.corrected_arm_allocator = ARMS[arm]
    if cfg.pipeline.model.max_steps_per_ray != 1024:
        raise ValueError('Cap removal is a separate scientific factor')
    return cfg
