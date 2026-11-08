"""Current anatomy needs target-human role evidence before becoming new FG.

This pure decision layer receives immutable preapplication CURRENT target
measurements and source-qualified SAME-FAMILY prior-role support. It emits no
foreground, background, occupancy or RGB confidence pixels by itself.
"""
def decide_current_role(current_A_pixels,current_RAW_pixels,near_fraction,role_supported,role_residual,bootstrap=False):
    current_cue=current_A_pixels>0 or current_RAW_pixels>0 or near_fraction>=.8
    prior_role=role_supported>=128 and role_supported/max(role_residual,1)>=.5
    return dict(current_target_cue=bool(current_cue),
                same_family_prior_role=bool(prior_role),
                finite_current_photo_bootstrap=bool(bootstrap),
                accepted=bool(current_cue or prior_role or bootstrap))
