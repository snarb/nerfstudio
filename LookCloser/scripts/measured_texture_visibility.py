"""Opt-in negative texture evidence from stable measured farther-depth layers.

Missing stereo is unknown, not rejection. This changes RGB source eligibility,
never geometry. It is intentionally not the multi-camera geometry deletion gate.
"""
import numpy as np
from carve_patchmatch_mesh_free_space import free_space_evidence


def reject_farther_source(depth, uv, z, eligible):
    uv,z,eligible=map(np.asarray,(uv,z,eligible))
    if uv.shape!=(len(z),2) or eligible.shape!=z.shape or eligible.dtype!=bool:
        raise ValueError('Expected native UV, depth and boolean source eligibility')
    result=np.zeros(z.shape,bool)
    selected=np.flatnonzero(eligible)
    if len(selected):
        free,_=free_space_evidence(depth,uv[selected,0],uv[selected,1],z[selected],
                                   minimum_gap=.005,near_gap=.0015)
        result[selected]=free
    return result
