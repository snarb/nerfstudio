"""Conservative color/region evidence against a farther, wrong-object depth.

This may disqualify a free-space witness; it does not measure surface depth.
Skin-to-skin, boundary, unavailable and weak-view evidence remain undecided.
"""
import numpy as np


def rejects_far_layer(query_warm, near_inside, near_warm, far_blue, available):
    query_warm=np.asarray(query_warm,bool)
    arrays=[np.asarray(a,bool) for a in [near_inside,near_warm,far_blue,available]]
    if any(a.shape!=(len(query_warm),4) for a in arrays):
        raise ValueError('Expected exactly four distinct physical witness cameras')
    inside,warm,blue,known=arrays
    return query_warm & inside.all(1) & warm.all(1) & known.all(1) & (blue.sum(1)>=3)
