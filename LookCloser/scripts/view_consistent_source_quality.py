"""Opt-in incidence ablations; visibility stays a separate mandatory gate."""
import numpy as np


def quality(incidence,distance,mode):
    incidence=np.asarray(incidence);distance=np.asarray(distance)
    if incidence.shape!=distance.shape or not np.isfinite(incidence).all() or not np.isfinite(distance).all() or (distance<=0).any():
        raise ValueError('Invalid source geometry')
    if mode=='incidence2':return np.abs(incidence)**2/np.maximum(distance,.01)**2
    if mode=='angular_only':return np.ones_like(incidence)
    raise ValueError('Unknown source quality mode')
