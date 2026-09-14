"""A free-space witness must distinguish observed depth from the nearer prior.

Flat same-color alternatives are ambiguous, not affirmative empty-space evidence.
This is an opt-in candidate comparison, never an evaluation mask or RGB refiner.
"""
import numpy as np
from forearm_rgb_witnesses import color_errors,patch_rgb
from study_confidence_depth_prior import project_integer


def carving_decision(votes):
    # Abstention needs positive evidence of ambiguity, not merely an unavailable
    # comparison. If fewer than three alternative projections can be inspected,
    # preserve the existing RGB-qualified rule instead of silently weakening it.
    old=votes['rgb_qualified']>=3
    return old&((votes['decisive']>=3)|(votes['comparable']<3))


def comparison_votes(observed,proposed,reference,rows,depths,images,margin=.01):
    if margin<=0:raise ValueError('Positive photometric discrimination margin required')
    chroma,rgb=color_errors(observed,reference,rows,depths,images)
    old=(chroma<=.04)&(rgb<=.12)
    uv,_=project_integer(reference,observed)
    query=patch_rgb(images[reference['physical_camera']],np.rint(uv).astype(int))
    decisive=np.zeros(old.shape,bool);available=np.zeros(old.shape,bool)
    for i,(row,depth) in enumerate(zip(rows,depths)):
        ids=np.flatnonzero(old[i])
        if not len(ids):continue
        uv,z=project_integer(row,proposed[ids]);xy=np.rint(uv).astype(int)
        inside=(z>0)&(xy[:,0]>=3)&(xy[:,0]<1917)&(xy[:,1]>=3)&(xy[:,1]<1077)
        local=np.flatnonzero(inside);qx,qy=xy[local].T;measured=depth[qy,qx]
        # Do not score a prior point hidden behind an independently observed surface.
        usable=~((measured>0)&np.isfinite(measured)&(measured<z[local]-.001))
        local=local[usable];selected=ids[local]
        alt=patch_rgb(images[row['physical_camera']],xy[local])
        error=np.abs(alt-query[selected]).mean(1)
        available[i,selected]=True;decisive[i,selected]=error>rgb[i,selected]+margin
    return dict(geometric=np.isfinite(chroma).sum(0),rgb_qualified=old.sum(0),
                comparable=available.sum(0),decisive=decisive.sum(0))
