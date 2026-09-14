"""Absolute calibrated display-color evidence alongside chromaticity.

Geometry matches are delegated to the frozen depth/chroma witness implementation.
No per-image gain or normalization is fitted to make a witness match.
"""
import numpy as np
from diagnose_forearm_color_witnesses import witness_errors
from study_confidence_depth_prior import project_integer


def patch_rgb(image, xy):
    xy=np.asarray(xy,dtype=int);rgb=np.zeros((len(xy),3),np.float64)
    for dy in range(-2,3):
        for dx in range(-2,3):
            rgb+=image[np.clip(xy[:,1]+dy,0,image.shape[0]-1),np.clip(xy[:,0]+dx,0,image.shape[1]-1)]
    return rgb/(25*255)


def color_errors(points, reference, rows, depths, images):
    chroma=witness_errors(points,reference,rows,depths,images)
    uv,_=project_integer(reference,points)
    query=patch_rgb(images[reference['physical_camera']],np.rint(uv).astype(int))
    rgb=np.full(chroma.shape,np.nan,np.float32)
    for i,row in enumerate(rows):
        ids=np.flatnonzero(np.isfinite(chroma[i]))
        if not len(ids):continue
        uv,_=project_integer(row,points[ids])
        other=patch_rgb(images[row['physical_camera']],np.rint(uv).astype(int))
        rgb[i,ids]=np.abs(other-query[ids]).mean(1)
    return chroma,rgb


def qualified_errors(points,reference,rows,depths,images,limit):
    chroma,rgb=color_errors(points,reference,rows,depths,images)
    return np.where((chroma<=.04)&(rgb<=limit),chroma,np.nan)
