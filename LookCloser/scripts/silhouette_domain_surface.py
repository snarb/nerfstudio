"""Discard implicit-surface faces caused by annotation/FOV availability edges."""
import numpy as np
from scipy.ndimage import map_coordinates


def availability_bits(uv, depths, domains):
    if len(domains)>16: raise ValueError('uint16 camera inventory exceeded')
    bits=np.zeros(uv.shape[1],np.uint16)
    for i,(xy,z,domain) in enumerate(zip(uv,depths,domains)):
        h,w=domain.shape
        valid=np.isfinite(xy).all(1)&np.isfinite(z)&(z>0)&(xy[:,0]>=0)&(xy[:,0]<=w-1)&(xy[:,1]>=0)&(xy[:,1]<=h-1)
        ids=np.flatnonzero(valid)
        known=map_coordinates(domain.astype(np.float32),xy[ids].T[::-1],order=1,mode='constant',cval=0)>=.99999
        bits[ids[known]] |= np.uint16(1<<i)
    return bits


def stable_domain_faces(vertices, triangles, lower, spacing, bits, minimum_views=3):
    """Only interpolate cells whose eight corners have identical known cameras.

    Insufficient observations mean unknown occupancy, not empty space. A camera
    becoming available at a frame edge must not create an anatomical cap either.
    """
    center=(np.asarray(vertices)[triangles].mean(1)-np.asarray(lower))/spacing
    ijk=np.floor(center).astype(int)
    valid=((ijk>=0)&(ijk<np.array(bits.shape)-1)).all(1)
    keep=np.zeros(len(triangles),bool)
    ids=np.flatnonzero(valid);p=ijk[ids]
    corners=np.array([bits[tuple((p+offset).T)] for offset in np.ndindex(2,2,2)])
    same=(corners==corners[:1]).all(0)
    # Popcount without requiring a recent numpy version.
    count=np.array([int(x).bit_count() for x in corners[0]])
    keep[ids]=same&(count>=minimum_views)
    return keep
