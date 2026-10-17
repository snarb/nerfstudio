"""Edge-aware scalar source costs, guided only by frozen target RGB and depth."""
import numpy as np

def aggregate(score, support, rgb, depth, spatial_sigma=3., color_sigma=.06, depth_sigma=.005, radius=3):
    """Normalize valid neighbors; preserve sharp appearance/surface boundaries.

    Output RGB and geometry never enter this function as mutable arguments.
    Invalid centers receive infinity. No wraparound or invented border taps.
    """
    assert spatial_sigma>0 and color_sigma>0 and depth_sigma>0 and radius>=0
    valid=support & np.isfinite(score) & np.isfinite(depth) & (depth>0) & np.isfinite(rgb).all(-1)
    numerator=np.zeros(score.shape,np.float32);mass=np.zeros_like(numerator)
    h,w=score.shape
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            center=np.s_[max(0,-dy):min(h,h-dy),max(0,-dx):min(w,w-dx)]
            neighbor=np.s_[max(0,dy):min(h,h+dy),max(0,dx):min(w,w+dx)]
            both=valid[center]&valid[neighbor]
            color_distance=np.mean((rgb[center]-rgb[neighbor])**2,axis=-1)/(color_sigma**2)
            zc=np.where(valid[center],depth[center],1.);zn=np.where(valid[neighbor],depth[neighbor],1.)
            depth_distance=((zc-zn)/np.maximum(.5*(zc+zn),1e-8)/depth_sigma)**2
            exponent=.5*((dx*dx+dy*dy)/(spatial_sigma**2)+color_distance+depth_distance)
            weight=np.where(both,np.exp(-np.minimum(np.nan_to_num(exponent,nan=1e6),80)),0).astype(np.float32)
            numerator[center]+=weight*np.where(both,score[neighbor],0)
            mass[center]+=weight
    return np.where(valid,numerator/np.maximum(mass,1e-8),np.inf).astype(np.float32)
