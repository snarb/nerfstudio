"""One selected photo replaces only corroborated frequency bands missing in LookCloser."""
import cv2
import numpy as np
from transfer_luster_hybrid import gaussian

def merge_missing(base,texture,labels,config):
    result=base.copy();applied=np.zeros(labels.shape,np.float32)
    for cid in np.unique(labels[labels>=0]):
        mask=(labels==cid).astype(np.float32);distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
        boundary=np.minimum(distance/3,1)*mask
        def low(value,sigma):return gaussian(value*mask[...,None],sigma)/np.maximum(gaussian(mask,sigma)[...,None],1e-8)
        previous_photo=texture;previous_base=base;delta=np.zeros_like(base);maximum=np.zeros(labels.shape,np.float32)
        for sigma in [1.,2.,4.,8.]:
            if sigma>config['sigma']:break
            lp=low(texture,sigma);lb=low(base,sigma);pband=previous_photo-lp;bband=previous_base-lb
            window=max(3.,sigma);mass=gaussian(mask,window)
            def moment(value):return gaussian(value*mask,window)/np.maximum(mass,1e-8)
            pe=np.sqrt(np.maximum(moment(np.mean(pband*pband,axis=-1)),0));be=np.sqrt(np.maximum(moment(np.mean(bband*bband,axis=-1)),0))
            correlation=moment(np.mean(pband*bband,axis=-1))/np.maximum(pe*be,1e-10)
            gain=np.clip((pe-config['energy_ratio']*be)/np.maximum(pe*.5,1e-8),0,1)
            gain*=np.clip((pe-config['energy_floor'])/max(config['energy_floor'],1e-8),0,1)
            gain*=np.where(be<config['model_energy_floor'],1.,np.clip((correlation-config['phase_min'])/.3,0,1))
            gain*=boundary*config['gain'];delta+=gain[...,None]*(pband-bband);maximum=np.maximum(maximum,gain)
            previous_photo=lp;previous_base=lb
        selected=(labels==cid)&(maximum>0)
        result[selected]=np.clip(base[selected]+delta[selected],0,1);applied[selected]=maximum[selected]
    return result,applied
