"""Attenuate detail that displaces coherent LookCloser appearance contours.

Prediction-only diagnostic hypothesis. No target photographs, evaluation masks,
coordinate shifts, geometry changes or mixing of texture sources.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import geometry, image, log, read, sha, write

cv2.setNumThreads(2)


def blur(a, sigma):
    return cv2.GaussianBlur(a, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101)


def guard(base, hybrid, sigma=2., displacement=1., coherence_threshold=.8):
    assert sigma > 0 and displacement > 0
    # RGB residual catches chromatic contours as well as luminance edges.
    b=blur(np.asarray(base,np.float32),sigma)
    h=blur(np.asarray(hybrid,np.float32),sigma)
    dx=cv2.Scharr(b,cv2.CV_32F,1,0)/32
    dy=cv2.Scharr(b,cv2.CV_32F,0,1)/32
    xx=blur(np.mean(dx*dx,axis=-1),4.)
    yy=blur(np.mean(dy*dy,axis=-1),4.)
    xy=blur(np.mean(dx*dy,axis=-1),4.)
    coherence=np.sqrt((xx-yy)**2+4*xy**2)/(xx+yy+1e-10)
    gradient=np.sqrt(np.mean(dx*dx+dy*dy,axis=-1))
    # First-order displacement estimate is used only to reject transfer,
    # never to move sampling coordinates or to reconstruct a surface.
    shift=np.sqrt(np.mean((h-b)**2,axis=-1))/(gradient+1e-4)
    edge=(coherence>=coherence_threshold)&(gradient*sigma>=.015)
    risk=np.where(edge,np.clip(shift/displacement-1,0,1),0).astype(np.float32)
    risk=cv2.dilate(risk,np.ones((5,5),np.uint8))
    # Compact smoothing: regions outside risk support remain exactly unchanged.
    risk=cv2.GaussianBlur(risk,(13,13),2.)
    return (1-risk).clip(0,1),shift,coherence


def combine(root,parent,name,settings):
    old=root/'candidates'/parent
    out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'),'parent':parent,'contour_guard':settings}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    for cid in [7,12,95,150]:
        receipt_path=out/f'cam{cid:03d}_receipt.json'
        if receipt_path.exists():
            assert read(receipt_path)['code_sha256']==sha(Path(__file__))
            continue
        base=np.asarray(geometry(root,cid)['rgb'])
        pred=np.load(old/f'cam{cid:03d}.npy',mmap_mode='r')
        attenuation,_,_=guard(base,pred,**settings)
        result=base+attenuation[...,None]*(pred-base)
        # Avoid even a floating-point round trip where transfer is untouched.
        result[attenuation==1]=pred[attenuation==1]
        confidence=np.load(old/f'cam{cid:03d}_confidence.npy')*attenuation
        labels=np.load(old/f'cam{cid:03d}_sources.npy').copy()
        labels[confidence==0]=-1
        assert np.isfinite(result).all()
        assert np.array_equal(result[confidence==0],base[confidence==0])
        for suffix,a in [('',result),('_confidence',confidence),('_sources',labels),
                         ('_contour_weight',attenuation)]:
            np.save(out/f'cam{cid:03d}{suffix}.npy',a)
        image(out/f'cam{cid:03d}.png',result)
        image(out/f'cam{cid:03d}_transfer.png',np.repeat(confidence[...,None],3,-1))
        inputs={str(old/f'cam{cid:03d}{s}'):sha(old/f'cam{cid:03d}{s}')
                for s in ['.npy','_receipt.json','_sources.npy','_confidence.npy']}
        write(receipt_path,{**read(old/f'cam{cid:03d}_receipt.json'),
            'config':config,'code_sha256':sha(Path(__file__)),
            'additional_input_hashes':inputs,'uses_target_rgb':False,
            'changed_fraction':float(np.any(result!=base,axis=-1).mean()),
            'attenuated_fraction':float(((attenuation<1)&(confidence>0)).mean())})
        log(root,'contour_guard_complete',camera=cid,candidate=name)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent',required=True);p.add_argument('--name',required=True)
    p.add_argument('--sigma',type=float,default=2.)
    p.add_argument('--displacement',type=float,default=1.)
    a=p.parse_args()
    combine(a.root,a.parent,a.name,dict(sigma=a.sigma,displacement=a.displacement))
