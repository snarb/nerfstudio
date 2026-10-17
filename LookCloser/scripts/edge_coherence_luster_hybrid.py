"""Frozen-render edge structure, with no photograph, mask or correspondence fit."""
import cv2
import numpy as np
from transfer_luster_hybrid import gaussian


def feature(rgb):
    gray=np.asarray(rgb,np.float32)@np.array([.2126,.7152,.0722],np.float32)
    low=gaussian(gray,2.)
    gx=cv2.Sobel(low,cv2.CV_32F,1,0,ksize=3,scale=1/8)
    gy=cv2.Sobel(low,cv2.CV_32F,0,1,ksize=3,scale=1/8)
    xx=gaussian(gx*gx,4.);yy=gaussian(gy*gy,4.);xy=gaussian(gx*gy,4.)
    trace=np.maximum(xx+yy,0)
    gap=np.sqrt(np.maximum((xx-yy)**2+4*xy*xy,0))
    coherence=np.where(trace>=1e-12,np.clip(gap/np.maximum(trace,1e-12),0,1),0)
    return coherence.astype(np.float32),np.sqrt(trace).astype(np.float32)
