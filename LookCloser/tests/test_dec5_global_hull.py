from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_dec5_global_hull import portrait_crop_uv, insert_three


def test_portrait_projection_matches_actual_rotated_crop_pixels():
    image=np.arange(35).reshape(5,7)
    crop=[1,2,5,6]
    portrait=np.rot90(image)[crop[1]:crop[3],crop[0]:crop[2]]
    yy,xx=np.indices(image.shape)
    uv=portrait_crop_uv(np.stack((xx+.5,yy+.5),-1),7,crop)
    valid=(uv[...,0]>=.5)&(uv[...,0]<portrait.shape[1])&(uv[...,1]>=.5)&(uv[...,1]<portrait.shape[0])
    samples=portrait[np.floor(uv[...,1][valid]).astype(int),np.floor(uv[...,0][valid]).astype(int)]
    np.testing.assert_array_equal(samples,image[valid])


def test_robust_silhouette_statistic_ignores_unknown_camera_samples():
    data=np.array([[-9.,2.],[np.inf,1.],[3.,np.inf],[1.,4.],[-2.,3.]])
    a=torch.full((2,),float('inf'));b=a.clone();c=a.clone()
    for row in data:a,b,c=insert_three(a,b,c,torch.tensor(row))
    np.testing.assert_array_equal(c.numpy(),np.sort(data,axis=0)[2])
