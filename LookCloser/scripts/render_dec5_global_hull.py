#!/usr/bin/env python3
"""CPU canary for global head geometry; fixed original-MVS RGB transfer.

Fresh raycasts recover surfaces behind removed triangles. Raw hair fallback is
allowed only inside the global foreground envelope; outside it, an independently
visible background layer is used. Body and protected face pixels retain B0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from dec5_hair import cv_pose, project, unproject
from study_dec5_hair_geometry import Scene, mesh_depth, display
from run_dec5_hair_campaign import atomic_json, sha


def bilinear(array, uv):
    """Exact continuous interpolation, avoiding OpenCV's 1/32-pixel quantization."""
    x,y=uv[:,0]-.5,uv[:,1]-.5
    x0,y0=np.floor(x).astype(int),np.floor(y).astype(int)
    dx,dy=x-x0,y-y0
    result=np.zeros((len(uv),)+array.shape[2:],np.float32)
    for oy,ox in ((0,0),(0,1),(1,0),(1,1)):
        xx,yy=x0+ox,y0+oy
        valid=(xx>=0)&(xx<array.shape[1])&(yy>=0)&(yy<array.shape[0])
        weight=(dx if ox else 1-dx)*(dy if oy else 1-dy)
        if array.ndim==3:weight=weight[:,None]
        result[valid]+=array[yy[valid],xx[valid]]*weight[valid]
    return result


class Transfer:
    def __init__(self,scene):
        self.scene=scene;self.rgb={}
        self.sources={r['camera']:r for r in json.loads((scene.root/'prepare.json').read_text())['records']}

    def warp(self,camera,uv,depth,background):
        xyz=unproject(uv,depth,camera)
        n=len(uv);rgb=np.zeros((n,3),np.float32);ids=np.full(n,-1,np.int16)
        support=np.zeros(n,np.int16);hair=np.zeros(n,np.float32);skin=hair.copy();error=hair.copy()
        order=np.argsort(np.linalg.norm(self.scene.centers-cv_pose(camera)[:3,3],axis=-1),kind='stable')[:8]
        for index in order:
            source=self.scene.cameras[index]
            name=self.scene.frames[index]['physical_camera']
            if index not in self.rgb:
                path=self.scene.root/'source/images'/Path(self.sources[name]['original_path']).name
                self.rgb[index]=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)[...,::-1].copy()
            p,z=project(xyz,source)
            observed=bilinear(self.scene.observed_depth[index],p)
            semantic=bilinear(self.scene.semantic[index],p)
            valid=((depth>0)&(z>0)&(observed>0)&(p[:,0]>=.5)&(p[:,0]<=source['w']-.5)
                   &(p[:,1]>=.5)&(p[:,1]<=source['h']-.5)&(abs(observed-z)<=.01*z+.002))
            valid&=~background|(semantic[:,2]>.8)
            support+=valid;hair+=valid*semantic[:,0];skin+=valid*semantic[:,1]
            back,_=project(unproject(p,observed,source),camera)
            error+=valid*np.minimum(np.linalg.norm(back-uv,axis=-1),20)
            take=valid&(ids<0)
            colors=bilinear(self.rgb[index],p)
            rgb[take]=colors[take];ids[take]=index
        return dict(rgb=rgb,source_id=ids,support=support,hair_probability=hair/np.maximum(support,1),
                    skin_probability=skin/np.maximum(support,1),
                    reprojection_error=np.where(support>0,error/np.maximum(support,1),20))


def render(root,output,mesh_path,baseline,hull,manifest=None):
    output.mkdir(parents=True,exist_ok=False)
    scene=Scene(root,load_rgb=False,mvs=root/'pm1920_native_v2')
    transfer=Transfer(scene)
    field=np.load(hull/'field.npz');soft=field['field'][...,0];known=field['known']
    lower=field['lower'];spacing=float(field['spacing']);upper=lower+(np.asarray(soft.shape)-1)*spacing
    corners=np.array(np.meshgrid(*zip(lower,upper),indexing='ij')).reshape(3,-1).T
    cameras=json.loads(manifest.read_text())['frames'] if manifest else scene.validation
    records=[]
    for index,camera in enumerate(cameras):
        started=time.monotonic();name=camera['physical_camera']
        source=np.load(baseline/(name+'.npz'))
        arrays={key:source[key].copy() for key in source.files}
        depth=mesh_depth(mesh_path,camera)
        h,w=depth.shape
        box,z=project(corners,camera);x0,y0=np.floor(box.min(0)).astype(int);x1,y1=np.ceil(box.max(0)).astype(int)
        head=np.zeros((h,w),bool);head[max(0,y0):min(h,y1+1),max(0,x0):min(w,x1+1)]=True
        domain=head&(arrays['skin_probability']<.3)&arrays['roi']
        yy,xx=np.nonzero(domain);uv=np.stack((xx+.5,yy+.5),-1)
        proposed=depth[yy,xx].copy();background=np.zeros(len(uv),bool)
        missing=proposed<=0
        raw=arrays['projected_hair_depth'][yy,xx]
        points=unproject(uv,raw,camera);ijk=(points-lower)/spacing
        inside=((ijk>=0)&(ijk<=np.asarray(soft.shape)-1)).all(-1)
        value=map_coordinates(soft,ijk.T,order=1,mode='constant',cval=-2*spacing,prefilter=False)
        observed=map_coordinates(known.astype(np.float32),ijk.T,order=0,mode='constant',cval=0,prefilter=False)
        consistent=inside&((value>=-spacing)|(observed<12))&(raw>0)
        proposed[missing&consistent]=raw[missing&consistent]
        background=missing&~consistent
        proposed[background]=arrays['back_depth'][yy[background],xx[background]]
        # Only actual depth changes can affect source RGB. Preserve the others
        # exactly, including their original CUDA interpolation and source IDs.
        changed=abs(proposed-arrays['front_depth'][yy,xx])>1e-5
        changed|=(proposed<=0)&(arrays['front_depth'][yy,xx]>0)
        y,x=yy[changed],xx[changed]
        result=transfer.warp(camera,uv[changed],proposed[changed],background[changed])
        protected=result['skin_probability']>=.3
        take=~protected
        for key,value in result.items():arrays[key][y[take],x[take]]=value[take]
        arrays['front_depth'][y[take],x[take]]=proposed[changed][take]
        edit=np.zeros((h,w),bool);edit[y[take],x[take]]=True
        arrays['edit_roi']=edit;arrays['mesh_depth']=depth
        np.testing.assert_array_equal(arrays['rgb'][~edit],source['rgb'][~edit])
        np.savez_compressed(output/(name+'.npz'),**arrays)
        rgb=np.rint(display(arrays['rgb'])*255).astype(np.uint8)
        cv2.imwrite(str(output/(name+'.png')),rgb[...,::-1])
        records.append(dict(camera=name,npz_sha256=sha(output/(name+'.npz')),png_sha256=sha(output/(name+'.png')),
                            edited_pixels=int(edit.sum()),background_hypotheses=int(background[changed][take].sum()),
                            seconds=time.monotonic()-started))
        atomic_json(output/'progress.json',dict(completed=index+1,total=len(cameras)))
    receipt=json.loads((baseline/'complete.json').read_text())
    receipt.update(records=records,method='global silhouette-constrained mesh',
        parent_receipt_sha256=sha(baseline/'complete.json'),mesh_sha256=sha(mesh_path),
        hull_sha256=sha(hull/'complete.json'),
        background_policy='original measured depth + semantic background >.8; no target RGB',
        source_interpolation='continuous bilinear NumPy, only changed depths; exact original RGB elsewhere',
        source_visibility_original_measured_depth=True,implementation_sha256=sha(__file__))
    atomic_json(output/'complete.json',receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','mesh','baseline','hull'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--camera-manifest',type=Path)
    a=p.parse_args();render(a.root,a.output,a.mesh,a.baseline,a.hull,a.camera_manifest)


if __name__=='__main__':main()
