"""Exact current-calibration RAW-mask to undistorted-camera raster contracts."""
from pathlib import Path
import numpy as np
import cv2
RIG=Path('/fsx/oregon/gta_marshawn_2_1_prep_20261028/root/Marshawn_2_1/golden/rig')

def raw_camera(name):
 from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_text,read_images_text
 images=read_images_text(RIG/'images.txt');cam=read_cameras_text(RIG/'cameras.txt')[next(v.camera_id for v in images.values()if v.name.startswith(name+'/'))]
 if cam.model!='OPENCV' or (cam.width,cam.height)!=(8192,4320):raise ValueError('Expected current OPENCV native RAW camera')
 fx,fy,cx,cy,*d=cam.params;K=np.array([[fx,0,cx-.5],[0,fy,cy-.5],[0,0,1.]],np.float64)
 return K,np.asarray(d,np.float64)

def target_intrinsics(row,width=8192,height=4320):
 sx=width/row['w'];sy=height/row['h']
 return dict(w=width,h=height,fl_x=row['fl_x']*sx,fl_y=row['fl_y']*sy,cx=row['cx']*sx,cy=row['cy']*sy)

def maps(row):
 K,d=raw_camera(row['physical_camera']);dst=np.array([[row['fl_x'],0,row['cx']-.5],[0,row['fl_y'],row['cy']-.5],[0,0,1.]])
 return cv2.initUndistortRectifyMap(K,d,None,dst,(row['w'],row['h']),cv2.CV_32FC1)

def raw_array_indices(uv_colmap,row):
 """Independent explicit OPENCV polynomial, u/v use COLMAP half-pixel centres."""
 K,d=raw_camera(row['physical_camera']);uv=np.asarray(uv_colmap,np.float64);x=(uv[...,0]-row['cx'])/row['fl_x'];y=(uv[...,1]-row['cy'])/row['fl_y'];k1,k2,p1,p2=d;r=x*x+y*y;rad=1+k1*r+k2*r*r;xd=x*rad+2*p1*x*y+p2*(r+2*x*x);yd=y*rad+p1*(r+2*y*y)+2*p2*x*y
 return np.stack([K[0,0]*xd+K[0,2],K[1,1]*yd+K[1,2]],axis=-1)

def raw_mask_coverage_to_HD(raw_mask,row):
 """Match RGB's AREA-prefilter-RAW-before-single-remap source footprint.

This maps a RAW semantic coverage estimate; it does not assert measured matte
alpha or infer actor pixels behind an occluder. Never resize a distorted mask
straight into an undistorted dataset.
 """
 x=np.asarray(raw_mask)
 if x.shape!=(4320,8192)or not np.isfinite(x).all()or x.min()<0 or x.max()>1:raise ValueError('Expected finite native RAW maskcoverage[0,1]')
 mx,my=maps(row);w,h=row['w'],row['h'];small=cv2.resize(x.astype(np.float32),(w,h),interpolation=cv2.INTER_AREA);out=cv2.remap(small,(mx+.5)*(w/8192)-.5,(my+.5)*(h/4320)-.5,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
 return np.clip(out,0,1)

def raw_mask_to_native_undistorted(raw_mask,row):
 full=dict(row,**target_intrinsics(row));mx,my=maps(full)
 return cv2.remap(np.asarray(raw_mask,np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT),full
