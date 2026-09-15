"""Exact calibrated portrait stereo geometry in the current normalized world."""
import cv2
import numpy as np


def portrait_calibration(row):
    if any(abs(row.get(k,0.))>1e-12 for k in ['k1','k2','p1','p2']):
        raise ValueError('This adapter requires already pinhole/undistorted calibration')
    # Renderer pixels address RGB array centers at K principal point minus .5.
    intrinsic=np.array([[row['fl_y'],0,row['cy']-.5],
                        [0,row['fl_x'],row['w']-.5-row['cx']],[0,0,1]],float)
    rotate=np.array([[0.,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
    extrinsic=rotate@np.linalg.inv(np.array(row['transform_matrix'])@np.diag([1.,-1.,-1.,1.]))
    return intrinsic,extrinsic


def rectify_pair(left,right):
    if (left['w'],left['h'])!=(right['w'],right['h']):raise ValueError('Unequal source sizes')
    k1,e1=portrait_calibration(left);k2,e2=portrait_calibration(right)
    relative=e2@np.linalg.inv(e1);size=(left['h'],left['w'])
    r1,r2,p1,p2,q,roi1,roi2=cv2.stereoRectify(k1,None,k2,None,size,relative[:3,:3],relative[:3,3],
        flags=cv2.CALIB_ZERO_DISPARITY,alpha=-1,newImageSize=size)
    if p1[0,0]<=0:raise ValueError('Invalid nonpositive rectified focal length')
    if abs(p2[1,3])>1e-8:raise ValueError('Vertical baseline: not compatible with horizontal stereo inference')
    if p2[0,3]>=0:raise ValueError('Swap physical left/right cameras for positive disparity')
    np.testing.assert_allclose(p1[:,:3],p2[:,:3],atol=1e-7,rtol=0)
    maps=[]
    for k,r,p in [(k1,r1,p1),(k2,r2,p2)]:
        maps.append(cv2.initUndistortRectifyMap(k,None,r,p[:,:3],size,cv2.CV_32FC1))
    rectified_extrinsic=np.eye(4);rectified_extrinsic[:3,:3]=r1@e1[:3,:3];rectified_extrinsic[:3,3]=r1@e1[:3,3]
    return dict(K1=k1,K2=k2,E1=e1,E2=e2,R1=r1,R2=r2,P1=p1,P2=p2,Q=q,
                rectified_extrinsic=rectified_extrinsic,baseline=-p2[0,3]/p1[0,0],size=size),maps


def rectify_local_pair(left,right,focus,size=(768,768),focus_disparity=128.):
    """Keep native focal scale and recenter both local fields independently.

    Narrow convergent cameras can need disparity beyond an image width under
    zero-disparity rectification. Different principal points keep the local
    matching range useful; their offset must be restored in metric depth.
    This changes projection matrices, not physical camera poses.
    """
    cal,_=rectify_pair(left,right)
    e=cal['rectified_extrinsic'];point=e[:3,:3]@np.asarray(focus)+e[:3,3]
    if point[2]<=0:raise ValueError('Local focus behind rectified camera')
    f=float((cal['K1'][0,0]+cal['K2'][0,0])/2)
    k=np.array([[f,0,(size[0]-1)/2-f*point[0]/point[2]],
                [0,f,(size[1]-1)/2-f*point[1]/point[2]],[0,0,1]])
    kr=k.copy();kr[0,2]+=f*cal['baseline']/point[2]-focus_disparity
    p1=np.column_stack((k,np.zeros(3)));p2=np.column_stack((kr,[-f*cal['baseline'],0,0]))
    offset=k[0,2]-kr[0,2]
    q=np.array([[1,0,0,-k[0,2]],[0,1,0,-k[1,2]],[0,0,0,f],[0,0,1/cal['baseline'],-offset/cal['baseline']]])
    maps=[cv2.initUndistortRectifyMap(ks,None,r,kt,size,cv2.CV_32FC1)
          for ks,r,kt in [(cal['K1'],cal['R1'],k),(cal['K2'],cal['R2'],kr)]]
    cal.update(P1=p1,P2=p2,Q=q,size=size,disparity_offset=offset,focus=np.array(focus),focus_disparity=focus_disparity)
    return cal,maps


def disparity_to_world(x,y,disparity,intrinsic,extrinsic,baseline,disparity_offset=0.):
    x,y,disparity=np.broadcast_arrays(x,y,disparity)
    effective=disparity-disparity_offset
    if not np.isfinite(effective).all() or (effective<=0).any() or baseline<=0:raise ValueError('Expected finite positive metric disparity and baseline')
    z=intrinsic[0,0]*baseline/effective
    points=np.stack(((x-intrinsic[0,2])*z/intrinsic[0,0],(y-intrinsic[1,2])*z/intrinsic[1,1],z),-1)
    return (points-extrinsic[:3,3])@extrinsic[:3,:3]
