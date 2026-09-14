"""Reparameterize one fitted inverse-depth quadric without refitting its shape."""
import numpy as np


def symmetric(a,b):return (np.outer(a,b)+np.outer(b,a))/2


def camera_linear_forms(camera):
    pose=np.asarray(camera['transform_matrix'],float);r=pose[:3,:3];c=pose[:3,3]
    x=np.r_[r[:,0],-r[:,0]@c];y=np.r_[r[:,1],-r[:,1]@c];z=np.r_[-r[:,2],r[:,2]@c]
    return x,y,z


def world_quadric(camera,fit):
    x,y,z=camera_linear_forms(camera);cx,cy=fit['reference_center']
    u=(camera['fl_x']*x+(camera['cx']-cx)*z)/100
    v=(-camera['fl_y']*y+(camera['cy']-cy)*z)/100
    a=np.asarray(fit['all_camera_coefficients']);one=np.array([0,0,0,1.])
    return a[0]*symmetric(u,z)+a[1]*symmetric(v,z)+a[2]*np.outer(z,z)+a[3]*np.outer(u,u) \
        +a[4]*symmetric(u,v)+a[5]*np.outer(v,v)-symmetric(z,one)


def world_plane(camera,coefficients):
    x,y,z=camera_linear_forms(camera);a=np.asarray(coefficients)
    return a[0]*(camera['fl_x']*x+camera['cx']*z)/100 \
        +a[1]*(-camera['fl_y']*y+camera['cy']*z)/100+a[2]*z-np.array([0,0,0,1.])


def intersect_near_plane(center,directions,quadric,plane):
    center=np.r_[np.asarray(center),1.];d=np.column_stack([directions,np.zeros(len(directions))])
    denominator=d@plane
    with np.errstate(divide='ignore',invalid='ignore'):
        plane_t=-(center@plane)/denominator
    a=np.einsum('ni,ij,nj->n',d,quadric,d);b=2*(d@quadric@center);c=float(center@quadric@center)
    discriminant=b*b-4*a*c;roots=np.full((len(d),2),np.nan)
    quadratic=(np.abs(a)>1e-12)&(discriminant>=0)
    q=-.5*(b+np.where(b>=0,1.,-1.)*np.sqrt(np.maximum(discriminant,0)))
    with np.errstate(divide='ignore',invalid='ignore'):
        roots[quadratic,0]=q[quadratic]/a[quadratic]
        roots[quadratic,1]=c/q[quadratic]
        linear=(~quadratic)&(np.abs(a)<=1e-12)&(np.abs(b)>1e-12)
        roots[linear,0]=-c/b[linear]
    distances=np.where(np.isfinite(roots)&(roots>0),np.abs(roots-plane_t[:,None]),np.inf)
    selected=roots[np.arange(len(d)),distances.argmin(1)]
    valid=np.isfinite(distances.min(1))&np.isfinite(plane_t)&(plane_t>0)
    return np.where(valid,selected,np.nan),plane_t
