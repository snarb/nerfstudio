"""Rebuild the reference proposal with explicit geometry-before-admission order.

The plane control exactly replays V3 admission, including its integer mask lookup
and renderer-footprint availability. Only the points queried by that same policy
change in the quadric arm. Final boundary feather and semantic/extent guards are
still applied by the existing production-delta runner.
"""
import numpy as np
import open3d as o3d
from scipy import ndimage
from joint_temporal_texture import read
from study_confidence_depth_prior import project_integer,unproject
from confidence_boundary_completion import grid_faces
import study_forearm_plane_transfer_v3 as prior


def point_votes(points,rows,names,masks,data,depths,semantic_domain):
    support=np.zeros(len(points),np.uint8);disagree=np.zeros(len(points),np.uint8);free=np.zeros(len(points),np.uint8)
    for name in names:
        i=next(i for i,r in enumerate(rows) if r['physical_camera']==name);row=rows[i]
        uv,z=project_integer(row,points);xy=np.rint(uv).astype(int)
        inside=(z>0)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
        ids=np.flatnonzero(inside);qx,qy=xy[ids].T
        available=semantic_domain(row,points,inside);selected=np.flatnonzero(available)
        sx,sy=xy[selected].T;skin=masks[name][sy,sx]
        support[selected]+=skin;disagree[selected]+=~skin
        free[ids]+=data[name+'_trusted'][qy,qx]&(depths[i][qy,qx]>z[ids]+.003)
    return support,disagree,free


def rebuild(frame,shape,fit):
    if shape not in ['plane','quadric']:raise ValueError('Unsupported admission geometry')
    prior.configure();v1=prior.v2.v1;root=prior.OUT/frame
    rows,depths,hashes=v1.load_real(frame);reference=next(r for r in rows if r['physical_camera']==v1.NAMES[0])
    spec=read(root/'input.json');analysis=read(root/'analysis.json');data=np.load(root/'diagnostic.npz');masks=v1.masks(frame)
    if hashes!=analysis['source_depth_sha256']:raise ValueError('Changed depth inputs')
    md=data[v1.NAMES[0]+'_mesh'];trusted=data[v1.NAMES[0]+'_trusted']
    y,x=np.nonzero(masks[v1.NAMES[0]]&(md==0));xy=np.column_stack([x,y])
    plane=1/(np.column_stack([xy/100,np.ones(len(x))])@np.array(analysis['plane_inverse_coefficients']))
    q=(xy-np.array(fit['reference_center']))/100
    inverse=np.column_stack([q,np.ones(len(q)),q[:,0]**2,q[:,0]*q[:,1],q[:,1]**2])@np.array(fit['all_camera_coefficients'])
    if not np.isfinite(inverse).all() or (inverse<=0).any():raise ValueError('Invalid admission quadric')
    evaluated=plane if shape=='plane' else 1/inverse
    points=unproject(reference,x,y,evaluated)
    support,disagree,free=point_votes(points,rows,v1.NAMES,masks,data,depths,prior.v2.semantic_domain)
    distance=ndimage.distance_transform_edt(~trusted)[y,x]
    selected=prior.v2.eligible_points(evaluated,distance,support,disagree,free)
    within_bound=np.abs(evaluated-plane)<=.01
    excluded_by_bound=int((selected&~within_bound).sum());selected&=within_bound
    # Keep the raw planar grid and original boundary interpolation unchanged;
    # the caller applies exactly the existing boundary-conditioned curvature.
    added=np.zeros(md.shape,np.float32);added[y[selected],x[selected]]=plane[selected];accepted=added>0
    baseline=np.load(root/'plane/evidence.npz')['accepted']
    if shape=='plane' and not np.array_equal(accepted,baseline):raise ValueError('Plane admission failed exact replay')
    domain=ndimage.binary_dilation(accepted)&((md>0)|accepted);yy,xx=np.nonzero(domain)
    base=o3d.io.read_triangle_mesh(spec['mesh']);v,t=np.asarray(base.vertices),np.asarray(base.triangles)
    newv=unproject(reference,xx,yy,np.where(accepted,added,md)[yy,xx]);vertices=np.vstack([v,newv])
    index=np.full(md.shape,-1,np.int32);index[yy,xx]=np.arange(len(xx))+len(v)
    faces=grid_faces(domain,accepted,index);faces=faces[np.ptp(vertices[faces],axis=1).max(1)<.002]
    triangles=np.vstack([t,faces])
    if shape=='plane':
        original=o3d.io.read_triangle_mesh(str(root/'plane/mesh.ply'))
        if not np.array_equal(vertices,np.asarray(original.vertices)) or not np.array_equal(triangles,np.asarray(original.triangles)):
            raise ValueError('Plane grid failed exact array replay')
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(triangles))
    record=dict(admission_shape=shape,previous_accepted=int(baseline.sum()),accepted=int(accepted.sum()),
        newly_accepted=int((accepted&~baseline).sum()),previously_accepted_removed=int((baseline&~accepted).sum()),
        rejected_semantic=int(((support<2)|(disagree>0)).sum()),rejected_trusted_free=int((free>0).sum()),
        raw_grid_triangles=len(faces),plane_exact_replay=shape=='plane',initial_lookup_unchanged=True,
        excluded_by_curvature_bound=excluded_by_bound,
        max_selected_depth_change=float(np.max(np.abs(evaluated[selected]-plane[selected]))),
        max_raw_evaluated_depth_change=float(np.max(np.abs(evaluated-plane))))
    arrays=dict(accepted=accepted,reference_xy=xy,evaluated_depth=evaluated,plane_depth=plane,
                skin_support=support,skin_disagreement=disagree,trusted_free=free,selected=selected)
    return mesh,accepted,record,arrays
