"""Build a coherent field within unchanged annotation and depth bounds."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from study_confidence_depth_prior import unproject,project_integer
from joint_temporal_texture import project
from annotation_mask_domain import known_domain
from forearm_quadric_rays import world_quadric,world_plane,intersect_near_plane
from ordered_forearm_admission import point_votes
from discrete_surface_constraints import solve
import study_forearm_plane_transfer_v3 as prior


def build(camera,reference,fit,analysis,rows,depths,masks,data):
    name=camera['physical_camera'];y,x=np.nonzero(masks[name]);center=np.asarray(camera['transform_matrix'])[:3,3]
    directions=unproject(camera,x,y,np.ones(len(x)))-center
    model_z,_=intersect_near_plane(center,directions,world_quadric(reference,fit),world_plane(reference,analysis['plane_inverse_coefficients']))
    finite=np.isfinite(model_z)&(model_z>.012);x,y,model_z=x[finite],y[finite],model_z[finite]
    names=prior.v2.v1.NAMES
    distance=distance_transform_edt(~data[name+'_trusted'])[y,x]
    def eligible(z):
        points=unproject(camera,x,y,z);uv,rz=project_integer(reference,points)
        inverse=np.column_stack([uv/100,np.ones(len(uv))])@analysis['plane_inverse_coefficients']
        within=(inverse>0)&(np.abs(rz-1/np.maximum(inverse,1e-12))<=.01)&(distance<=100)
        votes,negative,free=point_votes(points,rows,names,masks,data,depths,prior.v2.semantic_domain)
        # Enforce the existing final-vertex mask convention as well as initial
        # integer admission, so the solve cannot cross their disagreement band.
        final_support=np.zeros(len(points),np.uint8);final_negative=np.zeros(len(points),bool)
        for row in rows:
            if row['physical_camera'] not in masks:continue
            q,zz=project(points,[row]);q,zz=q[0],zz[0];xy=np.rint(q).astype(int)
            available=known_domain(q,zz,row['w'],row['h']);ids=np.flatnonzero(available)
            inside=np.zeros(len(points),bool);inside[ids]=masks[row['physical_camera']][xy[ids,1],xy[ids,0]]
            final_support+=inside;final_negative|=available&~inside
        return within&(votes>=2)&(negative==0)&(free==0)&(final_support>=2)&~final_negative
    offsets=np.linspace(-.012,.012,49);allowed=np.stack([eligible(model_z+delta) for delta in offsets],1)
    observed=depths[next(i for i,r in enumerate(rows) if r['physical_camera']==name)][y,x]
    trusted=data[name+'_trusted'][y,x]
    pin_ok=trusted&np.isfinite(observed)&(observed>0)&(np.abs(observed-model_z)<=.012)
    checked=eligible(np.where(pin_ok,observed,model_z));pin_ok&=checked
    conflicts=trusted&~pin_ok
    selected=(allowed.any(1)|pin_ok)&~conflicts
    if pin_ok[selected].sum()<30:raise ValueError('Insufficient feasible native pins')
    domain=np.zeros(masks[name].shape,bool);domain[y[selected],x[selected]]=True
    model=np.zeros(domain.shape);model[y[selected],x[selected]]=model_z[selected]
    pins=np.zeros(domain.shape,bool);pins[y[selected],x[selected]]=pin_ok[selected]
    choices=allowed[selected].copy();choices[pin_ok[selected],0]=True
    options=np.broadcast_to(offsets,(int(selected.sum()),len(offsets)))
    residual,stats=solve(domain,options,choices,pin_ok[selected],observed[selected]-model_z[selected],.05)
    solved=np.zeros(domain.shape);solved[domain]=model[domain]+residual[domain]
    verify_z=model_z.copy();verify_z[selected]=solved[domain]
    if not eligible(verify_z)[selected].all():raise ValueError('Solved points left geometric feasible set')
    if np.max(np.abs(residual[domain]))>.01200000001:raise ValueError('Exceeded fixed residual bound')
    stats.update(trusted_pixels_excluded_for_conflicting_constraints=int(conflicts.sum()),
        all_selected_points_rechecked=True,maximum_residual=float(np.max(np.abs(residual[domain]))),
        original_plane_bound_rechecked=True,discrete_offsets=49,offset_step=.0005,
        final_mask_convention_constrained=True)
    arrays=dict(query_xy=np.column_stack([x,y]),query_model_z=model_z,offsets=offsets,allowed=allowed,
        selected=selected,query_pin_ok=pin_ok,query_trusted=trusted,query_observed=observed)
    return domain,model,solved,pins,stats,arrays
