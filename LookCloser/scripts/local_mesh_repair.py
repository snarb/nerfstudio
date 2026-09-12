"""Local, explicit mesh edits with preserved outside geometry and replay receipts."""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def boundary_loops(triangles):
    """Only simple degree-two boundary components qualify; branchy ones are rejected."""
    triangles=np.asarray(triangles)
    directed=triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    edges=np.sort(directed,axis=1);_,inverse,count=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
    boundary=directed[count[inverse]==1];adj=defaultdict(list)
    for a,b in boundary:adj[int(a)].append(int(b));adj[int(b)].append(int(a))
    remaining=set(adj);loops=[];rejected=[]
    while remaining:
        start=min(remaining);stack=[start];component=set()
        while stack:
            v=stack.pop()
            if v in component:continue
            component.add(v);stack.extend(n for n in adj[v] if n not in component)
        remaining-=component
        if any(len(adj[v])!=2 for v in component):rejected.append(sorted(component));continue
        loop=[start];previous=None;current=start
        while True:
            nxt=next(v for v in adj[current] if v!=previous)
            if nxt==start:break
            loop.append(nxt);previous,current=current,nxt
            if len(loop)>len(component):raise ValueError('Invalid boundary cycle')
        if len(loop)!=len(component):raise ValueError('Incomplete boundary cycle')
        oriented=set(map(tuple,boundary))
        if (loop[0],loop[1]) not in oriented:loop=loop[::-1]
        loops.append(np.array(loop,np.int32))
    return loops,rejected


def project_crop(points,row,box):
    from joint_temporal_texture import project
    uv,z=project(points,[row]);uv=uv[0];x0,y0,x1,y1=box
    # Native integer-centered -> rotated 90deg CCW and resized by two.
    result=np.column_stack(((uv[:,1]-y0+.5)*2-.5,((x1-x0)-.5-(uv[:,0]-x0))*2-.5))
    return result,z[0]


def mask_votes(points,views,region):
    import cv2
    from PIL import Image,ImageDraw
    votes=np.zeros(len(points),np.uint8)
    for row,box,regions in views:
        image=Image.new('L',(1024,1024));ImageDraw.Draw(image).polygon([tuple(p) for p in regions[region]],fill=255)
        uv,z=project_crop(points,row,box)
        sample=np.concatenate([cv2.remap(np.asarray(image),uv[s:s+16000,0].astype(np.float32)[None],
                            uv[s:s+16000,1].astype(np.float32)[None],cv2.INTER_NEAREST)[0] for s in range(0,len(uv),16000)])
        votes+=(sample>0)&(z>0)
    return votes


def close_selected_holes(vertices,triangles,selected_loops,*,edge_length=.0005):
    """MeshLab ear filling on explicitly reviewed loops; all old vertices fixed."""
    import pymeshlab as ml
    vertices=np.asarray(vertices,np.float64);triangles=np.asarray(triangles,np.int32)
    if not selected_loops:raise ValueError('No explicitly selected boundary loops')
    allowed_edges=set()
    for loop in selected_loops:
        allowed_edges.update(tuple(sorted((int(a),int(b)))) for a,b in zip(loop,np.roll(loop,-1)))
    selected_vertices=np.unique(np.concatenate(selected_loops));selected=np.isin(triangles,selected_vertices).any(1)
    mesh=ml.Mesh(vertex_matrix=vertices,face_matrix=triangles,f_scalar_array=selected.astype(np.float64))
    ms=ml.MeshSet();ms.add_mesh(mesh);ms.compute_selection_by_condition_per_face(condselect='fq > 0.5')
    ms.meshing_close_holes(maxholesize=max(map(len,selected_loops))+1,selected=True,newfaceselected=True,
                          selfintersection=True,refinehole=True,refineholeedgelen=ml.PureValue(edge_length))
    result=ms.current_mesh();v=result.vertex_matrix();t=result.face_matrix()
    if len(t)<=len(triangles):raise ValueError('Selected holes were not filled')
    if not np.array_equal(v[:len(vertices)],vertices) or not np.array_equal(t[:len(triangles)],triangles):
        raise ValueError('Hole fill modified pre-existing geometry')
    def boundary_edges(faces):
        edges=np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
        unique,count=np.unique(edges,axis=0,return_counts=True)
        return set(map(tuple,unique[count==1])),int((count>2).sum())
    before,nonmanifold_before=boundary_edges(triangles);after,nonmanifold_after=boundary_edges(t)
    if not allowed_edges.issubset(before):raise ValueError('Selected edges are not original hole boundaries')
    if allowed_edges & after:raise ValueError('Selected hole remains partially open')
    if after-before:raise ValueError('Hole fill introduced new boundary edges')
    if not (before-allowed_edges).issubset(after):raise ValueError('Filled an unapproved hole')
    if nonmanifold_after>nonmanifold_before:raise ValueError('Introduced non-manifold edges')
    if not (before-after).issubset(allowed_edges):raise ValueError('Unexpected boundary edit')
    return v,t,{'method':'MeshLab selected ear filling with refined interior',
                 'selected_loop_edge_counts':list(map(len,selected_loops)),'added_vertices':len(v)-len(vertices),
                 'added_triangles':len(t)-len(triangles),'closed_boundary_edges':len(before-after),
                 'other_boundary_edges_unchanged':True,'all_original_vertices_unchanged':True,
                 'nonmanifold_edges_before':nonmanifold_before,'nonmanifold_edges_after':nonmanifold_after,
                 'self_intersection_guard':True,'global_self_intersection_free_not_certified':True}


def cylinder_geometry(parameters,x_reference):
    center=np.array([x_reference,parameters[0],parameters[1]])
    axis=np.array([1.,parameters[2],parameters[3]]);axis/=np.linalg.norm(axis)
    radius=np.exp(parameters[4]);return center,axis,radius


def conservative_depth_removal(proposed,base_ids,near_counts,free_counts):
    """An object-region proposal cannot overrule two real surface observations."""
    proposed=np.asarray(proposed,bool);ids=np.asarray(base_ids)
    if proposed.shape!=ids.shape:raise ValueError('Face proposal inventory mismatch')
    index=np.maximum(ids,0)
    return proposed&(ids>=0)&(np.asarray(near_counts)[index]<2)&(np.asarray(free_counts)[index]>=3)


def fit_cylinder(points,views,observations):
    """Real-depth side points plus weak synthetic silhouette constraints.

    The cylinder is a disclosed shape prior, not independent evidence. Its axis
    is near normalized +X for this physically vertical object in the fixed rig.
    """
    from scipy.optimize import least_squares
    points=np.asarray(points,np.float64);initial=points.mean(0);xref=initial[0]
    circle=np.linspace(0,2*np.pi,64,endpoint=False)
    def residual(parameters,return_diagnostics=False):
        center,axis,radius=cylinder_geometry(parameters,xref);delta=points-center
        radial=delta-np.outer(delta@axis,axis);real=np.linalg.norm(radial,axis=1)-radius
        basis=np.cross(axis,[0.,0.,1.]);basis/=np.linalg.norm(basis);other=np.cross(axis,basis)
        ring=center+radius*(np.cos(circle)[:,None]*basis+np.sin(circle)[:,None]*other)
        synthetic=[];details=[]
        for (row,box,_),obs in zip(views,observations):
            uv,z=project_crop(center[None],row,box);uv=uv[0];ring_uv,_=project_crop(ring,row,box)
            predicted_half_width=np.ptp(ring_uv[:,0])/2
            target_x=np.polyval(obs['centerline_x_of_y'],uv[1]);target_width=obs['median_half_width']
            error=np.array([uv[0]-target_x,predicted_half_width-target_width])
            synthetic.extend(error*z[0]/(row['fl_x']*2)*np.sqrt(len(points)/6)*.4)
            details.append({'centerline_error_crop_pixels':float(error[0]),'half_width_error_crop_pixels':float(error[1])})
        if return_diagnostics:return real,details
        return np.r_[real,synthetic]
    best=None
    for dy,dz in [(0,0),(.001,0),(-.001,0),(0,.001),(0,-.001)]:
        p=[initial[1]+dy,initial[2]+dz,0,0,np.log(.0009)]
        result=least_squares(residual,p,bounds=([initial[1]-.004,initial[2]-.004,-.6,-.6,np.log(.0004)],
                    [initial[1]+.004,initial[2]+.004,.6,.6,np.log(.0025)]),loss='soft_l1',f_scale=.00015,max_nfev=300,
                    diff_step=1e-3,xtol=1e-10,ftol=1e-10,gtol=1e-10)
        if best is None or result.cost<best.cost:best=result
    center,axis,radius=cylinder_geometry(best.x,xref);real,details=residual(best.x,True)
    def endpoint_residual(t,which):
        values=[]
        for (row,box,_),obs in zip(views,observations):
            uv,_=project_crop((center+axis*t[0])[None],row,box)
            desired=obs['top_y']+1 if which=='top' else 805.
            values.append(uv[0,1]-desired)
        return values
    upper=least_squares(lambda t:endpoint_residual(t,'top'),[.002],diff_step=1e-3).x[0]
    lower=least_squares(lambda t:endpoint_residual(t,'bottom'),[-.004],diff_step=1e-3).x[0]
    if not lower<upper or not .002<upper-lower<.015:raise ValueError('Invalid inferred cylinder extent')
    report={'center':center.tolist(),'axis':axis.tolist(),'radius':float(radius),'lower':float(lower),'upper':float(upper),
            'real_support_points':len(points),'real_radial_error_median_p95':np.quantile(np.abs(real),[.5,.95]).tolist(),
            'synthetic_silhouette_residuals':details,'synthetic_prior_weight':.4,'optimizer_success':bool(best.success),
            'actual_physical_shape_not_certified':True}
    report['passes_real_radial_gate']=bool(np.median(np.abs(real))<=.0005)
    return report
