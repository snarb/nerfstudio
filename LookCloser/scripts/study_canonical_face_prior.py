"""Regularized canonical human-face fit; original COLMAP mesh is never replaced.

One-time CPU pilot. MediaPipe z and prior independent landmark triangulations
are not inputs. All downstream geometry remains an explicitly inferred prior.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import urllib.request
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
from study_multiview_face_prior import sha,read,save,portrait_to_native,INTERIOR,CHEEK,JAW
from triangulate_face_prior import projection_matrices,project,quantiles

OUT=Path('/mnt/data/dec5_canonical_face_prior')
SOURCE=Path('/mnt/data/dec5_multiview_face_prior')
DEPTH=Path('/mnt/data/dec5_jaw_measured_depth/analysis')
FRAME='001193'
COMMIT='87f8074eb976e59655f243a8715c373e76ce3abb'
ASSETS={'canonical_face_model.obj':'mediapipe/modules/face_geometry/data/canonical_face_model.obj','LICENSE':'LICENSE'}
CORE=sorted(set(INTERIOR+CHEEK))

def parse_obj(path):
    vertices=[];faces=[]
    for line in Path(path).read_text().splitlines():
        row=line.split()
        if row and row[0]=='v':vertices.append([float(x) for x in row[1:4]])
        elif row and row[0]=='f':
            if len(row)!=4:raise ValueError('Expected canonical triangles')
            faces.append([int(x.split('/')[0])-1 for x in row[1:]])
    v=np.asarray(vertices,float);t=np.asarray(faces,np.int32)
    if v.shape!=(468,3) or t.ndim!=2 or t.shape[1]!=3 or t.min()<0 or t.max()>=468:raise ValueError('Unexpected canonical topology')
    return v,t

def smooth_basis(v,t,count=8):
    edges=np.unique(np.sort(t[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)
    a=np.zeros((len(v),len(v)));a[edges[:,0],edges[:,1]]=1;a[edges[:,1],edges[:,0]]=1
    lap=np.diag(a.sum(1))-a;values,vectors=eigh(lap,subset_by_index=[1,count])
    # Remove affine components to avoid duplicating the pose/global scale.
    affine=np.column_stack((np.ones(len(v)),v));q,_=np.linalg.qr(affine)
    b=vectors-q@(q.T@vectors);b,_=np.linalg.qr(b);b/=np.max(np.abs(b),axis=0)
    return b,values

def similarity(source,target):
    x=np.asarray(source);y=np.asarray(target);xc=x-x.mean(0);yc=y-y.mean(0)
    u,s,vt=np.linalg.svd(xc.T@yc);sign=np.ones(3);sign[-1]=np.linalg.det(vt.T@u.T)
    rotation=vt.T@np.diag(sign)@u.T;scale=(s*sign).sum()/(xc*xc).sum()
    translation=y.mean(0)-scale*x.mean(0)@rotation.T
    if scale<=0:raise ValueError('Invalid metric initialization')
    return scale,rotation,translation

def deform(parameters,v,basis):
    shaped=v.copy()
    if len(parameters)>7:shaped+=basis@parameters[7:].reshape(basis.shape[1],3)
    return np.exp(parameters[6])*shaped@Rotation.from_rotvec(parameters[:3]).as_matrix().T+parameters[3:6]

def assets():
    if OUT.exists():raise ValueError('Use a fresh canonical-prior root')
    OUT.mkdir(parents=True);records={}
    for name,path in ASSETS.items():
        url=f'https://raw.githubusercontent.com/google-ai-edge/mediapipe/{COMMIT}/{path}'
        urllib.request.urlretrieve(url,OUT/name);records[name]=dict(url=url,sha256=sha(OUT/name))
    license_text=(OUT/'LICENSE').read_text();assert 'Apache License' in license_text and 'Version 2.0' in license_text
    v,t=parse_obj(OUT/'canonical_face_model.obj');b,e=smooth_basis(v,t)
    np.savez_compressed(OUT/'canonical.npz',vertices=v,triangles=t,basis=b,eigenvalues=e)
    request=read(SOURCE/'request.json');stage=read(SOURCE/FRAME/'input.json')
    save(OUT/'protocol.json',dict(frame=FRAME,upstream_commit=COMMIT,assets=records,license='Apache-2.0 repository license; retained',
        canonical_npz_sha256=sha(OUT/'canonical.npz'),source_input_sha256=sha(SOURCE/FRAME/'input.json'),
        source_inference_sha256=sha(SOURCE/'inference.json'),source_request_sha256=sha(SOURCE/'request.json'),
        original_mesh=stage['mesh'],original_mesh_sha256=stage['mesh_sha256'],model_z_used=False,
        prior_independent_landmark_triangulations_used=False,fit_landmark_indices=CORE,
        validation_prefixes=request['validation_prefixes'],validation_used_for_depth_votes=False,
        arms=['similarity','regularized8'],shape_modes=8,shape_coefficients=24,
        shape_coefficient_sigma_fraction_width=.015,shape_coefficient_bound_fraction_width=.08,
        depth_agreement=.001,minimum_other_depth_votes=3,reprojection_pixels=1.5,minimum_parallax_degrees=1,
        landmark_sigma_pixels=3.,point_plane_sigma=.001,maximum_nonlinear_evaluations=150,
        acceptance=dict(validation_point_plane_p90=.002,validation_core_median_pixels=4.,validation_core_p90_pixels=8.,
            maximum_vertex_deformation_fraction_width=.15,maximum_split_fit_disagreement=.0015),
        locality=dict(maximum_prior_to_old_surface=.002,maximum_prior_to_original_boundary=.003,
            required_new_visibility_train_views=2,original_triangles_immutable=True),
        prior_is_inferred_not_observed=True,heldout_used=False,production_changed=False,script_sha256=sha(__file__)))
    print('Frozen canonical prior,',len(v),'vertices',len(t),'triangles',flush=True)

def stage():
    import open3d as o3d
    from joint_temporal_texture import cameras,HELD_CAMERAS
    from study_confidence_depth_prior import load_real,support,unproject
    from diffusion_mesh_repair import scene_for
    request=read(OUT/'protocol.json');spec=read(SOURCE/FRAME/'input.json');inference=read(SOURCE/'inference.json')
    assert sha(SOURCE/FRAME/'input.json')==request['source_input_sha256'];assert sha(SOURCE/'inference.json')==request['source_inference_sha256']
    if (OUT/'observations.npz').exists():raise ValueError('Keep frozen canonical observations')
    rows,depths,receipt=load_real(DEPTH,FRAME);assert not ({r['physical_camera'] for r in rows}&HELD_CAMERAS)
    validation=np.array([any(r['physical_camera'].startswith(p) for p in request['validation_prefixes']) for r in rows])
    fitrows=[r for r,k in zip(rows,validation) if not k];fitdepths=[d for d,k in zip(depths,validation) if not k]
    allrows={r['physical_camera']:i for i,r in enumerate(rows)}
    predictions={r['camera']:r for r in inference['records'] if r['frame']==FRAME and r['detected']==1}
    old=o3d.io.read_triangle_mesh(spec['mesh']);old.compute_triangle_normals();v=np.asarray(old.vertices);t=np.asarray(old.triangles)
    assert sha(spec['mesh'])==spec['mesh_sha256'];scene=scene_for(v,t);normal=np.asarray(old.triangle_normals)
    oval=set(np.array(read(SOURCE/'topology.json')['oval']).ravel().tolist())
    selected=sorted((set(range(0,468,4))|set(CORE))-oval)
    anchors=[];normals=[];indices=[];camera_ids=[];counts=[];landmark_uv=[];landmark_ids=[];landmark_camera=[];diagnostics=[]
    for ci,row in enumerate(rows):
        name=row['physical_camera']
        if name not in predictions:continue
        xy=portrait_to_native(predictions[name]['portrait_xy'])[:468];ids=np.array(selected);q=np.rint(xy[ids]).astype(int)
        inside=(q[:,0]>=0)&(q[:,0]<1920)&(q[:,1]>=0)&(q[:,1]<1080);ids=ids[inside];q=q[inside];pm=depths[ci][q[:,1],q[:,0]]
        center=np.asarray(row['transform_matrix'])[:3,3];points=unproject(row,q[:,0],q[:,1],np.ones(len(q)))
        rays=np.column_stack((np.broadcast_to(center,points.shape),points-center)).astype(np.float32)
        hit=scene.cast_rays(o3d.core.Tensor(rays));md=hit['t_hit'].numpy();tri=hit['primitive_ids'].numpy()
        valid=np.isfinite(md)&(pm>0)&(np.abs(pm-md)<=.001);valid_ids=np.flatnonzero(valid)
        measured=unproject(row,q[valid,0],q[valid,1],pm[valid]);votes,_=support(measured,row,fitrows,fitdepths)
        trusted=votes>=3;take=valid_ids[trusted]
        anchors.extend(measured[trusted]);normals.extend(normal[tri[take]]);indices.extend(ids[take]);camera_ids.extend([ci]*len(take));counts.extend(votes[trusted])
        for k in CORE:
            if 0<=xy[k,0]<1920 and 0<=xy[k,1]<1080:landmark_uv.append(xy[k]);landmark_ids.append(k);landmark_camera.append(ci)
        diagnostics.append(dict(camera=name,validation=bool(validation[ci]),queried=len(ids),original_and_depth_agree=int(valid.sum()),trusted_anchors=len(take)))
    arrays=dict(anchor_points=np.array(anchors),anchor_normals=np.array(normals),anchor_indices=np.array(indices),anchor_camera=np.array(camera_ids),
        anchor_other_votes=np.array(counts),landmark_uv=np.array(landmark_uv),landmark_indices=np.array(landmark_ids),landmark_camera=np.array(landmark_camera),validation_cameras=validation)
    np.savez_compressed(OUT/'observations.npz',**arrays)
    save(OUT/'observations.json',dict(cameras=rows,diagnostics=diagnostics,depth_receipt=receipt,observations_sha256=sha(OUT/'observations.npz'),
        protocol_sha256=sha(OUT/'protocol.json'),source_input_sha256=sha(SOURCE/FRAME/'input.json'),fit_observed_support_excludes_validation=True))
    print('Trusted anchors',len(anchors),'train',int((~validation[arrays['anchor_camera']]).sum()),'validation',int(validation[arrays['anchor_camera']].sum()),flush=True)

def fit_once(arm,train_selector=None):
    request=read(OUT/'protocol.json');data=np.load(OUT/'canonical.npz');v=data['vertices'];b=data['basis'];obs=np.load(OUT/'observations.npz')
    spec=read(OUT/'observations.json');rows=spec['cameras'];validation=obs['validation_cameras'];fitcams=~validation
    if train_selector is not None:fitcams&=train_selector
    amask=fitcams[obs['anchor_camera']];lmask=fitcams[obs['landmark_camera']]
    ai=obs['anchor_indices'][amask];ap=obs['anchor_points'][amask];an=obs['anchor_normals'][amask]
    li=obs['landmark_indices'][lmask];lc=obs['landmark_camera'][lmask];uv=obs['landmark_uv'][lmask]
    unique=np.unique(ai);target=np.array([np.median(ap[ai==i],axis=0) for i in unique]);s,r,t=similarity(v[unique],target)
    initial=np.r_[Rotation.from_matrix(r).as_rotvec(),t,np.log(s)]
    if arm=='regularized8':initial=np.r_[initial,np.zeros(24)]
    matrices=projection_matrices(rows);width=np.ptp(v[:,0]);scale_weight=np.sqrt(len(ap)/(2*len(li)));reg_weight=np.sqrt(len(ap)/24)
    def residual(p):
        world=deform(p,v,b);hom=np.column_stack((world[li],np.ones(len(li))));q=np.einsum('nij,nj->ni',matrices[lc],hom)
        pixels=q[:,:2]/q[:,2,None];land=(pixels-uv).ravel()/3*scale_weight
        surface=np.sum((world[ai]-ap)*an,axis=1)/.001
        reg=p[7:]/(.015*width)*reg_weight if len(p)>7 else np.empty(0)
        return np.r_[land,surface,reg]
    lower=np.full(len(initial),-np.inf);upper=np.full(len(initial),np.inf);lower[6]=np.log(s*.7);upper[6]=np.log(s*1.3)
    if len(initial)>7:lower[7:]=-.08*width;upper[7:]=.08*width
    started=time.monotonic();fit=least_squares(residual,initial,bounds=(lower,upper),loss='soft_l1',f_scale=1.,max_nfev=150,x_scale='jac')
    world=deform(fit.x,v,b);plain=deform(np.r_[fit.x[:7],np.zeros(24)],v,b)
    allworld=world[obs['landmark_indices']];m=matrices[obs['landmark_camera']];q=np.einsum('nij,nj->ni',m,np.column_stack((allworld,np.ones(len(allworld)))))
    errors=np.linalg.norm(q[:,:2]/q[:,2,None]-obs['landmark_uv'],axis=1)
    surface=np.sum((world[obs['anchor_indices']]-obs['anchor_points'])*obs['anchor_normals'],axis=1)
    lmval=validation[obs['landmark_camera']];aval=validation[obs['anchor_camera']]
    stats=dict(arm=arm,success=bool(fit.success),evaluations=fit.nfev,elapsed_seconds=time.monotonic()-started,
        train_landmark=quantiles(errors[~lmval]),validation_landmark=quantiles(errors[lmval]),
        train_point_plane=quantiles(np.abs(surface[~aval])),validation_point_plane=quantiles(np.abs(surface[aval])),
        maximum_deformation_fraction_width=float(np.linalg.norm(world-plain,axis=1).max()/(np.exp(fit.x[6])*width)),
        fit_camera_names=[r['physical_camera'] for r,k in zip(rows,fitcams) if k],model_z_used=False)
    return world,fit.x,stats,dict(landmark_errors=errors,point_plane_errors=surface)

def fit():
    import open3d as o3d
    canonical=np.load(OUT/'canonical.npz');tri=canonical['triangles'];records=[]
    for arm in ['similarity','regularized8']:
        dest=OUT/arm;dest.mkdir(exist_ok=False);world,p,stats,errors=fit_once(arm)
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(world),o3d.utility.Vector3iVector(tri));mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(dest/'prior_only.ply'),mesh)
        np.savez_compressed(dest/'fit.npz',vertices=world,parameters=p,**errors)
        save(dest/'result.json',dict(stats,mesh_sha256=sha(dest/'prior_only.ply'),fit_sha256=sha(dest/'fit.npz'),
            protocol_sha256=sha(OUT/'protocol.json'),original_mesh_changed=False,canonical_topology_only=True))
        records.append(stats);print(stats,flush=True)
    # Independent fitting-camera halves quantify instability, never use validation views.
    rows=read(OUT/'observations.json')['cameras'];worlds=[];splitstats=[]
    for parity in [0,1]:
        selector=np.arange(len(rows))%2==parity;world,p,stats,_=fit_once('regularized8',selector);worlds.append(world);splitstats.append(stats)
    disagreement=np.linalg.norm(worlds[0]-worlds[1],axis=1);np.savez_compressed(OUT/'split_fit.npz',worlds=np.array(worlds),disagreement=disagreement)
    req=read(OUT/'protocol.json');regularized=records[-1]
    gate=regularized['validation_point_plane']['p90']<=.002 and regularized['validation_landmark']['median']<=4 and regularized['validation_landmark']['p90']<=8 \
        and regularized['maximum_deformation_fraction_width']<=.15 and np.percentile(disagreement[JAW+CHEEK],90)<=.0015
    save(OUT/'fit_summary.json',dict(arms=records,split_fit=splitstats,lower_face_split_disagreement=quantiles(disagreement[JAW+CHEEK]),
        numerical_gate_passed=bool(gate),original_geometry_changed=False,visual_review_required=True,split_npz_sha256=sha(OUT/'split_fit.npz')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['assets','stage','fit']);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();OUT=a.output;globals()[a.command]()
