"""Two frozen MHR controls: similarity and regularized head20, CPU autograd."""
import time
import numpy as np
from study_mhr_local_head_prior import OUT,ASSET
from study_multiview_face_prior import read,save,sha
from triangulate_face_prior import projection_matrices,quantiles

def torch_rotation(w):
    import torch
    zero=w[0]*0
    skew=torch.stack((zero,-w[2],w[1],w[2],zero,-w[0],-w[1],w[0],zero)).reshape(3,3)
    return torch.matrix_exp(skew)

def run():
    import torch,open3d as o3d
    from diffusion_mesh_repair import scene_for
    torch.set_num_threads(2);proto=read(OUT/'protocol.json');initial=np.load(OUT/'initial.npz');obs=np.load(OUT/'anchors.npz');meta=read(OUT/'anchors.json');rows=meta['cameras']
    if (OUT/'fit_request.json').exists():raise ValueError('Preserve fit attempts')
    save(OUT/'fit_request.json',dict(protocol_sha256=sha(OUT/'protocol.json'),anchors_sha256=sha(OUT/'anchors.npz'),script_sha256=sha(__file__),
        objective='Robust observed point-plane plus weak point distance; equal face/neck weighting; interior model landmarks; strong unit-Gaussian head prior sigma .5',
        model_correspondence='iterative nearest-triangle barycentric; not measured landmark identity',outer=4,inner=25,maximum_head_coefficient=1.5,
        articulated_pose_zero=True,expression_zero=True,body_identity_zero=True,heldout_used=False))
    model=torch.jit.load(str(ASSET/'mhr_model.pt'),map_location='cpu').eval();tri=initial['triangles'];subtri=tri[(initial['neutral'][tri,1]>140).all(1)]
    tensor=lambda x:torch.tensor(x,dtype=torch.float64);matrix=tensor(projection_matrices(rows));rotation=tensor(initial['rotation']);translation=tensor(initial['translation']);scale=float(initial['scale'])
    point=tensor(obs['points']);normal=tensor(obs['normals']);fitmask=~obs['validation'][obs['camera']];neck=obs['neck'];lmfit=~obs['validation'][obs['landmark_camera']]
    lmtri=torch.tensor(initial['landmark_triangles'],dtype=torch.long);lmbary=tensor(initial['landmark_bary']);lmc=torch.tensor(obs['landmark_camera'][lmfit]);lmi=torch.tensor(obs['landmark_indices'][lmfit]);lmuv=tensor(obs['landmark_uv'][lmfit])
    results=[]
    for arm in ['similarity','head20']:
        dest=OUT/arm;dest.mkdir(exist_ok=False);pose=torch.nn.Parameter(torch.zeros(7,dtype=torch.float64));head=torch.nn.Parameter(torch.zeros(20,dtype=torch.float64));parameters=[pose]+([head] if arm=='head20' else [])
        def world():
            beta=1.5*torch.tanh(head) if arm=='head20' else head*0
            identity=torch.cat((torch.zeros(20,dtype=torch.float64),beta,torch.zeros(5,dtype=torch.float64))).to(torch.float32)[None]
            vertices,_=model(identity,torch.zeros(1,204),torch.zeros(1,72));vertices=vertices[0].to(torch.float64)
            wr=rotation@torch_rotation(pose[:3]);return scale*torch.exp(pose[6])*vertices@wr.T+translation+.001*pose[3:6],beta
        evaluations=0;started=time.monotonic();history=[]
        for outer in range(4):
            with torch.no_grad():current,_=world();v=current.numpy()
            scene=scene_for(v,subtri);closest=scene.compute_closest_points(o3d.core.Tensor(obs['points'].astype(np.float32)));ids=subtri[closest['primitive_ids'].numpy()];uv=closest['primitive_uvs'].numpy();bary=np.column_stack((1-uv.sum(1),uv));distance=np.linalg.norm(closest['points'].numpy()-obs['points'],axis=1)
            use=fitmask&(distance<=.006);assert (use&neck).sum()>=50 and (use&~neck).sum()>=50
            ids=torch.tensor(ids[use],dtype=torch.long);bary=tensor(bary[use]);pp=point[use];nn=normal[use];groups=[torch.tensor(neck[use]),torch.tensor(~neck[use])]
            optimizer=torch.optim.LBFGS(parameters,lr=.5,max_iter=25,line_search_fn='strong_wolfe',tolerance_grad=1e-7,tolerance_change=1e-9)
            def robust(x):return 2*(torch.sqrt(1+x*x)-1)
            def closure():
                nonlocal evaluations
                optimizer.zero_grad();v,beta=world();pred=(v[ids]*bary[:,:,None]).sum(1);delta=pred-pp;plane=(delta*nn).sum(1)/.001
                value=sum(robust(plane[g]).mean()*.5+robust(delta[g]/.004).mean()*.05 for g in groups)
                lp=(v[lmtri]*lmbary[:,:,None]).sum(1)[lmi];q=torch.einsum('nij,nj->ni',matrix[lmc],torch.cat((lp,torch.ones((len(lp),1))),dim=1));pixels=q[:,:2]/q[:,2,None]
                value=value+robust((pixels-lmuv)/4).mean()*.5+(beta/.5).square().mean()*.5
                value.backward();evaluations+=1;return value
            optimizer.step(closure);history.append(dict(outer=outer,face_anchors=int((use&~neck).sum()),neck_anchors=int((use&neck).sum()),evaluations=evaluations));print(arm,history[-1],flush=True)
        with torch.no_grad():current,beta=world();v=current.numpy()
        scene=scene_for(v,subtri);closest=scene.compute_closest_points(o3d.core.Tensor(obs['points'].astype(np.float32)));delta=closest['points'].numpy()-obs['points'];plane=np.sum(delta*obs['normals'],axis=1);distance=np.linalg.norm(delta,axis=1)
        lm=(v[initial['landmark_triangles']]*initial['landmark_bary'][:,:,None]).sum(1)[obs['landmark_indices']];q=np.einsum('nij,nj->ni',projection_matrices(rows)[obs['landmark_camera']],np.column_stack((lm,np.ones(len(lm)))));errors=np.linalg.norm(q[:,:2]/q[:,2,None]-obs['landmark_uv'],axis=1)
        metrics={}
        for split,selection in [('train',fitmask),('validation',~fitmask)]:
            for kind,region in [('face',~neck),('neck_underside',neck)]:metrics[split+'_'+kind]=dict(point_plane=quantiles(abs(plane[selection&region])),surface_distance=quantiles(distance[selection&region]))
        metrics['validation_landmark']=quantiles(errors[~lmfit]);metrics['train_landmark']=quantiles(errors[lmfit])
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(tri));mesh.compute_vertex_normals();o3d.io.write_triangle_mesh(str(dest/'prior_only.ply'),mesh)
        np.savez_compressed(dest/'fit.npz',vertices=v,pose=pose.detach().numpy(),head=beta.numpy(),plane=plane,distance=distance,landmark_error=errors)
        result=dict(arm=arm,metrics=metrics,seconds=time.monotonic()-started,evaluations=evaluations,history=history,maximum_head_coefficient=float(abs(beta.numpy()).max()),fit_sha256=sha(dest/'fit.npz'),mesh_sha256=sha(dest/'prior_only.ply'),fit_request_sha256=sha(OUT/'fit_request.json'),original_geometry_changed=False)
        save(dest/'result.json',result);results.append(result);print(arm,metrics,flush=True)
    save(OUT/'fit_summary.json',dict(arms=results,original_geometry_changed=False,needs_native_and_boundary_gate=True))

if __name__=='__main__':run()
