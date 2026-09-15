"""Explicit third MHR arm: named neck/head rotations, no guessed jaw/body DOFs."""
import time
import numpy as np
from study_mhr_local_head_prior import OUT,ASSET
from study_multiview_face_prior import read,save,sha
from fit_mhr_local_head_prior import torch_rotation
from triangulate_face_prior import projection_matrices,quantiles

def main():
    import torch,open3d as o3d
    from diffusion_mesh_repair import scene_for
    torch.set_num_threads(2);mapping_path='/mnt/data/dec5_mhr_articulation_mapping/mapping.json';mapping=read(mapping_path);model=torch.jit.load(str(ASSET/'mhr_model.pt'),map_location='cpu').eval()
    assert sha(ASSET/'mhr_model.pt')==mapping['model_sha256'];names=list(model.character_torch.parameter_transform.parameter_names)
    selected=['neck_twist','neck_lean','neck_bend','head_twist','head_lean','head_bend'];assert names[24:30]==selected
    dest=OUT/'head20_neck6';dest.mkdir(exist_ok=False);save(dest/'protocol.json',dict(parent_protocol_sha256=sha(OUT/'protocol.json'),parent_fit_sha256=sha(OUT/'head20/fit.npz'),mapping_sha256=sha(mapping_path),
        names=selected,indices=list(range(24,30)),limits=[[-.8,.8],[-.5,.5],[-.6,.5],[-.8,.8],[-.3,.3],[-.4,.4]],rotation_sigma_radians=.15,
        same_anchors=True,same_objective_except_rotation_prior=True,same_locality_gates=True,body_identity_fixed=True,
        pose_correctives_can_move_nonhead_prior_vertices=True,original_geometry_changed=False,heldout_used=False,script_sha256=sha(__file__)))
    initial=np.load(OUT/'initial.npz');obs=np.load(OUT/'anchors.npz');rows=read(OUT/'anchors.json')['cameras'];oldfit=np.load(OUT/'head20/fit.npz');tri=initial['triangles'];subtri=tri[(initial['neutral'][tri,1]>140).all(1)]
    tensor=lambda x:torch.tensor(x,dtype=torch.float64);rotation=tensor(initial['rotation']);translation=tensor(initial['translation']);scale=float(initial['scale']);matrix=tensor(projection_matrices(rows))
    pose=torch.nn.Parameter(tensor(oldfit['pose']));head=torch.nn.Parameter(tensor(np.arctanh(oldfit['head']/1.5)));art=torch.nn.Parameter(torch.zeros(6,dtype=torch.float64));limits=tensor([.8,.5,.5,.8,.3,.4])
    # Asymmetric neck-bend range retains zero initialization and official limits.
    def angles():return torch.where(art>=0,limits,torch.tensor([.8,.5,.6,.8,.3,.4],dtype=torch.float64))*torch.tanh(art)
    def world():
        beta=1.5*torch.tanh(head);identity=torch.cat((torch.zeros(20,dtype=torch.float64),beta,torch.zeros(5,dtype=torch.float64))).float()[None];a=angles();mp=torch.cat((torch.zeros(24,dtype=torch.float64),a,torch.zeros(174,dtype=torch.float64))).float()[None]
        v,_=model(identity,mp,torch.zeros(1,72));wr=rotation@torch_rotation(pose[:3]);return scale*torch.exp(pose[6])*v[0].double()@wr.T+translation+.001*pose[3:6],beta,a
    fitmask=~obs['validation'][obs['camera']];neck=obs['neck'];lmfit=~obs['validation'][obs['landmark_camera']];point=tensor(obs['points']);normal=tensor(obs['normals']);lmtri=torch.tensor(initial['landmark_triangles']);lmbary=tensor(initial['landmark_bary']);lmi=torch.tensor(obs['landmark_indices'][lmfit]);lmc=torch.tensor(obs['landmark_camera'][lmfit]);lmuv=tensor(obs['landmark_uv'][lmfit]);evaluations=0;history=[];started=time.monotonic()
    for outer in range(4):
        with torch.no_grad():v,_,_=world();v=v.numpy()
        scene=scene_for(v,subtri);cp=scene.compute_closest_points(o3d.core.Tensor(obs['points'].astype(np.float32)));uv=cp['primitive_uvs'].numpy();bary=np.column_stack((1-uv.sum(1),uv));ids=subtri[cp['primitive_ids'].numpy()];distance=np.linalg.norm(cp['points'].numpy()-obs['points'],axis=1);use=fitmask&(distance<=.006)
        ids=torch.tensor(ids[use]);bary=tensor(bary[use]);pp=point[use];nn=normal[use];groups=[torch.tensor(neck[use]),torch.tensor(~neck[use])];optimizer=torch.optim.LBFGS([pose,head,art],lr=.5,max_iter=25,line_search_fn='strong_wolfe',tolerance_grad=1e-7,tolerance_change=1e-9)
        def robust(x):return 2*(torch.sqrt(1+x*x)-1)
        def closure():
            nonlocal evaluations
            optimizer.zero_grad();v,beta,a=world();pred=(v[ids]*bary[:,:,None]).sum(1);delta=pred-pp;plane=(delta*nn).sum(1)/.001
            value=sum(robust(plane[g]).mean()*.5+robust(delta[g]/.004).mean()*.05 for g in groups)
            lp=(v[lmtri]*lmbary[:,:,None]).sum(1)[lmi];q=torch.einsum('nij,nj->ni',matrix[lmc],torch.cat((lp,torch.ones((len(lp),1))),dim=1));pixels=q[:,:2]/q[:,2,None]
            value=value+robust((pixels-lmuv)/4).mean()*.5+(beta/.5).square().mean()*.5+(a/.15).square().mean()*.5;value.backward();evaluations+=1;return value
        optimizer.step(closure);history.append(dict(outer=outer,face_anchors=int((use&~neck).sum()),neck_anchors=int((use&neck).sum()),evaluations=evaluations));print(history[-1],flush=True)
    with torch.no_grad():v,beta,a=world();v=v.numpy();beta=beta.numpy();a=a.numpy()
    scene=scene_for(v,subtri);cp=scene.compute_closest_points(o3d.core.Tensor(obs['points'].astype(np.float32)));delta=cp['points'].numpy()-obs['points'];plane=np.sum(delta*obs['normals'],axis=1);distance=np.linalg.norm(delta,axis=1)
    lm=(v[initial['landmark_triangles']]*initial['landmark_bary'][:,:,None]).sum(1)[obs['landmark_indices']];q=np.einsum('nij,nj->ni',projection_matrices(rows)[obs['landmark_camera']],np.column_stack((lm,np.ones(len(lm)))));error=np.linalg.norm(q[:,:2]/q[:,2,None]-obs['landmark_uv'],axis=1);metrics={}
    for split,mask in [('train',fitmask),('validation',~fitmask)]:
        for kind,region in [('face',~neck),('neck_underside',neck)]:metrics[split+'_'+kind]=dict(point_plane=quantiles(abs(plane[mask&region])),surface_distance=quantiles(distance[mask&region]))
    metrics['validation_landmark']=quantiles(error[~lmfit]);metrics['train_landmark']=quantiles(error[lmfit])
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(tri));mesh.compute_vertex_normals();o3d.io.write_triangle_mesh(str(dest/'prior_only.ply'),mesh)
    np.savez_compressed(dest/'fit.npz',vertices=v,pose=pose.detach().numpy(),head=beta,articulation=a,plane=plane,distance=distance,landmark_error=error)
    save(dest/'result.json',dict(arm='head20_neck6',metrics=metrics,seconds=time.monotonic()-started,evaluations=evaluations,history=history,named_angles=dict(zip(selected,a.tolist())),maximum_head_coefficient=float(abs(beta).max()),fit_sha256=sha(dest/'fit.npz'),mesh_sha256=sha(dest/'prior_only.ply'),protocol_sha256=sha(dest/'protocol.json'),original_geometry_changed=False))
    print(metrics,flush=True)

if __name__=='__main__':main()
