"""Hash/replay audit of retained MHR fits and train-only geometric support."""
from pathlib import Path
import numpy as np
from study_mhr_local_head_prior import OUT,RGB,ASSET,FRAME
from study_multiview_face_prior import read,save,sha
from fit_mhr_local_head_prior import torch_rotation
from triangulate_face_prior import projection_matrices

def main():
    import torch,open3d as o3d
    from study_confidence_depth_prior import load_real,support
    from diffusion_mesh_repair import scene_for
    from joint_temporal_texture import HELD_CAMERAS
    torch.set_num_threads(2);checked=[]
    def check(path,digest):assert sha(path)==digest,str(path);checked.append(str(path))
    proto=read(OUT/'protocol.json');meta=read(OUT/'anchors.json');obs=np.load(OUT/'anchors.npz');initial=np.load(OUT/'initial.npz');fitrequest=read(OUT/'fit_request.json')
    check(OUT/'initial.npz',proto['initial_sha256']);check(OUT/'anchors.npz',meta['anchors_sha256']);check(OUT/'protocol.json',meta['protocol_sha256']);check(OUT/'protocol.json',fitrequest['protocol_sha256'])
    check(proto['model_path'],proto['model_sha256']);check(proto['semantic_path'],proto['semantic_sha256']);check(proto['original_mesh'],proto['original_mesh_sha256'])
    check(RGB/FRAME/'input.json',proto['rgb_input_sha256']);check(RGB/'inference.json',proto['inference_sha256']);check(OUT/'semantics/complete.json',meta['semantics_receipt_sha256'])
    check(Path(__file__).with_name('study_mhr_local_head_prior.py'),proto['script_sha256']);check(Path(__file__).with_name('fit_mhr_local_head_prior.py'),fitrequest['script_sha256'])
    for item in read(RGB/FRAME/'input.json')['inputs']:check(item['path'],item['sha256'])
    for item in read(OUT/'semantics/complete.json')['records']:check(OUT/'semantics'/(item['camera']+'.npz'),item['output_sha256'])
    rows,depths,receipt=load_real(Path('/mnt/data/dec5_jaw_measured_depth/analysis'),FRAME);assert receipt==meta['depth_receipt'];assert not ({r['physical_camera'] for r in rows}&HELD_CAMERAS)
    np.testing.assert_allclose(projection_matrices(rows),projection_matrices(meta['cameras']),atol=1e-9)
    val=np.array([any(r['physical_camera'].startswith(p) for p in proto['validation_prefixes']) for r in rows]);np.testing.assert_array_equal(val,obs['validation']);assert val.sum()==8
    fitrows=[r for r,k in zip(rows,val) if not k];fitdepths=[d for d,k in zip(depths,val) if not k];votes=0
    for ci,row in enumerate(rows):
        pick=obs['camera']==ci;other,_=support(obs['points'][pick],row,fitrows,fitdepths);np.testing.assert_array_equal(other,obs['other_votes'][pick]);assert (other>=3).all();votes+=int(pick.sum())
    model=torch.jit.load(str(ASSET/'mhr_model.pt'),map_location='cpu').eval();tri=initial['triangles'];subtri=tri[(initial['neutral'][tri,1]>140).all(1)];residuals=0
    for arm in ['similarity','head20','head20_neck6']:
        result=read(OUT/arm/'result.json');check(OUT/arm/'fit.npz',result['fit_sha256']);check(OUT/arm/'prior_only.ply',result['mesh_sha256']);f=np.load(OUT/arm/'fit.npz')
        if arm=='head20_neck6':
            p=read(OUT/arm/'protocol.json');check(OUT/arm/'protocol.json',result['protocol_sha256']);check(Path(__file__).with_name('fit_mhr_named_articulation.py'),p['script_sha256']);check('/mnt/data/dec5_mhr_articulation_mapping/mapping.json',p['mapping_sha256'])
        identity=torch.zeros(1,45);identity[0,20:40]=torch.tensor(f['head']);mp=torch.zeros(1,204)
        if 'articulation' in f:mp[0,24:30]=torch.tensor(f['articulation'])
        with torch.no_grad():
            v,_=model(identity,mp,torch.zeros(1,72));rotation=torch.tensor(initial['rotation'])@torch_rotation(torch.tensor(f['pose'][:3]));world=float(initial['scale'])*np.exp(f['pose'][6])*v[0].double()@rotation.T+torch.tensor(initial['translation'])+.001*torch.tensor(f['pose'][3:6])
        np.testing.assert_allclose(world.numpy(),f['vertices'],atol=1e-10);scene=scene_for(f['vertices'],subtri);hit=scene.compute_closest_points(o3d.core.Tensor(obs['points'].astype(np.float32)));cp=hit['points'].numpy();delta=cp-obs['points'];plane=np.sum(delta*obs['normals'],axis=1);distance=np.linalg.norm(delta,axis=1)
        uv=hit['primitive_uvs'].numpy();associations=dict(triangles=subtri[hit['primitive_ids'].numpy()],barycentric=np.column_stack((1-uv.sum(1),uv)),prior_closest_points=cp,observed_points=obs['points'],observed_normals=obs['normals'],camera=obs['camera'],validation=val[obs['camera']],distance=distance)
        association_path=OUT/arm/'final_associations.npz'
        if association_path.exists():
            previous=np.load(association_path)
            for key,value in associations.items():np.testing.assert_array_equal(previous[key],value)
        else:np.savez_compressed(association_path,**associations)
        np.testing.assert_allclose(plane,f['plane'],atol=1e-10);np.testing.assert_allclose(distance,f['distance'],atol=1e-10)
        lm=(f['vertices'][initial['landmark_triangles']]*initial['landmark_bary'][:,:,None]).sum(1)[obs['landmark_indices']];q=np.einsum('nij,nj->ni',projection_matrices(rows)[obs['landmark_camera']],np.column_stack((lm,np.ones(len(lm)))));error=np.linalg.norm(q[:,:2]/q[:,2,None]-obs['landmark_uv'],axis=1);np.testing.assert_allclose(error,f['landmark_error'],atol=1e-9);residuals+=len(plane)*2+len(error)
    for folder in ['review_initial','review_similarity_head20','review_head20_neck6']:
        r=read(OUT/folder/'manifest.json');check(Path(__file__).with_name('review_mhr_local_head_prior.py'),r['script_sha256'])
        for item in r['files']:check(item['path'],item['sha256'])
    for folder in ['probe_similarity_head20','probe_head20_neck6']:
        r=read(OUT/folder/'result.json');check(Path(__file__).with_name('probe_mhr_local_head_prior.py'),r['script_sha256'])
        for item in r['files']:check(item['path'],item['sha256'])
        assert all(a['target_hits_passing_locality']==0 for a in r['arms'])
    save(OUT/'audit.json',dict(status='passed',checked_bindings=len(checked),checked_paths=checked,geometric_depth_maps_hash_verified=len(receipt['depth_sha256']),support_counts_replayed=votes,fit_residuals_replayed=residuals,exact_model_forward_replays=3,
        heldout_used=False,original_geometry_changed=False,production_promoted=False,original_mesh_sha256=sha(proto['original_mesh']),script_sha256=sha(__file__),
        excluded_validation_camera_names=[r['physical_camera'] for r,k in zip(rows,val) if k],
        root_artifact_hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='audit.json' and p.suffix!='.log'}))
    print('passed',len(checked),'bindings;',votes,'support counts;',residuals,'residuals')

if __name__=='__main__':main()
