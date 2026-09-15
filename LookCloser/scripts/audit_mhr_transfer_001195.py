"""Per-time measurement/forward replay and exact fixed-objective continuation audit."""
from pathlib import Path
import argparse
import numpy as np
from transfer_mhr_001195 import ROOT,HEAD,CONFORM,TEN,FINAL,FRAME,configure,verify
from study_multiview_face_prior import read,save,sha


def head():
    import torch,open3d as o3d
    from study_confidence_depth_prior import load_real,support,unproject
    from fit_mhr_local_head_prior import torch_rotation
    from triangulate_face_prior import projection_matrices
    from admit_mhr_local_patch_depth import Scene2
    torch.set_num_threads(2);q=read(HEAD/'protocol.json');obs=np.load(HEAD/'anchors.npz');initial=np.load(HEAD/'initial.npz')
    rows,depths,receipt=load_real(Path('/mnt/data/dec5_jaw_measured_depth/analysis'),FRAME)
    assert receipt==read(HEAD/'anchors.json')['depth_receipt']
    val=np.array([any(r['physical_camera'].startswith(p) for p in q['validation_prefixes']) for r in rows]);np.testing.assert_array_equal(val,obs['validation'])
    assert val.sum()==8
    fr=[r for r,v in zip(rows,val) if not v];fd=[d for d,v in zip(depths,val) if not v]
    old=o3d.io.read_triangle_mesh(q['original_mesh']);oldscene=Scene2(np.asarray(old.vertices),np.asarray(old.triangles))
    for ci,row in enumerate(rows):
        take=obs['camera']==ci;xy=obs['native_xy'][take];d=depths[ci][xy[:,1],xy[:,0]];p=unproject(row,xy[:,0],xy[:,1],d)
        np.testing.assert_array_equal(p,obs['points'][take]);votes,_=support(p,row,fr,fd);np.testing.assert_array_equal(votes,obs['other_votes'][take]);assert (votes>=3).all()
        center=np.asarray(row['transform_matrix'])[:3,3];direction=unproject(row,xy[:,0],xy[:,1],np.ones(len(xy)))-center
        hit=oldscene.cast_rays(o3d.core.Tensor(np.c_[np.broadcast_to(center,direction.shape),direction].astype(np.float32)))['t_hit'].numpy()
        assert np.isfinite(hit).all() and (abs(hit-d)<=.001).all()
    model=torch.jit.load(q['model_path'],map_location='cpu').eval();assert sha(q['model_path'])==q['model_sha256']
    tri=initial['triangles'];headtri=tri[(initial['neutral'][tri,1]>140).all(1)]
    counts={}
    for arm in ['similarity','head20','head20_neck6']:
        a=np.load(HEAD/arm/'fit.npz');beta=torch.zeros(1,45);beta[0,20:40]=torch.tensor(a['head']);pose=torch.zeros(1,204)
        if 'articulation' in a:pose[0,24:30]=torch.tensor(a['articulation'])
        with torch.no_grad():
            vertices,_=model(beta,pose,torch.zeros(1,72));r=torch.tensor(initial['rotation'])@torch_rotation(torch.tensor(a['pose'][:3]))
            world=float(initial['scale'])*np.exp(a['pose'][6])*vertices[0].double()@r.T+torch.tensor(initial['translation'])+.001*torch.tensor(a['pose'][3:6])
        np.testing.assert_allclose(world.numpy(),a['vertices'],atol=1e-10,rtol=0)
        cp=Scene2(a['vertices'],headtri).compute_closest_points(o3d.core.Tensor(obs['points'].astype(np.float32)));delta=cp['points'].numpy()-obs['points'];dist=np.linalg.norm(delta,axis=1)
        np.testing.assert_allclose(dist,a['distance'],atol=1e-10,rtol=0);np.testing.assert_allclose(np.sum(delta*obs['normals'],axis=1),a['plane'],atol=1e-10,rtol=0)
        uv=cp['primitive_uvs'].numpy();np.savez_compressed(HEAD/arm/'final_associations.npz',triangles=headtri[cp['primitive_ids'].numpy()],barycentric=np.c_[1-uv.sum(1),uv],distance=dist,camera=obs['camera'],validation=val[obs['camera']])
        lm=(a['vertices'][initial['landmark_triangles']]*initial['landmark_bary'][:,:,None]).sum(1)[obs['landmark_indices']]
        proj=np.einsum('nij,nj->ni',projection_matrices(rows)[obs['landmark_camera']],np.c_[lm,np.ones(len(lm))]);error=np.linalg.norm(proj[:,:2]/proj[:,2,None]-obs['landmark_uv'],axis=1)
        np.testing.assert_allclose(error,a['landmark_error'],atol=1e-9,rtol=0)
        use=(dist<=.006)&~val[obs['camera']];counts[arm]=dict(final_distance_associated_face=int((use&~obs['neck']).sum()),final_distance_associated_neck=int((use&obs['neck']).sum()))
    save(ROOT/'head_audit.json',dict(status='passed',script_sha256=sha(__file__),observed_points_replayed=len(obs['points']),
        exact_model_forward_replays=3,depth_hashes=receipt['depth_sha256'],reserved_views=8,counts=counts,
        input_hashes={str(HEAD/'protocol.json'):sha(HEAD/'protocol.json'),str(HEAD/'anchors.npz'):sha(HEAD/'anchors.npz')}))
    print('head audit passed',counts,flush=True)


def continuation():
    from scipy.ndimage import distance_transform_edt
    import fit_mhr_silhouette_conformance as fit
    import continue_mhr_silhouette_convergence as cont
    from triangulate_face_prior import quantiles
    fit.ROOT=FINAL;a=np.load(FINAL/'fit.npz');q=read(FINAL/'protocol.json');result=read(FINAL/'result.json');state=read(FINAL/'continuation_result.json')
    assert q['frame']==FRAME and q['recipe']==dict(read(TEN/'protocol.json')['recipe'],outer_iterations=100)
    _,rows,masks,names,evidence,val=fit.prepare();assert evidence==q['evidence'];obs=np.load(CONFORM/'anchors.npz');train=~val[obs['camera']]
    sdfs=[]
    for row in rows:
        m=masks[names.index(row['physical_camera'])].astype(bool);sdfs.append((distance_transform_edt(~m)-distance_transform_edt(m)).astype(np.float32))
    cont.ROOT=FINAL/'audit_replay';cont.ROOT.mkdir(exist_ok=False);cont.FROZEN_OPTIMIZER=fit.optimize
    worker,replayed=cont.make_optimizer();v,history=worker(a['baseline'],a['triangles'],a['neutral'],obs['points'][train],obs['normals'][train],obs['neck'][train],[r for r,x in zip(rows,val) if not x],[s for s,x in zip(sdfs,val) if not x])
    np.testing.assert_array_equal(v,a['vertices']);assert history==result['history']
    for key in ['records','stop_reason','consecutive_small','consecutive_increases','first10_exact']:assert replayed[key]==state[key],key
    for i in range(1,len(history)+1):np.testing.assert_array_equal(np.load(FINAL/'iterates'/f'{i:03d}.npz')['vertices'],np.load(cont.ROOT/'iterates'/f'{i:03d}.npz')['vertices'])
    grids=[]
    for vertices in [a['baseline'],v]:
        values=np.full((62,int(a['active'].sum())),np.nan)
        for ci,(ids,s,_) in enumerate(fit.silhouette_samples(vertices[a['active']],rows,sdfs)):values[ci,ids]=s
        grids.append(values)
    b,f=grids;common=np.isfinite(b)&np.isfinite(f);metrics={}
    for name,selection in [('fit',~val),('reserved',val)]:
        take=common&selection[:,None];before=np.maximum(b[take]-2,0);after=np.maximum(f[take]-2,0)
        metrics[name]=dict(fixed_samples=len(before),before=quantiles(before),after=quantiles(after),before_mean=float(before.mean()),after_mean=float(after.mean()),before_outside=int((before>0).sum()),after_outside=int((after>0).sum()))
    np.savez_compressed(FINAL/'audit_evidence.npz',before_sdf=b,after_sdf=f,common=common,validation=val)
    save(FINAL/'audit.json',dict(status='passed',exact_all_iterates_replayed=len(history),first10_exact=True,metrics=metrics,
        script_sha256=sha(__file__),input_hashes={str(FINAL/'protocol.json'):sha(FINAL/'protocol.json'),str(FINAL/'fit.npz'):sha(FINAL/'fit.npz')},production_accepted=False))
    print('continuation audit passed',len(history),metrics,flush=True)


def conformance():
    import conform_mhr_measured_surface as model
    replay=ROOT/'conformance_replay';model.ROOT=replay;original=model.write
    def write(path,value):
        if Path(path)==replay/'protocol.json':value=dict(value,frame=FRAME,transfer_config_sha256=sha(ROOT/'config.json'))
        original(path,value)
    model.write=write;model.init();model.fit()
    a=np.load(CONFORM/'smooth100/fit.npz');b=np.load(replay/'smooth100/fit.npz')
    for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    save(ROOT/'conformance_audit.json',dict(status='passed',all_arrays_exact=a.files,script_sha256=sha(__file__),
        input_hashes={str(CONFORM/'protocol.json'):sha(CONFORM/'protocol.json'),str(CONFORM/'smooth100/fit.npz'):sha(CONFORM/'smooth100/fit.npz')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['head','conformance','continuation']);a=p.parse_args();verify();configure();globals()[a.stage]()
