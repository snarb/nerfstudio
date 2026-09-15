"""Smooth train-depth conformance of an anatomical prior; original mesh untouched.

Fit three predeclared regularization strengths, reserving the prior experiment's
eight validation cameras. This creates prior-only controls, never an accepted
replacement or a mesh-hole patch by itself.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr
from build_train_hair_semantics import read,sha,write

ROOT=Path('/mnt/data/dec5_mhr_measured_conformance')
PARENT=Path('/mnt/data/dec5_mhr_local_head_prior')
ARMS={'smooth025':.25,'smooth100':1.,'smooth400':4.}


def uniform_laplacian(triangles,n):
    edges=np.unique(np.sort(triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)
    adjacency=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    degree=np.asarray(adjacency.sum(1)).ravel()
    return sparse.eye(n,format='csr')-sparse.diags(1/np.maximum(degree,1))@adjacency


def barycentric_matrix(triangles,bary,n):
    if triangles.shape!=bary.shape or triangles.ndim!=2 or triangles.shape[1]!=3:
        raise ValueError('Expected matching triangle and barycentric arrays')
    if not np.isfinite(bary).all() or not np.allclose(bary.sum(1),1) or (bary<-.00001).any():
        raise ValueError('Invalid barycentric weights')
    return sparse.coo_matrix((bary.ravel(),(np.repeat(np.arange(len(triangles)),3),triangles.ravel())),shape=(len(triangles),n)).tocsr()


def init():
    ROOT.mkdir(exist_ok=False)
    parent=read(PARENT/'protocol.json')
    files=[PARENT/'head20_neck6/fit.npz',PARENT/'head20_neck6/result.json',PARENT/'initial.npz',PARENT/'anchors.npz',PARENT/'anchors.json',PARENT/'protocol.json']
    recipe=dict(arms=ARMS,outer_iterations=4,association_max_distance=.006,normal_cosine_minimum=.25,
        data_sigma=.001,laplacian_sigma=.0005,magnitude_sigma=.006,maximum_lsmr_iterations=600,
        deform_neutral_y_min=135,lsmr_atol=1e-8,lsmr_btol=1e-8,
        inherited_eight_camera_validation_split=True,target_or_eval_rgb_used=False,
        original_mesh_changed=False,prior_only=True,patch_admission_not_performed=True)
    write(ROOT/'protocol.json',dict(frame='001193',original_mesh=parent['original_mesh'],
        original_mesh_sha256=parent['original_mesh_sha256'],validation_prefixes=parent['validation_prefixes'],
        recipe=recipe,input_hashes={str(p):sha(p) for p in files},script_sha256=sha(__file__)))
    for name in ['initial.npz','anchors.npz','anchors.json']:(ROOT/name).symlink_to(PARENT/name)


def fit():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from triangulate_face_prior import quantiles
    protocol=read(ROOT/'protocol.json');recipe=protocol['recipe']
    assert protocol['script_sha256']==sha(__file__)
    for path,digest in protocol['input_hashes'].items():assert sha(path)==digest,path
    assert sha(protocol['original_mesh'])==protocol['original_mesh_sha256']
    initial=np.load(ROOT/'initial.npz');obs=np.load(ROOT/'anchors.npz')
    base=np.load(PARENT/'head20_neck6/fit.npz')['vertices'];tri=initial['triangles'];n=len(base)
    active=initial['neutral'][:,1]>135;active_ids=np.flatnonzero(active)
    head_tri=tri[(initial['neutral'][tri,1]>140).all(1)]
    fitmask=~obs['validation'][obs['camera']];points=obs['points'];normals=obs['normals'];neck=obs['neck']
    laplacian=uniform_laplacian(tri,n)[active_ids][:,active_ids]
    magnitude=sparse.eye(len(active_ids),format='csr')/(recipe['magnitude_sigma']*np.sqrt(len(active_ids)))
    records=[]
    for arm,strength in ARMS.items():
        destination=ROOT/arm;destination.mkdir(exist_ok=False)
        current=base.copy();history=[]
        for outer in range(recipe['outer_iterations']):
            mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(current),o3d.utility.Vector3iVector(head_tri));mesh.compute_triangle_normals()
            scene=scene_for(current,head_tri)
            nearest=scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))
            ids=nearest['primitive_ids'].numpy();uv=nearest['primitive_uvs'].numpy();bary=np.c_[1-uv.sum(1),uv]
            closest=nearest['points'].numpy();distance=np.linalg.norm(closest-points,axis=1)
            dot=np.sum(np.asarray(mesh.triangle_normals)[ids]*normals,axis=1)
            use=fitmask&(distance<=recipe['association_max_distance'])&(dot>=recipe['normal_cosine_minimum'])
            assert (use&neck).sum()>=50 and (use&~neck).sum()>=50,'Insufficient normal-consistent train support'
            association=barycentric_matrix(head_tri[ids[use]],bary[use],n)
            robust_weight=1/np.sqrt(1+(distance[use]/recipe['data_sigma'])**2)
            # Equal total weight per pre-existing face/neck group, never validation data.
            group=np.where(neck[use],int((use&neck).sum()),int((use&~neck).sum()))
            weights=np.sqrt(robust_weight/(2*group))/recipe['data_sigma']
            data=sparse.diags(weights)@association[:,active_ids]
            target=(points[use]-association@base)*weights[:,None]
            smooth=laplacian*np.sqrt(strength)/(recipe['laplacian_sigma']*np.sqrt(len(active_ids)))
            system=sparse.vstack((data,smooth,magnitude),format='csr')
            rhs=np.vstack((target,np.zeros((2*len(active_ids),3))))
            solutions=[lsmr(system,rhs[:,axis],atol=recipe['lsmr_atol'],btol=recipe['lsmr_btol'],
                maxiter=recipe['maximum_lsmr_iterations']) for axis in range(3)]
            displacement=np.column_stack([value[0] for value in solutions])
            assert np.isfinite(displacement).all()
            current=base.copy();current[active_ids]+=displacement
            history.append(dict(outer=outer,train_face=int((use&~neck).sum()),train_neck=int((use&neck).sum()),
                lsmr_istop=[int(s[1]) for s in solutions],lsmr_iterations=[int(s[2]) for s in solutions],
                maximum_displacement=float(np.linalg.norm(displacement,axis=1).max())))
            print(arm,history[-1],flush=True)
        scene=scene_for(current,head_tri)
        cp=scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))['points'].numpy()
        delta=cp-points;plane=np.sum(delta*normals,axis=1);distance=np.linalg.norm(delta,axis=1)
        stats={}
        for split,selection in [('train',fitmask),('validation',~fitmask)]:
            for name,groupmask in [('face',~neck),('neck_candidate_group',neck)]:
                take=selection&groupmask
                stats[split+'_'+name]=dict(surface_distance=quantiles(distance[take]),point_plane=quantiles(abs(plane[take])))
        old_cross=np.cross(base[tri[:,1]]-base[tri[:,0]],base[tri[:,2]]-base[tri[:,0]])
        new_cross=np.cross(current[tri[:,1]]-current[tri[:,0]],current[tri[:,2]]-current[tri[:,0]])
        flipped=np.sum(old_cross*new_cross,axis=1)<=0
        np.testing.assert_array_equal(current[~active],base[~active])
        np.savez_compressed(destination/'fit.npz',vertices=current,plane=plane,distance=distance,
            displacement=current-base,normal_reversed_triangles=flipped)
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(current),o3d.utility.Vector3iVector(tri));mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(destination/'prior_only.ply'),mesh)
        result=dict(arm=arm,stats=stats,history=history,normal_reversed_triangles=int(flipped.sum()),
            fit_sha256=sha(destination/'fit.npz'),mesh_sha256=sha(destination/'prior_only.ply'),
            protocol_sha256=sha(ROOT/'protocol.json'),prior_only=True,original_geometry_changed=False,
            visual_status='pending',local_patch_or_production_approval=False)
        write(destination/'result.json',result);records.append(result)
        print(arm,'validation',stats['validation_neck_candidate_group'],'flipped',int(flipped.sum()),flush=True)
    write(ROOT/'fit_summary.json',dict(arms=records,original_geometry_changed=False,requires_native_boundary_and_free_space_checks=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['init','fit'])
    init() if parser.parse_args().action=='init' else fit()
