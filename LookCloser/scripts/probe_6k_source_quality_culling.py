"""Read-only centroid visibility versus quality culling at inspected seams."""
from pathlib import Path
import numpy as np
from joint_temporal_texture import read, sha, atomic_json, cameras, project
from temporal_texture_view_prior import angle_weights
from diagnose_6k_source_seams import ROOT, NEW


def main():
    import open3d as o3d
    import torch
    from bake_joint_temporal_mesh import camera_depth
    from admit_mhr_local_patch_depth import Scene2
    from native_texture_footprint import snap_centers, sample_native
    out=ROOT/'centroid_quality';out.mkdir(exist_ok=False)
    frame='001083';config=read(NEW/'request.json');parent=Path(config['parent'])
    request=read(parent/'request.json');record=next(r for r in request['inventory'] if r['frame_id']==frame)
    rows,_,_=cameras(frame);names=[r['physical_camera'] for r in rows]
    assert names==read(parent/'frames'/frame/'result.json')['source_cameras']
    assert sha(record['mesh'])==record['mesh_sha256']
    mesh=o3d.io.read_triangle_mesh(record['mesh']);v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles)
    labels=np.load(NEW/'frames'/frame/'face_source_labels.npy');selections={}
    for region in ['hair','lips']:
        a=np.load(ROOT/'mesh_label_attribution'/f'{frame}_{region}.npz')
        faces=np.unique(a['face_ids'][a['valid']]);selections[region]=faces[labels[faces]!=33]
    faces=np.unique(np.concatenate(list(selections.values())));tv=v[t[faces]];points=tv.mean(1)
    normals=np.cross(tv[:,1]-tv[:,0],tv[:,2]-tv[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None].clip(1e-12)
    uv,z=project(points,rows);uv=snap_centers(torch.from_numpy(uv)).numpy()
    maskspec=record['source_masks'];mr=Path(maskspec['root'])
    for name,key in [('masks.npz','masks_sha256'),('cameras.json','cameras_sha256'),('complete.json','complete_sha256')]:assert sha(mr/name)==maskspec[key]
    masks=dict(zip(read(mr/'cameras.json'),np.load(mr/'masks.npz')['masks']))
    scene=Scene2(v,t);valid=np.zeros((len(rows),len(points)),bool)
    for ci,row in enumerate(rows):
        d,_,_=camera_depth(scene,row);d=np.where(np.isfinite(d)&(masks[row['physical_camera']]>0),d,0)
        sampled=sample_native(torch.from_numpy(d[None,None]),torch.from_numpy(uv[ci][None,None]))[0,0,0].numpy()
        valid[ci]=(z[ci]>0)&(sampled>0)&(np.abs(sampled-z[ci])<.0015*z[ci])
        valid[ci]&=(uv[ci,:,0]>2)&(uv[ci,:,0]<1917)&(uv[ci,:,1]>2)&(uv[ci,:,1]<1077)
    centers=np.array([r['transform_matrix'] for r in rows],np.float32)[:,:3,3]
    direction=centers[:,None]-points;length=np.linalg.norm(direction,axis=-1);direction/=length[...,None]
    quality=np.abs((direction*normals).sum(-1))**2/length.clip(.01)**2*valid
    quality*=angle_weights(rows,record['camera'],4.)[0][:,None]
    ratio=quality/quality.max(0).clip(1e-12);retained=ratio>=.12
    records=[]
    for region,subset in selections.items():
        take=np.isin(faces,subset)
        records.append(dict(region=region,other_source_faces=int(take.sum()),
            dominant_camera=names[33],dominant_geometrically_visible=int(valid[33,take].sum()),
            dominant_quality_culled=int((valid[33,take]&~retained[33,take]).sum()),
            dominant_retained=int((valid[33,take]&retained[33,take]).sum())))
    np.savez_compressed(out/'evidence.npz',face_ids=faces,centroids=points,uv=uv,valid=valid,
        quality=quality,quality_ratio=ratio,retained=retained,face_labels=labels[faces])
    atomic_json(out/'result.json',dict(records=records,source_cameras=names,script_sha256=sha(__file__),
        input_hashes={str(parent/'request.json'):sha(parent/'request.json'),record['mesh']:sha(record['mesh']),
            str(mr/'masks.npz'):sha(mr/'masks.npz'),str(mr/'cameras.json'):sha(mr/'cameras.json')},
        evidence_sha256=sha(out/'evidence.npz'),rgb_not_used=True,labels_not_changed=True,
        centroid_visibility_only_not_pixel_footprint_approval=True))
    print(records,flush=True)


if __name__=='__main__':main()
