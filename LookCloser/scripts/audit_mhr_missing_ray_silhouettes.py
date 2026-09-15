"""Replay bounded silhouette-ray evidence and its independent depth check."""
from pathlib import Path
import numpy as np
import cv2
from build_train_hair_semantics import read, sha, write
from admit_mhr_local_patch_depth import inputs, PRIOR, OUT, Scene2
from joint_temporal_texture import project
from study_jaw_depth_footprint import train_reference_votes


def main():
    import open3d as o3d
    q,rows,depths,masks,names,binding=inputs(); root=OUT/'ray_silhouette_probe'; result=read(root/'result.json')
    assert result['inputs']==binding
    assert result['script_sha256']==sha(Path(__file__).with_name('probe_mhr_missing_ray_silhouettes.py'))
    source=PRIOR/'probe_smooth025_smooth100_smooth400/smooth025.npz'; assert sha(source)==result['query_sha256']
    camera_path=Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json'); assert sha(camera_path)==result['target_parent_sha256']
    cam=next(r['camera'] for r in read(camera_path)['inventory'] if r['frame_id']=='001193')
    query=np.load(source)['target_prior_points']; center=np.asarray(cam['transform_matrix'])[:3,3]
    unit=query-center; unit/=np.linalg.norm(unit,axis=1,keepdims=True); offsets=np.linspace(-.01,.01,401)
    points=(query[:,None]+unit[:,None]*offsets[None,:,None]).reshape(-1,3); records=[]
    for radius in [0,2]:
        data=np.load(root/('radius'+str(radius)+'.npz'))
        np.testing.assert_array_equal(data['query_points'],query); np.testing.assert_array_equal(data['unit'],unit)
        np.testing.assert_array_equal(data['offsets'],offsets)
        for ci,row in enumerate(rows):
            uv,z=project(points,[row]); uv=uv[0]; z=z[0]; xy=np.rint(uv).astype(int)
            valid=(z>0)&(uv[:,0]>2)&(uv[:,0]<1917)&(uv[:,1]>2)&(uv[:,1]<1077)
            mask=masks[names.index(row['physical_camera'])].astype(np.uint8)
            if radius: mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
            bad=np.zeros(len(points),bool); idx=np.flatnonzero(valid); bad[idx]=~mask[xy[idx,1],xy[idx,0]].astype(bool)
            np.testing.assert_array_equal(bad.reshape(44,401),data['outside_by_camera'][ci])
            np.testing.assert_array_equal(valid.reshape(44,401),data['available_by_camera'][ci])
        counts=data['outside_by_camera'].sum(0); feasible=(counts==0)&(data['available_by_camera'].sum(0)>=2)
        np.testing.assert_array_equal(feasible,data['feasible']); assert feasible.any(1).all()
        distance=np.where(feasible,abs(offsets)[None],np.inf).min(1)
        records.append(dict(radius=radius,nearest_feasible_offset_quantiles=np.quantile(distance,[0,.5,1]).tolist(),
            zero_offset_outside_quantiles=np.quantile(counts[:,200],[0,.5,1]).tolist()))
        if radius==0: selected=query+unit*offsets[np.argmax(feasible,axis=1),None]
    mesh=o3d.io.read_triangle_mesh(q['source_mesh']); scene=Scene2(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
    nearest=scene.compute_closest_points(o3d.core.Tensor(selected.astype(np.float32)))['points'].numpy()
    distance=np.linalg.norm(nearest-selected,axis=1); votes,_=train_reference_votes(selected,rows,depths)
    np.savez_compressed(root/'feasible_depth_evidence.npz',points=selected,nearest_original_points=nearest,distance=distance,votes=votes)
    verification=dict(ray_samples_replayed=44*401*62*2,arms=records,
        first_strict_feasible_surface_distance_quantiles=np.quantile(distance,[0,.5,1]).tolist(),
        first_strict_feasible_at_least_two_depth_views=int((votes>=2).sum()),
        first_strict_feasible_depth_vote_quantiles=np.quantile(votes,[0,.5,1]).tolist(),
        bounded_silhouette_feasibility_is_not_surface_truth=True,geometry_changed=False)
    write(root/'verification.json',verification)
    files=[p for p in root.iterdir() if p.is_file() and p.name!='artifact_manifest.json']
    files += [Path(__file__).resolve(),Path(__file__).with_name('probe_mhr_missing_ray_silhouettes.py').resolve(),
              Path(__file__).resolve().parents[1]/'experiments/dec5_mhr_missing_ray_silhouettes.md']
    write(root/'artifact_manifest.json',dict(bindings={str(p):sha(p) for p in files},status='diagnostic_replay_passed_not_geometry_repair'))
    print(verification,flush=True)


if __name__=='__main__': main()
