"""Independent integral-mask replay and fresh revealed-shell safety checks."""
from pathlib import Path
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json,project
from study_weak_fringe_replacement import ROOT,INSET,MASKS,SOURCE,FRAMES
from study_confidence_depth_prior import load_real
from study_jaw_depth_footprint import train_reference_votes
from guard_jaw_measured_depth import measured_pixel_veto
from diffusion_mesh_repair import scene_for


def integral_background(points,rows,masks,names):
    results=[]
    for row in rows:
        mask=masks[names.index(row['physical_camera'])]
        integral=np.pad(mask.astype(np.int32),((1,0),(1,0))).cumsum(0,dtype=np.int64).cumsum(1,dtype=np.int64)
        uv,z=project(points.reshape(-1,3),[row]);uv,z=uv[0],z[0];xy=np.rint(uv).astype(int)
        valid=(z>0)&(xy[:,0]>=4)&(xy[:,0]<1916)&(xy[:,1]>=4)&(xy[:,1]<1076)
        ids=np.flatnonzero(valid);x,y=xy[ids].T
        area=integral[y+5,x+5]-integral[y-4,x+5]-integral[y+5,x-4]+integral[y-4,x-4]
        empty=np.zeros(len(uv),bool);empty[ids]=area==0
        results.append(empty.reshape(-1,4).all(1))
    return np.stack(results)


def run():
    records=[]
    for frame in FRAMES:
        root=ROOT/frame;q=read(root/'request.json');a=np.load(root/'evidence.npz');complete=read(root/'complete.json')
        assert sha(root/'request.json')==complete['request_sha256'] and sha(root/'evidence.npz')==complete['evidence_sha256']
        for p,h in q['scripts'].items():assert sha(p)==h,p
        assert sha(q['source_mesh'])==q['source_mesh_sha256']
        old=o3d.io.read_triangle_mesh(q['source_mesh']);v=np.asarray(old.vertices);t=np.asarray(old.triangles)
        head=np.flatnonzero((v[t,0]>-.03).all(1));np.testing.assert_array_equal(head,a['head_triangles'])
        points=np.concatenate([v[t[head]],v[t[head]].mean(1)[:,None]],axis=1)
        base=read(SOURCE/frame/'request.json');rows,depths,receipt=load_real(Path(base['depth_root']),frame)
        assert receipt==q['depth_receipt']
        masks=np.load(MASKS/frame/'masks.npz')['masks'];names=read(MASKS/frame/'cameras.json')
        assert sha(MASKS/frame/'masks.npz')==q['refined_masks_sha256']
        bg=integral_background(points,rows,masks,names);counts=bg.sum(0)
        np.testing.assert_array_equal(bg,a['background_by_camera']);np.testing.assert_array_equal(counts,a['background_counts'])
        candidate=np.flatnonzero(counts>=6);np.testing.assert_array_equal(candidate,a['candidate_indices'])
        votes,refs=train_reference_votes(points[candidate].reshape(-1,3),rows,depths)
        votes=votes.reshape(-1,4);np.testing.assert_array_equal(votes,a['depth_votes'])
        np.testing.assert_array_equal(refs.reshape(-1,4),a['depth_references'])
        removed=head[candidate[(votes.max(1)<2)&(counts[candidate]>=6)]]
        np.testing.assert_array_equal(removed,a['removed_triangles'])
        keep=np.ones(len(t),bool);keep[removed]=False
        shell=o3d.io.read_triangle_mesh(str(INSET/frame/'guarded/mesh.ply'))
        assert sha(INSET/frame/'guarded/mesh.ply')==q['shell_mesh_sha256']
        for arm in ['remove_only','replace']:
            folder=root/arm;r=read(folder/'result.json');ret=np.load(folder/'retained.npz')
            assert sha(folder/'result.json')==complete['results'][arm]
            assert sha(folder/'mesh.ply')==r['hashes']['mesh.ply']
            np.testing.assert_array_equal(ret['original_triangle_ids'],np.flatnonzero(keep))
            mesh=o3d.io.read_triangle_mesh(str(folder/'mesh.ply'));mv=np.asarray(mesh.vertices);mt=np.asarray(mesh.triangles)
            np.testing.assert_array_equal(mv,v if arm=='remove_only' else np.asarray(shell.vertices))
            expected=t[keep] if arm=='remove_only' else np.concatenate([t[keep],np.asarray(shell.triangles)[len(t):][ret['shell_triangle_ids']]])
            np.testing.assert_array_equal(mt,expected)
            checks=[]
            if arm=='replace':
                scene=scene_for(mv,mt)
                for row,depth in zip(rows,depths):
                    for offset in [0,.5]:
                        ids,count,raw_count=measured_pixel_veto(scene,row,depth,rows,depths,int(keep.sum()),len(mt),offset)
                        assert not len(ids) and count==0
                        checks.append(dict(camera=row['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw_count))
            records.append(dict(frame=frame,arm=arm,removed_original_triangles=len(removed),retained_shell_triangles=len(ret['shell_triangle_ids']),
                original_vertex_positions_exact=True,remaining_original_triangles_exact=True,
                background_integral_replay=True,all_removed_samples_below_two_depth_votes=True,native_ray_checks=checks,
                mesh_sha256=sha(folder/'mesh.ply'),production_updated=False))
        print(frame,'audited removal',len(removed),'and revealed shell',flush=True)
    atomic_json(ROOT/'geometry_audit.json',dict(records=records,script_sha256=sha(__file__),quality_approval=False,production_updated=False))


if __name__=='__main__':run()
