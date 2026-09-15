"""Foreground-layer omission canary: a hole can expose an existing deeper mesh."""
from pathlib import Path
import time
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from build_foundation_consensus_patch import ROOT as CONTROL, MOVIE
from study_foundation_anchor_bias import ROOT as BIAS
from review_foundation_hand_geometry import grid_triangles
from diffusion_mesh_repair import scene_for
from guard_jaw_measured_depth import measured_pixel_veto

ROOT=Path('/mnt/data/dec5_foundation_foreground_patch')


def missing_foreground(agreement,old_depth,proposed_depth,minimum_separation=.003):
    return agreement & np.isfinite(proposed_depth) & (proposed_depth>0) & \
        (~np.isfinite(old_depth) | (old_depth>proposed_depth+minimum_separation))


def run():
    import study_forearm_plane_transfer_v3 as real
    start=time.monotonic();request=read(CONTROL/'request.json');result=read(CONTROL/'result.json')
    assert sha(CONTROL/'request.json')==result['request_sha256']
    assert sha(BIAS/'point_fields.npz')==request['point_fields_sha256']
    assert sha(BIAS/'result.json')==request['bias_result_sha256']
    assert sha(request['source_mesh'])==request['source_mesh_sha256']
    assert sha(Path(__file__).with_name('build_foundation_consensus_patch.py'))==request['script_sha256']
    data=np.load(CONTROL/'proposal.npz'); fields=np.load(BIAS/'point_fields.npz')
    prefix=request['reference']+'_offset_diagnostic_';xyz=fields[prefix+'xyz'];z=fields[prefix+'depth']
    eligible=missing_foreground(data['agreement'],data['olddepth'],z)
    mesh=o3d.io.read_triangle_mesh(request['source_mesh']);v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    newv,newt=grid_triangles(xyz,eligible,maximum_edge=.002)
    vv=np.concatenate((v,newv));tt=np.concatenate((t,newt+len(v)))
    ROOT.mkdir(exist_ok=False)
    atomic_json(ROOT/'request.json',dict(control_request_sha256=sha(CONTROL/'request.json'),
        control_result_sha256=sha(CONTROL/'result.json'),control_proposal_sha256=sha(CONTROL/'proposal.npz'),
        source_mesh=request['source_mesh'],source_mesh_sha256=request['source_mesh_sha256'],
        point_fields_sha256=sha(BIAS/'point_fields.npz'),minimum_depth_layer_separation=.003,
        geometry_inferred=True,no_old_geometry_deleted=True,heldout_used=False,
        rule='All control cross-pair/mask/edge guards; replace empty-ray-only eligibility with missing foreground layer',
        script_sha256=sha(__file__),guard_sha256=sha(Path(__file__).with_name('guard_jaw_measured_depth.py'))))
    np.savez_compressed(ROOT/'proposal.npz',eligible=eligible,added_vertices=newv,added_triangles=newt)
    real.configure();rows,depths,hashes=real.v2.v1.load_real('001037');assert hashes==request['source_depth_hashes']
    rounds=[]
    for iteration in range(8):
        scene=scene_for(vv,tt);remove=set();checks=[]
        for row,depth in zip(rows,depths):
            for offset in [0,.5]:
                bad,count,raw=measured_pixel_veto(scene,row,depth,rows,depths,len(t),len(tt),offset)
                remove.update(bad.tolist());checks.append(dict(camera=row['physical_camera'],offset=offset,trusted_free=count,raw_far=raw))
        rounds.append(dict(iteration=iteration,removed=len(remove),checks=checks))
        print('foreground guard',iteration,'remove',len(remove),'from',len(tt)-len(t),flush=True)
        if not remove:break
        take=np.ones(len(tt),bool);take[list(remove)]=False;assert take[:len(t)].all();tt=tt[take]
    final=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),o3d.utility.Vector3iVector(tt));final.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(ROOT/'mesh.ply'),final)
    saved=o3d.io.read_triangle_mesh(str(ROOT/'mesh.ply'))
    np.testing.assert_array_equal(np.asarray(saved.vertices)[:len(v)],v)
    np.testing.assert_array_equal(np.asarray(saved.triangles)[:len(t)],t)
    result=dict(request_sha256=sha(ROOT/'request.json'),mesh_sha256=sha(ROOT/'mesh.ply'),
        eligible_pixels=int(eligible.sum()),proposed_triangles=len(newt),retained_triangles=len(tt)-len(t),
        original_arrays_preserved=True,guard_passed=not rounds[-1]['removed'],rounds=rounds,
        production_updated=False,visual_status='pending',elapsed_seconds=time.monotonic()-start)
    atomic_json(ROOT/'result.json',result)
    print({k:v for k,v in result.items() if k!='rounds'},flush=True)


if __name__=='__main__':run()
