"""Append only dual-pair agreed learned surface in existing rectified-view holes.

High-confidence PatchMatch native-pixel free-space guards remain mandatory.
No replacement of existing triangles and no automatic production acceptance.
"""
from pathlib import Path
import time
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_erosion
from joint_temporal_texture import read, sha, atomic_json, cameras
from study_foundation_anchor_bias import ROOT as BIAS, SOURCES, sample
from review_foundation_hand_geometry import grid_triangles
from diffusion_mesh_repair import scene_for
from guard_jaw_measured_depth import measured_pixel_veto
from review_hand_silhouette_volume import shaded
from review_jaw_repair_transfer import panel

ROOT = Path('/mnt/data/dec5_foundation_consensus_patch')
MOVIE = Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150')


def project(points, k, e):
    p = points @ e[:3,:3].T + e[:3,3]; uv = p @ k.T
    return uv[:,:2]/uv[:,2:], p[:,2]


def rectified_mesh_depth(scene, k, e, shape):
    y,x = np.indices(shape); direction = np.stack(((x-k[0,2])/k[0,0],(y-k[1,2])/k[1,1],np.ones(shape)),axis=-1) @ e[:3,:3]
    center = -e[:3,3] @ e[:3,:3]
    rays = np.concatenate((np.broadcast_to(center,direction.shape),direction),axis=-1).astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()


def run():
    import study_forearm_plane_transfer_v3 as real
    start = time.monotonic(); analysis = read(BIAS/'result.json'); q = read(BIAS/'request.json')
    assert sha(BIAS/'request.json') == analysis['request_sha256']
    for path,digest in {**q['source_depth_hashes'],**q['scripts'],**analysis['dependencies']}.items(): assert sha(path)==digest
    real.configure(); rows,depths,hashes = real.v2.v1.load_real('001037'); assert hashes == q['source_depth_hashes']
    inventory = read(MOVIE/'request.json')['inventory']; entry = next(r for r in inventory if r['frame_id']=='001037')
    assert sha(entry['mesh']) == entry['mesh_sha256']
    maps = np.load(BIAS/'point_fields.npz'); accepted = {r['pair']:r for r in analysis['pairs'] if r['status']=='passes_anchor_bias_gate'}
    pairs = []
    for source in SOURCES:
        for r in read(source/'request.json')['pairs']:
            name = Path(r['directory']).name
            if name not in accepted: continue
            cal = np.load(Path(r['directory'])/'calibration.npz'); prefix = name+'_offset_diagnostic_'
            m = {key:maps[prefix+key] for key in ['xyz','valid','depth','K','E']}
            yy,xx = np.indices(m['valid'].shape)
            fb = float(cal['cropped_intrinsic'][0,0]*cal['baseline'])
            corrected_disparity = fb/m['depth']+float(cal['disparity_offset'])
            # Recheck the right mask at corrected, not original disparity.
            right = sample(binary_erosion(cal['right_mask'].astype(bool),iterations=2).astype(float),
                           np.column_stack((xx.ravel()-corrected_disparity.ravel(),yy.ravel()))).reshape(xx.shape)>.999
            m['valid'] &= binary_erosion(cal['left_mask'].astype(bool),iterations=2)&right
            m.update(name=name,baseline=r['baseline'],physical=[r['left'],r['right']]); pairs.append(m)
    assert len(pairs)==2 and len(set(sum([p['physical'] for p in pairs],[])))==4
    # Deterministic calibration-only reference choice, not a render-selected view.
    a,b = sorted(pairs,key=lambda r:(-r['baseline'],r['name']))
    points = a['xyz'].reshape(-1,3); uv,z = project(points,b['K'],b['E'])
    partner_valid = sample(b['valid'].astype(float),uv)>.999
    dz = np.abs(sample(b['depth'],uv)-z)
    partner_points = np.column_stack([sample(b['xyz'][...,axis],uv) for axis in range(3)])
    back,_ = project(partner_points,a['K'],a['E']); yy,xx = np.indices(a['valid'].shape)
    roundtrip = np.linalg.norm(back-np.column_stack((xx.ravel(),yy.ravel())),axis=1)
    agreement = (partner_valid&(z>0)&(dz<=.001)&(roundtrip<=2)).reshape(a['valid'].shape)&a['valid']
    mesh = o3d.io.read_triangle_mesh(entry['mesh']); v,t = np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    olddepth = rectified_mesh_depth(scene_for(v,t),a['K'],a['E'],agreement.shape)
    eligible = agreement&~np.isfinite(olddepth)
    newv,newt = grid_triangles(a['xyz'],eligible,maximum_edge=.002)
    vv = np.concatenate((v,newv)); tt = np.concatenate((t,newt+len(v)))
    ROOT.mkdir(exist_ok=False)
    atomic_json(ROOT/'request.json',dict(frame='001037',source_mesh=entry['mesh'],source_mesh_sha256=sha(entry['mesh']),
        bias_request_sha256=sha(BIAS/'request.json'),bias_result_sha256=sha(BIAS/'result.json'),
        point_fields_sha256=sha(BIAS/'point_fields.npz'),source_depth_hashes=hashes,
        reference=a['name'],partner=b['name'],reference_selection='largest calibrated baseline',
        cross_pair_depth_tolerance=.001,cross_pair_roundtrip=2,maximum_edge=.002,
        only_preexisting_reference_mesh_holes=True,original_arrays_preserved=True,
        all_62_camera_native_free_space_guard=True,guard_offsets=[0,.5],max_guard_rounds=8,
        script_sha256=sha(__file__),guard_sha256=sha(Path(__file__).with_name('guard_jaw_measured_depth.py')),
        geometric_prior_not_measured_surface=True,heldout_used=False,production_updated=False))
    np.savez_compressed(ROOT/'proposal.npz',eligible=eligible,agreement=agreement,olddepth=olddepth,
        proposed_vertices=newv,proposed_triangles=newt,cross_difference=dz.reshape(agreement.shape),roundtrip=roundtrip.reshape(agreement.shape))
    rounds=[]
    for iteration in range(8):
        scene=scene_for(vv,tt); remove=set(); checks=[]
        for row,depth in zip(rows,depths):
            for offset in [0,.5]:
                bad,count,raw = measured_pixel_veto(scene,row,depth,rows,depths,len(t),len(tt),offset)
                remove.update(bad.tolist());checks.append(dict(camera=row['physical_camera'],offset=offset,trusted_free=count,raw_far=raw))
        rounds.append(dict(iteration=iteration,removed=len(remove),checks=checks))
        print('guard',iteration,'remove',len(remove),'from',len(tt)-len(t),flush=True)
        if not remove: break
        take=np.ones(len(tt),bool);take[list(remove)]=False;assert take[:len(t)].all();tt=tt[take]
    final=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),o3d.utility.Vector3iVector(tt));final.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(ROOT/'mesh.ply'),final)
    saved=o3d.io.read_triangle_mesh(str(ROOT/'mesh.ply'))
    np.testing.assert_array_equal(np.asarray(saved.vertices)[:len(v)],v)
    np.testing.assert_array_equal(np.asarray(saved.triangles)[:len(t)],t)
    record=dict(agreed_pixels=int(agreement.sum()),eligible_hole_pixels=int(eligible.sum()),
        proposed_triangles=len(newt),retained_triangles=len(tt)-len(t),guard_passed=not rounds[-1]['removed'],
        rounds=rounds,mesh_sha256=sha(ROOT/'mesh.ply'),request_sha256=sha(ROOT/'request.json'),
        elapsed_seconds=time.monotonic()-start,geometry_inferred=True,production_updated=False,visual_status='pending')
    atomic_json(ROOT/'result.json',record)
    views={r['physical_camera']:r for r in rows if r['physical_camera'] in ['H004_A005_1210M6','E004_C005_1210YM']}
    views['moving']=entry['camera']; stats=[]
    for name,row in views.items():
        old,od=shaded(mesh,row);new,nd=shaded(saved,row)
        path=ROOT/'review'/(name+'.png');box=(0,1400,500,1920) if name!='moving' else (100,1360,720,1920)
        panel(path,[old,new],['original geometry','dual-pair guarded addition'],box)
        stats.append(dict(view=name,newly_visible=int(((od==0)&(nd>0)).sum()),
            nearer_than_old_001=int(((od>0)&(nd>0)&(nd<od-.001)).sum()),
            note='visibility diagnostics include background; nearer counts are not automatically trusted-old regressions',
            panel=str(path),panel_sha256=sha(path)))
    atomic_json(ROOT/'review/result.json',dict(records=stats,visual_status='pending'))
    print('complete',{k:val for k,val in record.items() if k!='rounds'},stats,flush=True)


if __name__=='__main__':run()
