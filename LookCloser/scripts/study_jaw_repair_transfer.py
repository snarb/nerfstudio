"""Transfer the frozen train-anchor/footprint notch recipe to additional times.

No target camera or target RGB constructs proposals/confidence. Defaults of all
production tools stay unchanged. Historical canaries must replay exact meshes.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import time
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json, project, cameras
from study_jaw_boundary_notches import propose, SETTINGS
from diagnose_jaw_measured_depth import barycentric_samples
from study_jaw_depth_footprint import train_reference_votes
from study_jaw_train_confidence import footprint_veto
from guard_jaw_measured_depth import initial_admission, measured_pixel_veto
from study_confidence_depth_prior import load_real
from diffusion_mesh_repair import scene_for

PARENT = Path('/mnt/data/dec5_phase30_early_texture_dynamic_150')
OUT = Path('/mnt/data/dec5_jaw_repair_transfer')
FRAMES = ['001083', '001123', '001193', '001195']
SCRIPTS = ['study_jaw_repair_transfer.py', 'study_jaw_boundary_notches.py',
           'boundary_cycle_blocks.py', 'diagnose_jaw_measured_depth.py',
           'study_jaw_depth_footprint.py', 'study_jaw_train_confidence.py',
           'guard_jaw_measured_depth.py', 'study_confidence_depth_prior.py']


def mask_votes(vertices, proposals, rows, masks, names):
    points = np.concatenate([vertices[proposals], vertices[proposals].mean(1)[:, None]], axis=1)
    support = np.zeros(len(proposals), int)
    outside = np.zeros(len(proposals), int)
    for row in rows:
        uv, z = project(points.reshape(-1, 3), [row]); uv, z = uv[0], z[0]
        available = (z > 0) & (uv[:, 0] > 2) & (uv[:, 0] < 1917) & (uv[:, 1] > 2) & (uv[:, 1] < 1077)
        xy = np.rint(uv).astype(int); ids = np.flatnonzero(available)
        inside = np.zeros(len(uv), bool)
        inside[ids] = masks[names.index(row['physical_camera']), xy[ids, 1], xy[ids, 0]]
        support += inside.reshape(-1, 4).all(1)
        outside += (available & ~inside).reshape(-1, 4).any(1)
    return support, outside


def prepare(output, frame, mask_override_root=None):
    source = next(r for r in read(PARENT/'request.json')['inventory'] if r['frame_id'] == frame)
    folder = output/frame; folder.mkdir(parents=True, exist_ok=True)
    old_root = Path('/mnt/data/dec5_jaw_train_confidence/footprint')
    depth_root = old_root/'analysis' if frame in ['001193', '001195'] else Path('/mnt/data/dec5_confidence_depth_prior')
    rows, depths, receipt = load_real(depth_root, frame)
    if frame in ['001193', '001195']:
        expected = read(old_root/'guarded'/frame/'result.json')['depth_receipt']
    else:
        analysis = read(depth_root/frame/'analysis.json')
        expected = analysis['real_depth']
    if expected != receipt: raise ValueError('Changed frozen measured depth')
    if len(rows) != 62 or len({r['physical_camera'] for r in rows}) != 62:
        raise ValueError('Expected 62 unique train cameras')
    if any(r['physical_camera'] in ['F004_B005_1210O9', 'J004_D005_1210TA', 'L004_B005_12106A'] for r in rows):
        raise ValueError('Heldout leakage')
    maskroot = Path(source['source_masks']['root'])
    if sha(source['mesh']) != source['mesh_sha256'] or sha(maskroot/'masks.npz') != source['source_masks']['masks_sha256']:
        raise ValueError('Changed source geometry/masks')
    request = dict(frame=frame, source_mesh=source['mesh'], source_mesh_sha256=source['mesh_sha256'],
        parent_request_sha256=sha(PARENT/'request.json'), depth_root=str(depth_root), depth_receipt=receipt,
        source_mask_sha256=sha(maskroot/'masks.npz'), mask_names_sha256=sha(maskroot/'cameras.json'),
        settings=SETTINGS, sample_rule='frozen train anchor plus four-tap footprint',
        minimum_supported_vertices=2, median_sample_votes=2, depth_tolerance=.001,
        final_free_separation=.003, final_other_depth_support=3, ray_offsets=[0,.5], max_rounds=8,
        geometry_uses_target_camera=False, heldout_rgb_used=False, production_accepted=False,
        scripts={n:sha(Path(__file__).with_name(n)) for n in SCRIPTS})
    masks=np.load(maskroot/'masks.npz')['masks'];names=read(maskroot/'cameras.json')
    if mask_override_root is not None:
        override=mask_override_root/frame;mr=read(override/'result.json')
        if mr['request_sha256']!=sha(override/'request.json') or mr['original_masks_sha256']!=sha(maskroot/'masks.npz'):
            raise ValueError('Override source mismatch')
        for name,h in mr['hashes'].items():
            if sha(override/name)!=h:raise ValueError('Changed mask override')
        replacement=np.load(override/'mask.npy');index=names.index(mr['camera'])
        if replacement.shape!=masks[index].shape or not replacement[masks[index].astype(bool)].all():
            raise ValueError('Override must preserve the original mask')
        masks=masks.copy();masks[index]=replacement
        request['mask_override']=dict(root=str(mask_override_root),result_sha256=sha(override/'result.json'),
            camera=mr['camera'],geometry_only=True,texture_masks_unchanged=True)
    if (folder/'request.json').exists() and read(folder/'request.json') != request:
        raise ValueError('Frozen transfer mismatch')
    atomic_json(folder/'request.json', request)
    if (folder/'result.json').exists():
        result = read(folder/'result.json')
        if result['request_sha256'] != sha(folder/'request.json'): raise ValueError('Changed result request')
        for p,h in result['hashes'].items():
            if sha(folder/p) != h: raise ValueError('Changed geometry artifact')
        print(frame, 'verified completed transfer', flush=True); return
    mesh = o3d.io.read_triangle_mesh(source['mesh']); v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    raw, notes = propose(v, t); proposals = raw[len(t):]
    points = barycentric_samples(v[proposals]); votes, references = train_reference_votes(points.reshape(-1,3), rows, depths)
    free = footprint_veto(points.reshape(-1,3), rows, depths).reshape(62,-1,10)
    mask_support, mask_outside = mask_votes(v, proposals, rows, masks, names)
    keep = initial_admission(votes.reshape(-1,10), free, mask_support, mask_outside)
    ids = np.flatnonzero(keep); triangles = np.concatenate([t,proposals[keep]]); rounds=[]
    for iteration in range(8):
        scene = scene_for(v,triangles); remove=set(); checks=[]
        for ci,(camera,depth) in enumerate(zip(rows,depths)):
            for offset in [0,.5]:
                implicated,count,raw_count = measured_pixel_veto(scene,camera,depth,rows,depths,len(t),len(triangles),offset)
                remove.update(implicated.tolist())
                checks.append(dict(camera=camera['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw_count))
            if (ci+1)%20==0:
                atomic_json(folder/'progress.json',dict(stage='measured_ray_guard',iteration=iteration,cameras=ci+1,unix_time=time.time()))
        rounds.append(dict(removed_triangles=len(remove),checks=checks))
        print(frame,'guard',iteration,'removed',len(remove),flush=True)
        if not remove: break
        take=np.ones(len(triangles),bool); take[list(remove)]=False
        if not take[:len(t)].all(): raise ValueError('Attempted original deletion')
        ids=ids[take[len(t):]]; triangles=triangles[take]
    if rounds[-1]['removed_triangles']: raise ValueError('Guard did not converge')
    final=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(triangles)); final.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(folder/'mesh.ply'),final): raise IOError('Mesh write failed')
    saved=o3d.io.read_triangle_mesh(str(folder/'mesh.ply'))
    if not np.array_equal(v,np.asarray(saved.vertices)) or not np.array_equal(t,np.asarray(saved.triangles)[:len(t)]):
        raise ValueError('Original prefix changed')
    replay=None
    if frame in ['001193','001195'] and mask_override_root is None:
        replay=sha(folder/'mesh.ply')==sha(old_root/'guarded'/frame/'mesh.ply')
        if not replay: raise ValueError('Historical canary did not replay exact mesh bytes')
    np.savez_compressed(folder/'evidence.npz',proposals=proposals,points=points,votes=votes.reshape(-1,10),
        references=references.reshape(-1,10),free=free,mask_support=mask_support,mask_outside=mask_outside,retained_proposal_ids=ids)
    atomic_json(folder/'result.json',dict(request_sha256=sha(folder/'request.json'),proposal_notes=notes,
        proposed=len(proposals),initially_admitted=int(keep.sum()),added=len(ids),rounds=rounds,
        historical_mesh_byte_replay=replay,original_prefix_exact=True,observed_guard_passed=True,
        hashes={p:sha(folder/p) for p in ['mesh.ply','evidence.npz']},visual_status='pending',production_accepted=False))
    print(frame,'done',len(proposals),int(keep.sum()),len(ids),'historical replay',replay,flush=True)


def render(output,frame):
    import render_smooth_temporal_mesh_video as renderer
    from study_early_texture_prior import install
    install(); renderer.torch.set_num_threads(2)
    parent=renderer.verify_request(PARENT); result=read(output/frame/'result.json')
    for p,h in result['hashes'].items():
        if sha(output/frame/p)!=h: raise ValueError('Changed candidate')
    if not result['observed_guard_passed']: raise ValueError('Failed geometry guard')
    train,_,_=cameras(frame)
    for view in ['moving','F004_E005_1210FP']:
        for variant in ['baseline','repaired']:
            dest=output/'rgb'/frame/view/variant; request=deepcopy(parent)
            request['inventory']=[r for r in request['inventory'] if r['frame_id']==frame]; row=request['inventory'][0]
            if view!='moving': row['camera']=next(r for r in train if r['physical_camera']==view)
            if variant=='repaired': row.update(mesh=str(output/frame/'mesh.ply'),mesh_sha256=sha(output/frame/'mesh.ply'))
            request.update(partial_diagnostic_only=True,full_video_candidate=False,jaw_transfer_variant=variant,
                geometry_result_sha256=sha(output/frame/'result.json'))
            request['script_hashes'][Path(__file__).name]=sha(__file__)
            dest.mkdir(parents=True,exist_ok=True); (dest/'frames').mkdir(exist_ok=True)
            if (dest/'request.json').exists() and read(dest/'request.json')!=request: raise ValueError('Changed RGB request')
            atomic_json(dest/'request.json',request); renderer.render(dest,[frame])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['prepare','render'])
    p.add_argument('--frame',choices=FRAMES,required=True); p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--mask-override-root',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.output,a.frame,a.mask_override_root)
    elif a.mask_override_root is not None:p.error('Mask override belongs to prepare, not render')
    else:render(a.output,a.frame)
