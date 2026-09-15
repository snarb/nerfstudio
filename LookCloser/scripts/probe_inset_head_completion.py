"""Train-mask-constrained inward Poisson shell probes; NOT observed geometry.

One fixed four-arm recipe at two times. Only new proposal vertices move, along
their radius toward the median existing head vertex. No target/heldout geometry
input. This first screen checks semantic feasibility and native clay coverage;
it deliberately does NOT certify measured-depth safety or promote a mesh.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
from joint_temporal_texture import read, sha, atomic_json, cameras
from transfer_close_boundary_completion import ROOT as RAW, SOURCE, FRAMES, MOVIE
from refine_measured_head_masks import ROOT as MASKS
from study_jaw_repair_transfer import mask_votes
from study_confidence_depth_prior import REGIONS, region_masks
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from review_jaw_repair_transfer import panel

ROOT = Path('/mnt/data/dec5_inset_head_completion')
INSETS = (0., .001, .003, .006)


def inset_vertices(vertices, center, distance):
    delta = np.asarray(vertices) - np.asarray(center)
    radius = np.linalg.norm(delta, axis=1)
    # Never pass through the center; relevant head vertices are well outside it.
    step = np.minimum(distance, radius * .25)
    return np.asarray(vertices) - delta * (step / np.maximum(radius, 1e-12))[:, None]


def run(frame):
    folder = ROOT / frame; folder.mkdir(parents=True, exist_ok=False)
    base = read(SOURCE / frame / 'request.json')
    assert sha(base['source_mesh']) == base['source_mesh_sha256']
    mask_result = read(MASKS / frame / 'result.json')
    for p, h in mask_result['hashes'].items():
        assert sha(MASKS / frame / p) == h
    names = read(MASKS / frame / 'cameras.json')
    masks = np.load(MASKS / frame / 'masks.npz')['masks']
    rows, _, _ = cameras(frame)
    mesh = o3d.io.read_triangle_mesh(base['source_mesh']); mesh.compute_triangle_normals()
    v = np.asarray(mesh.vertices); t = np.asarray(mesh.triangles); nt = len(t)
    center = np.median(v[v[:, 0] > -.03], axis=0)
    raw = o3d.io.read_triangle_mesh(str(RAW / frame / 'poisson_raw.ply'))
    raw.compute_vertex_normals(); rv = np.asarray(raw.vertices); rt = np.asarray(raw.triangles)
    original = scene_for(v, t)
    edges, counts = np.unique(np.sort(t[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_counts=True)
    boundary = cKDTree(v[np.unique(edges[counts == 1])])
    req = dict(frame=frame, insets=list(INSETS), center=center.tolist(), min_head_x=-.03,
               maximum_surface_distance=.006, maximum_boundary_distance=.006,
               maximum_triangle_edge=.0015, minimum_normal_dot=.25,
               semantic_support_minimum=2, semantic_outside_maximum=0,
               source_request_sha256=sha(SOURCE / frame / 'request.json'),
               source_mesh=base['source_mesh'], source_mesh_sha256=sha(base['source_mesh']),
               raw_mesh_sha256=sha(RAW / frame / 'poisson_raw.ply'),
               refined_masks_sha256=sha(MASKS / frame / 'masks.npz'),
               scripts={str(Path(__file__).resolve().with_name(n)): sha(Path(__file__).with_name(n))
                        for n in [Path(__file__).name, 'study_jaw_repair_transfer.py']},
               heldout_used=False, geometry_uses_target=False, original_geometry_preserved=True,
               inferred_not_measured=True, observed_depth_guard_passed=False, production_updated=False)
    atomic_json(folder / 'request.json', req)
    native = next(r for r in rows if r['physical_camera'] == REGIONS[frame]['camera'])
    moving = next(r['camera'] for r in read(MOVIE / 'request.json')['inventory'] if r['frame_id'] == frame)
    views = [('native_train', native), ('moving', moving)]
    old_depth = {name: camera_depth(original, cam)[0] for name, cam in views}
    native_roi = region_masks(frame)['hair']
    records = []; images = {name: [] for name, _ in views}
    for inset in INSETS:
        label = f'inset_{round(inset * 1000000):06d}'; arm = folder / label; arm.mkdir()
        shifted = inset_vertices(rv, center, inset)
        closest = original.compute_closest_points(o3d.core.Tensor(shifted.astype(np.float32)))
        dist = np.linalg.norm(shifted - closest['points'].numpy(), axis=1)
        bd = boundary.query(shifted)[0]
        dot = np.sum(np.asarray(raw.vertex_normals) * np.asarray(mesh.triangle_normals)[closest['primitive_ids'].numpy()], axis=1)
        good = (dist <= .006) & (bd <= .006) & (shifted[:, 0] > -.03) & (dot >= .25)
        length = np.linalg.norm(shifted[rt] - shifted[rt[:, [1, 2, 0]]], axis=2).max(1)
        ids = np.flatnonzero(good[rt].all(1) & (length <= .0015))
        support, outside = mask_votes(shifted, rt[ids], rows, masks, names)
        keep = ids[(support >= 2) & (outside == 0)]
        vv = np.concatenate([v, shifted]); tt = np.concatenate([t, rt[keep] + len(v)])
        candidate = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv), o3d.utility.Vector3iVector(tt))
        candidate.compute_triangle_normals(); scene = scene_for(vv, tt)
        o3d.io.write_triangle_mesh(str(arm / 'mesh.ply'), candidate)
        np.savez_compressed(arm / 'evidence.npz', proposal_ids=ids, mask_support=support,
                            mask_outside=outside, retained_raw_triangle_ids=keep)
        stats = []
        for name, cam in views:
            d, hit, _ = camera_depth(scene, cam); valid = np.isfinite(d); old = np.isfinite(old_depth[name])
            new = valid & ~old
            rgb = np.zeros((*d.shape, 3), np.uint8)
            shade = np.abs(np.asarray(candidate.triangle_normals) @ np.array([.3, .4, .866]))
            rgb[valid] = (60 + 170 * shade[hit[valid], None]).astype(np.uint8)
            rgb[new] = [255, 70, 70]
            portrait = np.rot90(rgb).copy(); Image.fromarray(portrait).save(arm / (name + '.png'))
            images[name].append(portrait)
            np.savez_compressed(arm / (name + '_depth.npz'), depth=d, triangle_ids=hit)
            stat = dict(view=name, new_depth_pixels=int(new.sum()), lost_depth_pixels=int((old & ~valid).sum()),
                        visible_added_pixels=int((valid & (hit >= nt)).sum()), counts_not_anatomical_metrics=True)
            if name == 'native_train':
                stat['coarse_hair_new_depth_pixels'] = int((new & native_roi).sum())
            stats.append(stat)
        record = dict(inset=inset, local_proposals=len(ids), semantic_admitted=len(keep), views=stats,
                      hashes={str(p.name): sha(p) for p in arm.iterdir() if p.is_file()})
        atomic_json(arm / 'result.json', record); records.append(record)
        atomic_json(folder / 'progress.json', dict(completed_arms=len(records), unix_time=time.time()))
        print(frame, label, len(keep), stats, flush=True)
    for name, _ in views:
        panel(folder / (name + '_comparison.png'), images[name], [str(x) for x in INSETS], (170, 450, 970, 850))
    atomic_json(folder / 'result.json', dict(request_sha256=sha(folder / 'request.json'), records=records,
                observed_depth_guard_passed=False, visual_status='pending', production_updated=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--frame', required=True, choices=FRAMES)
    run(parser.parse_args().frame)
