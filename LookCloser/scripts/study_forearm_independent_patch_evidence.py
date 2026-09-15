"""Contrast competing arm depths in train views excluded from prior inference.

Read-only confidence canary, including deliberately displaced measured controls.
No geometry admission changes, RGB refinement, or automatic quality approval.
"""
from pathlib import Path
import time
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json
from calibrated_depth_witness import load_images
from study_confidence_depth_prior import support, unproject
from bake_joint_temporal_mesh import camera_depth
from diffusion_mesh_repair import scene_for
from study_foundation_lower_forearm import ROOT as LOWER, FRAME
from plane_patch_evidence import compare_planes, evidence_decision

ROOT = Path('/mnt/data/dec5_forearm_independent_patch_evidence')


def spaced(ids, count):
    return ids[np.linspace(0, len(ids)-1, min(count, len(ids))).astype(int)] if len(ids) else ids


def select_witnesses(query, rows, excluded, center):
    a = center-np.array(query['transform_matrix'])[:3, 3]
    a /= np.linalg.norm(a)
    candidates = []
    for row in rows:
        if row['physical_camera'] in excluded | {query['physical_camera']}:
            continue
        b = center-np.array(row['transform_matrix'])[:3, 3]
        angle = np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(b), -1, 1)))
        if 1 <= angle <= 35:
            candidates.append((angle, row['physical_camera'], row))
    return [r for _, _, r in sorted(candidates)[:8]]


def make_panel(path, result, witnesses, index, title):
    side = int(np.sqrt(result['query_rgb'].shape[1]));scale = 3
    width = max(150, side*scale+8)
    canvas = Image.new('RGB', ((len(witnesses)+1)*width, 300), 'black')
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill='white')
    def tile(a):
        return Image.fromarray(np.nan_to_num(a).clip(0, 255).astype(np.uint8).reshape(side, side, 3)).resize((side*scale, side*scale), Image.Resampling.NEAREST)
    canvas.paste(tile(result['query_rgb'][index]), (4, 55))
    draw.text((4, 28), 'fixed query', fill='white')
    for i, row in enumerate(witnesses):
        x = (i+1)*width
        draw.text((x+2, 28), row['physical_camera'], fill='white')
        for j, label in enumerate(['near', 'far']):
            y = 55+j*120
            canvas.paste(tile(result['patches'][i, j, index]), (x+2, y))
            score = result[label][index, i]
            text = f'{label} NCC {score:.3f}' if np.isfinite(score) else f'{label}: unavailable'
            draw.text((x+2, y+side*scale+2), text, fill='white')
    path.parent.mkdir(parents=True, exist_ok=True);canvas.save(path)


def run():
    import study_forearm_plane_transfer_v3 as real
    started = time.monotonic()
    ROOT.mkdir(exist_ok=False)
    real.configure();rows, depth_list, depth_receipt = real.v2.v1.load_real(FRAME)
    depths = {r['physical_camera']: d for r, d in zip(rows, depth_list)}
    q = read(LOWER / 'foreground/request.json');proposal = np.load(LOWER / 'foreground/proposal.npz')
    stage = read(LOWER / FRAME / 'request.json')
    excluded = {n for p in stage['pairs'] for n in [p['left'], p['right']]}
    assert len(excluded) == 4
    original = o3d.io.read_triangle_mesh(q['source_mesh'])
    assert sha(q['source_mesh']) == q['source_mesh_sha256']
    vertices = np.concatenate([np.asarray(original.vertices), proposal['added_vertices']])
    triangles = np.concatenate([np.asarray(original.triangles), proposal['added_triangles']+len(original.vertices)])
    normals = np.cross(vertices[triangles[:, 1]]-vertices[triangles[:, 0]], vertices[triangles[:, 2]]-vertices[triangles[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    images, _, receipt = load_images(FRAME)
    assert receipt == stage['rgb_receipt']
    request = dict(frame=FRAME, geometry_changed=False, heldout_used=False,
        source_rgb_receipt=receipt, source_depth_receipt=depth_receipt,
        source_mesh=q['source_mesh'], source_mesh_sha256=q['source_mesh_sha256'],
        proposal_sha256=sha(LOWER / 'foreground/proposal.npz'),
        stage_request_sha256=sha(LOWER / FRAME / 'request.json'),
        legacy_events_sha256=sha(LOWER / 'veto_diagnosis/result.json'),
        excluded_from_photo_validation=sorted(excluded), query_excluded=True,
        witness_choice='up to8 smallest central parallax angles in1..35deg, no RGB selection',
        rule=dict(configurations=['query_parallel_15', 'query_parallel_31', 'proposal_tangent_15', 'proposal_tangent_31'],
                  near_ncc=.65, near_minus_far=.2, minimum_query_std=2,
                  same_witnesses_all_configurations=3, maximum_far_favoring_witnesses=1,
                  query_conflicts_maximum=32, query_controls_maximum=16, measured_control_shift=.024,
                  measured_control_surface_agreement=.001, other_depth_votes=3,
                  source_center_occlusion_margin=.003),
        limitations=['Source center occlusion check does not certify every patch pixel.',
                     'Both hypotheses use same plane orientation; folds and boundaries invalidate planar approximation.',
                     'Measured controls are multi-view consistency controls, not ground-truth geometry.'],
        scripts={str(Path(__file__).with_name(n).resolve()): sha(Path(__file__).with_name(n)) for n in
                 [Path(__file__).name, 'plane_patch_evidence.py']})
    atomic_json(ROOT / 'request.json', request)
    raw_scene = scene_for(vertices, triangles)
    old_scene = scene_for(np.asarray(original.vertices), np.asarray(original.triangles))
    legacy = {g['camera']: g['samples'] for g in read(LOWER / 'veto_diagnosis/result.json')['records']}
    records = [];totals = dict(conflict=0, conflict_rejected=0, measured_control=0, measured_control_rejected=0, legacy=0, legacy_rejected=0)
    for ci, row in enumerate(rows):
        name = row['physical_camera'];actual = dict(row);actual['cx'] += .5;actual['cy'] += .5
        d, ids, _ = camera_depth(raw_scene, actual);observed = depths[name]
        candidate = np.isfinite(d) & (ids >= len(original.triangles)) & (ids < len(triangles))
        y, x = np.nonzero(candidate & (observed > 0) & (observed > d+.003))
        votes, _ = support(unproject(row, x, y, observed[y, x]), row, rows, depth_list)
        chosen = spaced(np.flatnonzero(votes >= 3), 32);x, y = x[chosen], y[chosen]
        events = [dict(kind='conflict', native_xy=[int(xx), int(yy)], near=float(d[yy, xx]),
                       far=float(observed[yy, xx]), triangle=int(ids[yy, xx])) for xx, yy in zip(x, y)]
        raw_count = int((votes >= 3).sum())
        # Controls are sampled from independently corroborated old-mesh skin
        # inside the proposal's query footprint bounding box, not chosen by NCC.
        ys, xs = np.nonzero(candidate)
        if len(xs):
            od, oi, _ = camera_depth(old_scene, actual)
            region = np.zeros(d.shape, bool);region[ys.min():ys.max()+1, xs.min():xs.max()+1] = True
            im = images[name];warm = im[..., 0].astype(float)-im[..., 2] > 8
            y, x = np.nonzero(region & warm & np.isfinite(od) & (observed > .024) & (np.abs(od-observed) <= .001))
            pre = spaced(np.arange(len(x)), 256);x, y = x[pre], y[pre]
            votes, _ = support(unproject(row, x, y, observed[y, x]), row, rows, depth_list)
            take = spaced(np.flatnonzero(votes >= 3), 16)
            for j in take:
                xx, yy = x[j], y[j]
                events.append(dict(kind='measured_control', native_xy=[int(xx), int(yy)],
                    near=float(observed[yy, xx]-.024), far=float(observed[yy, xx]), triangle=int(oi[yy, xx])))
        for j, event in enumerate(legacy.get(name, [])):
            events.append(dict(kind='legacy', legacy_index=j, native_xy=event['native_xy'], near=event['proposed_depth'],
                               far=event['observed_depth'], triangle=event['triangle']))
        if not events:
            continue
        xy = np.array([e['native_xy'] for e in events])
        near = unproject(row, xy[:, 0], xy[:, 1], np.array([e['near'] for e in events]))
        far = unproject(row, xy[:, 0], xy[:, 1], np.array([e['far'] for e in events]))
        witnesses = select_witnesses(row, rows, excluded, np.median(near, axis=0))
        if len(witnesses) < 3:
            records.append(dict(camera=name, status='insufficient_independent_views', events=len(events)));continue
        config = [];panels = []
        for normal_name, ns in [('query_parallel', np.tile(np.array(row['transform_matrix'])[:3, 2], (len(events), 1))),
                                ('proposal_tangent', normals[[e['triangle'] for e in events]])]:
            for radius in [7, 15]:
                result = compare_planes(row, witnesses, images, depths, near, far, ns, radius)
                config.append({k: result[k] for k in ['near', 'far', 'available', 'query_std']})
                if normal_name == 'proposal_tangent' and radius == 15:
                    for i, event in enumerate(events):
                        if event['kind'] == 'legacy' and (name == 'F004_E005_1210FP' or event['legacy_index'] == 1):
                            path = ROOT / 'review' / f'{name}_{event["legacy_index"]}.png'
                            make_panel(path, result, witnesses, i, f'{name} sample {event["legacy_index"]}: parallel competing tangent planes,31px; query fixed')
                            panels.append(dict(path=str(path), sha256=sha(path)))
        arrays = {k: np.array([c[k] for c in config]) for k in config[0]}
        decision = evidence_decision(**arrays)
        dest = ROOT / name;dest.mkdir()
        np.savez_compressed(dest / 'evidence.npz', **arrays, **decision, near_points=near, far_points=far)
        counts = {}
        for kind in ['conflict', 'measured_control', 'legacy']:
            selected = np.array([e['kind'] == kind for e in events])
            counts[kind] = int(selected.sum());counts[kind+'_rejected'] = int((selected & decision['reject_far']).sum())
            totals[kind] += counts[kind];totals[kind+'_rejected'] += counts[kind+'_rejected']
        atomic_json(dest / 'events.json', dict(events=events, witnesses=[w['physical_camera'] for w in witnesses],
                    evidence_sha256=sha(dest / 'evidence.npz'), request_sha256=sha(ROOT / 'request.json')))
        records.append(dict(camera=name, status='scored', raw_trusted_conflicts=raw_count, counts=counts, panels=panels))
        print(ci+1, name, counts, flush=True)
    atomic_json(ROOT / 'result.json', dict(request_sha256=sha(ROOT / 'request.json'), records=records, totals=totals,
        elapsed_seconds=time.monotonic()-started, geometry_changed=False, production_updated=False, visual_status='pending'))
    print('TOTAL', totals, flush=True)


if __name__ == '__main__':
    run()
