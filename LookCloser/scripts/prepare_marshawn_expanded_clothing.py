"""Seal manually reviewed extra clothing geometry; never authorize training.

The point/region labels below record inspection of the three-camera review
atlases. They must not be generalized to unreviewed semantic-mask candidates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from marshawn_original_feature_tracks import recover_track
from marshawn_softplus_prefix_candidate import sha256, validate_geometry


REGIONS = {
    7961: 'black_trouser_lettering', 8394: 'black_trouser_lettering',
    8502: 'black_trouser_lettering', 9020: 'black_trouser_lettering',
    8679: 'black_trouser_lettering', 8173: 'black_trouser_lettering',
    5568: 'black_sleeve_RESERVED', 2219: 'grey_trouser_RESERVED',
    5556: 'grey_trouser_RESERVED', 2579: 'grey_jacket_shoulder',
    943: 'grey_jacket_lowerback', 7784: 'grey_jacket_forearm',
    6820: 'grey_jacket_elbow_hem', 981: 'grey_jacket_upperback',
    7588: 'grey_jacket_elbow', 3000: 'grey_jacket_back_seam',
    946: 'grey_jacket_lowerback', 8931: 'grey_jacket_shoulder',
    7015: 'grey_jacket_elbow_hem', 8492: 'grey_jacket_front',
    1521: 'grey_jacket_upperback', 5452: 'grey_jacket_elbow',
    352: 'grey_jacket_lowerback', 815: 'grey_jacket_upperback',
    2047: 'grey_jacket_back_shoulder_seam',
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(campaign, output):
    campaign, output = Path(campaign), Path(output)
    root, data = campaign.parent, campaign/'data'
    review_root = campaign/'expanded_clothing_review'
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    original_tf = root/'data/transforms.json'
    old_rows = {r['physical_camera']: r for r in json.loads(original_tf.read_text())['frames']}
    rows = {Path(r['file_path']).stem.split('_')[-1]: r
            for r in json.loads((data/'transforms.json').read_text())['frames']}
    tracks_path, witness_path = root/'geometry/strict_tracks.json', root/'geometry/train_witnesses.npz'
    tracks = json.loads(tracks_path.read_text())
    with np.load(witness_path) as z:
        xyz, names = z['xyz'].copy(), z['cameras'].astype(str).tolist()
    audit = json.loads((root/'geometry/audit.json').read_text())
    if sha256(original_tf) != audit['transforms_sha256']:
        raise ValueError('Original geometry camera transforms changed')
    pairs = {(p['a'], p['b']) for p in audit['pairs']}
    ready = json.loads((data/'ready.json').read_text())
    records = {r['camera']: r for r in ready['records']}
    train = sorted(r['camera'] for r in ready['records'] if Path(r['image']).name.startswith('train_'))
    if names != train or set(train) & {'DC', 'HC', 'KA'}:
        raise ValueError('Source domain must be exactly 63 TRAIN cameras')
    box = json.loads((data/'box.json').read_text())
    transform, bounds = np.array(box['world_to_box']), np.array(box['aabb'])
    candidates = {}
    source_files = [original_tf, data/'transforms.json', data/'box.json', tracks_path,
                    witness_path, root/'geometry/audit.json', Path(__file__),
                    Path(__file__).with_name('marshawn_original_feature_tracks.py')]
    for actor in ('black', 'black_round2', 'grey'):
        source = review_root/f'{actor}_review_candidates.json'
        source_files.append(source)
        for p in json.loads(source.read_text()):
            candidates[p['point']] = p
    features, evidence, selected, measured = {}, {}, [], []
    original_archive = root/'native_png_radialK4_COLD_STAGED/geometry/scene_loose.npz'
    source_files.append(original_archive)
    with np.load(original_archive) as z:
        original = {k: z[k].copy() for k in z.files}
    if original['camera_names'].astype(str).tolist() != names:
        raise ValueError('Original native source camera identity mismatch')
    seen_pixels = set()
    for point, region in REGIONS.items():
        candidate = candidates[point]
        cameras = [names[i] for i in tracks['camera_ids'][point]]
        for camera in cameras:
            if camera in features:
                continue
            path = root/'geometry/features'/f'{camera}.npz'
            with np.load(path) as z:
                features[camera] = {k: z[k].copy() for k in z.files}
            rgb = root/'data'/old_rows[camera]['file_path']
            if sha256(rgb) != str(features[camera]['sha']):
                raise ValueError('Cached detector source RGB changed')
            source_files.extend([path, rgb])
        recovered = recover_track(xyz[point], cameras, old_rows, features, pairs)
        current_xyz = transform[:3, :3]@xyz[point]+transform[:3, 3]
        if not ((current_xyz > bounds[0]) & (current_xyz < bounds[1])).all():
            raise ValueError('Current actor point outside field')
        actor = candidate['candidate_actor']
        reviewed = candidate['review_cameras']
        observations = []
        for camera in reviewed:
            source_observation = next(o for o in candidate['observations'] if o['camera'] == camera)
            i = int(source_observation['observation_idx'])
            if int(original['point_idx'][i]) != point or names[int(original['camera_idx'][i])] != camera:
                raise ValueError('Original observation membership changed')
            uv = original['uv'][i]
            x, y = np.rint(uv).astype(int)
            if (camera, x, y) in seen_pixels:
                raise ValueError('Duplicate reviewed image pixel')
            seen_pixels.add((camera, x, y))
            image, semantic = data/records[camera]['image'], data/'semantics'/f'{camera}.npz'
            if sha256(image) != records[camera]['rgb_sha256'] or sha256(semantic) != records[camera]['semantic_sha256']:
                raise ValueError('Reviewed source RGB/semantics changed')
            with np.load(semantic) as z:
                if not z[actor][y, x] or not z['valid'][y, x] or z['grey' if actor == 'black' else 'black'][y, x]:
                    raise ValueError('Reviewed source identity changed')
            evidence[camera] = dict(rgb=str(image), rgb_sha256=sha256(image),
                                    semantic_sha256=sha256(semantic))
            observations.append(dict(camera=camera, original_index=i, projected_uv=uv.tolist(),
                detected_uv=next(o['detected_uv'] for o in recovered['observations'] if o['camera'] == camera)))
            if not region.endswith('_RESERVED'):
                selected.append(i)
        measured.append(dict(point=point, region=region, actor=actor,
            status='reserved_whole_patch' if region.endswith('_RESERVED') else 'extra_prefix_supervision',
            reviewed_observations=observations, original_feature_graph=recovered, xyz_floor=current_xyz.tolist()))
    # Keep the exact original18 supervised rays; only source coverage changes.
    prior = campaign/'prefix_candidate_20261106/geometry_v2'
    with np.load(prior/'prefixes.npz') as z:
        base = {k: z[k].copy() for k in z.files}
    source_files.extend([prior/'prefixes.npz', prior/'manifest.json', prior/'visibility_review.json'])
    ids = np.asarray(selected, dtype=np.int64)
    origins = original['origins'][ids]@transform[:3, :3].T+transform[:3, 3]
    directions = original['directions'][ids]@transform[:3, :3].T
    depth = original['witness_depth'][ids]
    parallel = np.abs(directions) < 1e-12
    inv = 1./np.where(parallel, 1., directions)
    a, b = (bounds[0]-origins)*inv, (bounds[1]-origins)*inv
    enter = np.where(parallel, -np.inf, np.minimum(a, b)).max(1)
    leave = np.where(parallel, np.inf, np.maximum(a, b)).min(1)
    extra = dict(origins=origins, directions=directions, witness_xyz=origins+directions*depth[:, None],
        witness_depth=depth, t_start=np.maximum(np.maximum(enter+1e-5, original['t_start'][ids]), .01),
        t_end=np.minimum(depth-np.maximum(.01, .05*depth), leave-1e-5),
        point_idx=original['point_idx'][ids], camera_idx=original['camera_idx'][ids],
        source_track_view_count=np.array([len(tracks['camera_ids'][int(p)]) for p in original['point_idx'][ids]]))
    arrays = {k: (v if k == 'camera_names' else np.concatenate([v, extra[k]], axis=0)) for k, v in base.items()}
    validate_geometry(arrays, bounds, train)
    np.savez_compressed(output/'prefixes.npz', **arrays)
    # Reserve3 complete new clothing patches + preserve4 original nearby shirt reserves.
    reserved = [p for p, region in REGIONS.items() if region.endswith('_RESERVED')]
    review = dict(kind='MANUAL_TRAIN_CLOTHING_REVIEW_WITH_RECOVERED_FEATURE_GRAPH',
        points=measured, source_images=evidence, old_review_sha256=sha256(prior/'visibility_review.json'),
        rejected_categories=['hair', 'face', 'hands', 'mixed wrist/skin boundary', 'reflective boots', 'floor', 'bag'],
        reserves=dict(new_whole_patch_points=reserved, old_nearby_points=[63, 64, 71, 72],
                      new_black_back_lettering_points_not_admitted=True),
        uncertainty_note='Detected residuals and leave-one-view-out are correlated fitted TRAIN evidence, not independent GT; >=3view points without a fourth source have no leave-one-out3view check',
        prefix_policy='Unchanged max(.01, .05*depth) safety margin and strict AABB clipping',
        source_selection='Only the three visually inspected original source cameras per new point enter supervision; other source cameras establish the measured track')
    # All25 points receive actual source crops, including the reserved patch group.
    for page in range((len(measured)+4)//5):
        sheet = Image.new('RGB', (960, 5*208), (24, 24, 24)); draw = ImageDraw.Draw(sheet)
        for rr, p in enumerate(measured[page*5:page*5+5]):
            for col, ob in enumerate(p['reviewed_observations']):
                camera = ob['camera']; u, v = ob['detected_uv']; x, y = np.rint([u-.5, v-.5]).astype(int)
                crop = Image.open(evidence[camera]['rgb']).convert('RGB').crop((x-96, y-86, x+96, y+86))
                crop = crop.resize((288, 172)); d = ImageDraw.Draw(crop)
                d.line((139, 86, 149, 86), fill='red'); d.line((144, 81, 144, 91), fill='red')
                sheet.paste(crop, (col*320, rr*208+32))
                draw.text((col*320+2, rr*208+2), f"p{p['point']} {camera} {p['region']}", fill='white')
        sheet.save(output/f'review_{page+1:02d}.jpg', quality=96)
    review['atlases'] = {p.name: sha256(p) for p in sorted(output.glob('review_*.jpg'))}
    write(output/'visibility_review.json', review)
    semantic_hashes = {camera: records[camera]['semantic_sha256'] for camera in sorted(evidence)}
    oldmeta = json.loads((prior/'manifest.json').read_text())
    semantic_hashes.update(oldmeta['source_semantic_hashes'])
    bindings = dict(transforms_sha256=sha256(data/'transforms.json'), box_sha256=sha256(data/'box.json'),
        tracks_sha256=sha256(tracks_path), witnesses_sha256=sha256(witness_path),
        semantic_source_sha256=hashlib.sha256(json.dumps(semantic_hashes, sort_keys=True).encode()).hexdigest(),
        visibility_review_sha256=sha256(output/'visibility_review.json'))
    manifest = dict(kind='reviewed_train_clothing_visible_prefixes', coordinate_system='accepted_floor_world',
        uses_eval_supervision=False, original_observed_track_graph_only=True, manual_clothing_visibility_review=True,
        distance_scale=1., aabb=bounds.tolist(), bindings=bindings, archive='prefixes.npz',
        archive_sha256=sha256(output/'prefixes.npz'), source_semantic_hashes=semantic_hashes,
        original_supervised_points=9, original_supervised_rays=18,
        new_reviewed_points=len(REGIONS), new_supervised_points=len(REGIONS)-len(reserved),
        supervised_points=len(np.unique(arrays['point_idx'])), supervised_rays=len(arrays['point_idx']),
        reserved_new_whole_patch_points=reserved, preserved_old_reserved_points=[63, 64, 71, 72],
        approved_for_training=False, pending=['Root source/atlas review', 'Actual native AMP gradient gate for each arm',
                                             'Separate frozen SH+prefix integration and durable save/resume gate'],
        source_files_sha256={str(p): sha256(p) for p in sorted(set(source_files))})
    write(output/'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ['new_reviewed_points', 'new_supervised_points', 'supervised_points', 'supervised_rays', 'approved_for_training']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); prepare(args.campaign, args.output)
