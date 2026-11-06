"""Find measured TRAIN background rays whose surface lies beyond the actor box.

Read-only diagnosis of full-image RGB supervision in a restricted field domain.
Recovered unique original features are used, never projected UV as observations.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
from audit_marshawn_geometry import projection


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('/fsx/tmp/marshawn_001660_hd')
    campaign = root/'tuning_20261106'
    data = campaign/'data'
    output = campaign/'bounded_background_audit'
    output.mkdir(exist_ok=False)
    original = json.loads((root/'data/transforms.json').read_text())
    prepared = json.loads((data/'transforms.json').read_text())
    source_frames = {f['physical_camera']: f for f in original['frames']}
    frames = {f['physical_camera']: f for f in prepared['frames']}
    tracks = json.loads((root/'geometry/strict_tracks.json').read_text())
    with np.load(root/'geometry/train_witnesses.npz') as archive:
        points, cameras = archive['xyz'].copy(), archive['cameras'].tolist()
    box = json.loads((data/'box.json').read_text())
    transform = np.asarray(box['world_to_box'])
    bounds = np.asarray(box['aabb'])
    floor = points@transform[:3, :3].T + transform[:3, 3]
    records, summaries, source_hashes = [], [], {}
    for camera in ['AA', 'AB', 'HE', 'IC', 'JA', 'MC', 'NE', 'OA']:
        f, old = frames[camera], source_frames[camera]
        assert f['file_path'] in prepared['train_filenames']
        assert np.allclose(transform@np.asarray(old['transform_matrix']), f['transform_matrix'], atol=1e-7)
        c = cameras.index(camera)
        ids = np.asarray([i for i, cs in enumerate(tracks['camera_ids']) if c in cs and len(cs) >= 3])
        proj = projection(old)[0]
        q = np.c_[points[ids], np.ones(len(ids))]@proj.T
        uv = q[:, :2]/q[:, 2, None]
        feature_path = root/'geometry/features'/f'{camera}.npz'
        with np.load(feature_path) as archive:
            features = archive['uv'].astype(float)
            assert str(archive['sha']) == sha(root/'data'/old['file_path'])
        tree = cKDTree(features)
        candidates = tree.query_ball_point(uv, 1.00001)
        with np.load(data/'semantics'/f'{camera}.npz') as archive:
            weights = archive['W64'].copy()
            valid = archive['valid'].copy()
        pose = np.asarray(f['transform_matrix'])
        origin = pose[:3, 3]
        rgb = np.asarray(Image.open(data/f['file_path']).convert('RGB'))
        found = []
        for pid, predicted, near in zip(ids, uv, candidates):
            measured = np.unique(features[near], axis=0)
            if len(measured) != 1:
                continue
            observed = measured[0]
            x, y = np.floor(observed).astype(int)
            if not (0 <= x < f['w'] and 0 <= y < f['h'] and valid[y, x] and weights[y, x] == 0):
                continue
            ray = pose[:3, :3]@np.array([(x+.5-f['cx'])/f['fl_x'], -(y+.5-f['cy'])/f['fl_y'], -1.])
            ray /= np.linalg.norm(ray)
            if np.any(np.abs(ray) < 1e-10):
                continue
            left, right = (bounds[0]-origin)/ray, (bounds[1]-origin)/ray
            enter = max(float(np.minimum(left, right).max()), .01)
            leave = float(np.maximum(left, right).min())
            depth = float((floor[pid]-origin)@ray)
            miss = float(np.linalg.norm(floor[pid]-origin-depth*ray))
            if leave-enter <= .05 or depth-leave <= .03 or miss > .003 or rgb[y, x].mean() < 25:
                continue
            found.append(dict(camera=camera, point_id=int(pid), source_track_views=len(tracks['camera_ids'][pid]),
                              measured_uv_colmap=observed.tolist(), pixel_xy=[int(x), int(y)],
                              fitted_reprojection_error=float(np.linalg.norm(observed-predicted)),
                              world_point=floor[pid].tolist(), origin=origin.tolist(), direction=ray.tolist(),
                              box_interval=[enter, leave], measured_surface_ray_depth=depth,
                              beyond_box_gap=depth-leave, point_to_pixel_ray_distance=miss,
                              frozen_actor_weight=0., rgb_code=rgb[y, x].tolist()))
        found.sort(key=lambda r: -r['beyond_box_gap'])
        # Spatially separate examples so one printed edge is not repeated.
        chosen = []
        for row in found:
            if all(np.linalg.norm(np.array(row['pixel_xy'])-r['pixel_xy']) > 80 for r in chosen):
                chosen.append(row)
            if len(chosen) == 4:
                break
        records.extend(chosen)
        summaries.append(dict(camera=camera, qualified_rays=len(found), displayed=len(chosen)))
        source_hashes[str(feature_path)] = sha(feature_path)
        for row in chosen:
            x, y = row['pixel_xy']
            patch = Image.fromarray(rgb).crop((x-64, y-64, x+64, y+64)).resize((256, 256))
            draw = ImageDraw.Draw(patch)
            draw.line((120, 128, 136, 128), fill='red', width=1)
            draw.line((128, 120, 128, 136), fill='red', width=1)
            patch.save(output/f"{camera}_{row['point_id']}.png")
    sheet = Image.new('RGB', (4*280, ((len(records)+3)//4)*290), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(records):
        x, y = (i % 4)*280, (i//4)*290
        sheet.paste(Image.open(output/f"{row['camera']}_{row['point_id']}.png"), (x, y+30))
        draw.text((x+3, y+3), f"{row['camera']} p{row['point_id']} gap {row['beyond_box_gap']:.3f}", fill='black')
    sheet.save(output/'background_source_examples.jpg', quality=94)
    for path in [data/'transforms.json', data/'box.json', root/'geometry/train_witnesses.npz', root/'geometry/strict_tracks.json', Path(__file__)]:
        source_hashes[str(path)] = sha(path)
    result = dict(kind='TRAIN_MEASURED_BACKGROUND_BEYOND_BOUNDED_FIELD_DIAGNOSTIC',
                  summaries=summaries, examples=records, source_hashes=source_hashes,
                  limitation='Sparse source diagnosis, not a measured fraction of all training rays. Does not establish the cause of every artifact.',
                  implication='Full RGB reconstruction supervises these nonblack pixels despite zero actor confidence; their measured surface is outside the bounded field. Actor-weighted objective gives these rays zero RGB weight.',
                  training_or_data_modified=False)
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
