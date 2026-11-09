"""Potential shared-node TRAIN exposure; never an EVAL density contributor trace."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from marshawn_frozen_semantic_head import FrozenActorGrid
from prepare_marshawn_frozen_semantic_cpu import sha, write


def nodes(points, bounds, resolution=128):
    q = (points-bounds[0])/(bounds[1]-bounds[0])*(resolution-1)
    lower = np.clip(np.floor(q).astype(np.int64), 0, resolution-2)
    fraction = np.clip(q-lower, 0, 1)
    for corner in itertools.product([0, 1], repeat=3):
        corner = np.asarray(corner)
        index = lower+corner
        basis = np.prod(np.where(corner, fraction, 1-fraction), axis=1)
        yield (index[:, 0]*resolution+index[:, 1])*resolution+index[:, 2], basis


def run(c, output):
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    output.mkdir(parents=True, exist_ok=False)
    mp = c/'frozen_native_semantic_CPU_v2/manifest.json'
    cp = c/'frozen_native_semantic_capture_plan_v2/capture/capture.json'
    hp = c/'frozen_native_semantic_prior001_candidate_v1/runs/head2064/head.pt'
    m, capture = json.loads(mp.read_text()), json.loads(cp.read_text())
    state = torch.load(hp, map_location='cpu', weights_only=False)
    head = FrozenActorGrid(m['bounds'], m['native_world_to_box'], initial_probability=.01)
    head.load_state_dict(state['head'], strict=True)
    bounds, transform = np.asarray(m['bounds']), np.asarray(m['native_world_to_box'])
    frames = {r['physical_camera']: r for r in json.loads((Path(m['native_data'])/'transforms.json').read_text())['frames']}
    cases = [('HC_isolated_fleck', 'HC', [875, 602]), ('DC_gray_shoe_bag_edge', 'DC', [1097, 572])]
    result = []
    for name, cam, pixel in cases:
        f = frames[cam]; pose = np.asarray(f['transform_matrix'])
        local = np.asarray([(pixel[0]+.5-f['cx'])/f['fl_x'], -(pixel[1]+.5-f['cy'])/f['fl_y'], -1.])
        direction = pose[:3, :3]@local; direction /= np.linalg.norm(direction)
        origin = pose[:3, 3]
        origin_floor = transform[:3, :3]@origin+transform[:3, 3]
        direction_floor = transform[:3, :3]@direction
        crossings = (bounds-origin_floor)/direction_floor
        near, far = crossings.min(0).max(), crossings.max(0).min()
        assert 0 < near < far
        spacing = float(np.min((bounds[1]-bounds[0])/127)/4)
        distance = np.linspace(near+1e-7, far-1e-7, int(np.ceil((far-near)/spacing))+1)
        world = origin+distance[:, None]*direction
        with torch.no_grad(): probability = head(torch.as_tensor(world, dtype=torch.float32)).numpy()
        floor = world@transform[:3, :3].T+transform[:3, 3]
        high = probability >= .5
        candidate = set()
        for index, basis in nodes(floor[high], bounds): candidate.update(index[basis > 1e-8].tolist())
        lookup = np.zeros(128**3, bool); lookup[list(candidate)] = True
        intervals = []
        starts = np.flatnonzero(high & ~np.r_[False, high[:-1]])
        ends = np.flatnonzero(high & ~np.r_[high[1:], False])
        for start, end in zip(starts, ends):
            intervals.append(dict(distance=[float(distance[start]), float(distance[end])], floor_xyz=[floor[start].tolist(), floor[end].tolist()]))
        result.append(dict(name=name, camera=cam, pixel_xy=pixel, sampled_classifier_points=len(distance),
            classifier_only_p_range=[float(probability.min()), float(probability.max())],
            classifier_hard05_ray_intervals=intervals, potential_node_count=len(candidate), _lookup=lookup,
            _positive=[], _negative=[], visible_positive_exposure=0., visible_negative_exposure=0.))
    source_hashes = {str(p): sha(p) for p in [mp, cp, hp, Path(m['native_data'])/'transforms.json']}
    global_offset = 0
    for row in capture['records']:
        if not row['rays']: continue
        archive = Path(row['archive']); assert sha(archive) == row['archive_sha256']
        source_hashes[str(archive)] = row['archive_sha256']
        with np.load(archive) as z:
            xyz, weight, ids = z['xyz'], z['weight'].reshape(-1), z['ray_indices'].astype(int)
            target, confidence, pixels = z['semantic_target'].copy(), z['label_weight'].copy(), z['pixel_xy']
        if row['camera'] == 'IA':
            for pixel in [[1019, 567], [1013, 562]]: confidence[np.all(pixels == pixel, axis=1)] = 0
        floor = xyz@transform[:3, :3].T+transform[:3, 3]
        inside = ((floor >= bounds[0]) & (floor <= bounds[1])).all(1)
        keep = inside & (weight > 1e-8) & (confidence[ids] > 0)
        floor, w, ray = floor[keep], weight[keep], ids[keep]
        for case in result:
            exposure = np.zeros(len(target), np.float64)
            for index, basis in nodes(floor, bounds):
                exposure += np.bincount(ray, weights=w*basis*case['_lookup'][index], minlength=len(target))
            for positive, key in [(True, '_positive'), (False, '_negative')]:
                chosen = np.flatnonzero((exposure > 0) & ((target == 1) if positive else (target == 0)))
                for index in chosen:
                    case[key].append(dict(camera=row['camera'], pixel_xy=pixels[index].tolist(),
                        camera_ray_id=int(index), global_query_id=global_offset+int(index),
                        potential_native_visibility_basis_exposure=float(exposure[index])))
                case['visible_positive_exposure' if positive else 'visible_negative_exposure'] += float(exposure[chosen].sum())
        global_offset += row['rays']
    panels = []
    for case in result:
        del case['_lookup']
        for key in ['_positive', '_negative']:
            values = sorted(case.pop(key), key=lambda r: -r['potential_native_visibility_basis_exposure'])
            case['positive_query_count' if key == '_positive' else 'negative_query_count'] = len(values)
            case['top20_positive_queries' if key == '_positive' else 'top20_negative_queries'] = values[:20]
        panel = Image.new('RGB', (1200, 500), '#181818'); draw = ImageDraw.Draw(panel)
        draw.text((5, 5), case['name']+': potential shared-node TRAIN positive exposure, NOT actual EVAL contributors', fill='white')
        for i, record in enumerate(case['top20_positive_queries'][:10]):
            frame = frames[record['camera']]; path = Path(frame['file_path']); source_hashes[str(path)] = sha(path)
            x, y = record['pixel_xy']; image = Image.open(path).convert('RGB').crop((x-20, y-20, x+21, y+21)).resize((160, 160), Image.Resampling.NEAREST)
            ImageDraw.Draw(image).rectangle((77, 77, 82, 82), outline='red')
            xx, yy = (i%5)*240, 35+(i//5)*230
            panel.paste(image, (xx, yy)); draw.text((xx, yy+165), record['camera']+' '+str([x,y])+' id'+str(record['global_query_id']), fill='white')
            draw.text((xx, yy+182), 'exposure %.6g'%record['potential_native_visibility_basis_exposure'], fill='white')
        panel.save(output/(case['name']+'_potential_TRAIN_support.png')); panels.append(panel)
    write(output/'potential_exposure.json', dict(records=result, source_sha256=source_hashes, helper_sha256=sha(__file__),
        CPU_only=True, CUDA_initialized=torch.cuda.is_initialized(), density_queries=0, labels_changed=False,
        interpretation='Classifier-only ray samples at min-node-pitch/4. Candidate nodes touch any sampled p>=.5 interval; sums native TRAIN w*trilinear-basis for all such nodes. Positive/negative exposure is NOT historical gradient, actual EVAL density contribution, or cross-view visibility. EVAL sigma/depth traces absent. Nativew<=1e-8 ignored for numerical exposure reporting only; no fitting change. Geometry/occlusion and node sharing can explain support behind visible rig/bag; source photos do not qualify whole-ray negatives.'))
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--campaign', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.campaign, args.output)
