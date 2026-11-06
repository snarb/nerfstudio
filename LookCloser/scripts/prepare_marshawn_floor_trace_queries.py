"""Predeclare bounded foot-ROI diagnostics and known TRAIN shoe controls, without field queries."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from audit_marshawn_tuning_checkpoint import ray_for_pixel


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def interval(ray, bounds):
    origin, direction = np.asarray(ray['origin']), np.asarray(ray['direction'])
    near, far = 0., np.inf
    for axis in range(3):
        if direction[axis] == 0:
            if not bounds[0, axis] <= origin[axis] <= bounds[1, axis]:
                return None
        else:
            values = (bounds[:, axis] - origin[axis]) / direction[axis]
            near, far = max(near, float(values.min())), min(far, float(values.max()))
    return [near, far] if far > near else None


def main(a):
    ready = json.loads((a.data / 'ready.json').read_text())
    frames = {row['physical_camera']: row for row in json.loads((a.data / 'transforms.json').read_text())['frames']}
    rows = {row['camera']: row for row in ready['records']}
    assert sha(a.data / 'transforms.json') == ready['transforms_sha256']
    bounds = np.asarray(ready['aabb'])
    regions_path = a.native_base / 'corrected_v2/review/region_crops.json'
    regions = json.loads(regions_path.read_text())
    upper_path = a.native_base / 'corrected_v2/train_shoe_feature_depth.json'
    sole_path = a.native_base / 'mat_geometry_TRAIN/qualified_material_points_v2.json'
    upper = json.loads(upper_path.read_text())['points'][0]
    sole = next(row for row in json.loads(sole_path.read_text())['records'] if row['source_point_id'] == 6862)
    order = json.loads(a.clothing_queries.read_text())['camera_order']
    assert len(order) == 63 and not set(order) & {'DC', 'HC', 'KA'}
    a.output.mkdir(parents=True, exist_ok=False)
    source_bindings = {str(path): sha(path) for path in [a.data / 'ready.json', a.data / 'transforms.json', regions_path, upper_path, sole_path, a.clothing_queries, Path(__file__)]}
    prepared = []
    for split, cameras in [('eval', ['DC', 'HC', 'KA']), ('train', order)]:
        candidates, pixels, rays, excluded = [], [], [], []
        if split == 'eval':
            for camera in cameras:
                x0, y0, x1, y1 = regions[camera]['feet']
                for iy, fy in enumerate([.25, .5, .75]):
                    for ix, fx in enumerate([.15, .5, .85]):
                        candidates.append(dict(camera_name=camera, x=int(x0+fx*(x1-x0)), y=int(y0+fy*(y1-y0)),
                            region='fixed_foot_ROI_%d_%d' % (iy, ix), actor='unclassified_ROI',
                            semantic='diagnostic_only_no_material_or_depth_label'))
        else:
            for source in upper['observations']:
                x, y = np.rint(source['array_xy']).astype(int)
                candidates.append(dict(camera_name=source['camera'], x=int(x), y=int(y), region='toe_upper',
                    actor='black', witness_xyz=upper['xyz_floor'], witness_id='three_view_upper_toe',
                    source_coordinate=source['array_xy'], source_coordinate_contract='array pixel center; nearest integer index'))
            for source in sole['source_graph']['observations']:
                x, y = np.floor(source['detected_uv']).astype(int)
                candidates.append(dict(camera_name=source['camera'], x=int(x), y=int(y), region='sole6862',
                    actor='black', witness_xyz=sole['xyz_floor'], witness_id=6862,
                    source_coordinate=source['detected_uv'], source_coordinate_contract='COLMAP half-center UV; floor to integer pixel index'))
        for pixel in candidates:
            name = pixel['camera_name']
            ray = ray_for_pixel(frames[name], pixel['x'], pixel['y'])
            hit = interval(ray, bounds)
            if hit is None:
                excluded.append(dict(pixel=pixel, reason='ray misses accepted AABB; no empty-space inference'))
                continue
            pixel.update(camera=cameras.index(name), query_id=len(pixels), accepted_AABB_interval=hit)
            if 'witness_xyz' in pixel:
                delta = np.asarray(pixel['witness_xyz']) - ray['origin']
                depth = float(delta @ np.asarray(ray['direction']))
                pixel['query_witness_depth'] = depth
                pixel['point_to_pixel_ray_distance'] = float(np.linalg.norm(delta - depth*np.asarray(ray['direction'])))
            pixels.append(pixel); rays.append(ray)
        request = dict(split=split, camera_order=cameras, pixels=pixels, rays=rays,
            ready_sha256=sha(a.data / 'ready.json'), transforms_sha256=sha(a.data / 'transforms.json'),
            excluded_no_AABB_intersection=excluded, source_bindings=source_bindings,
            no_EVAL_fitting=True, no_new_masks=True, no_field_or_GPU_queries=True,
            limitation='Posthoc fixed-ROI diagnostic plus qualified TRAIN shoe points. No near-foot floor depth or physical empty-space labels.')
        path = a.output / (split + '_queries.json')
        path.write_text(json.dumps(request, indent=2) + '\n')
        prepared.append(dict(split=split, path=str(path), sha256=sha(path), rays=len(pixels), excluded=len(excluded)))
        for name in sorted(set(p['camera_name'] for p in pixels)):
            row = rows[name]; source = a.data / row['image']
            assert sha(source) == row['rgb_sha256']
            image = Image.open(source).convert('RGB'); draw = ImageDraw.Draw(image)
            for pixel in [p for p in pixels if p['camera_name'] == name]:
                x,y=pixel['x'],pixel['y']; draw.ellipse((x-4,y-4,x+4,y+4),outline='red',width=2)
                draw.text((x+5,y-10),str(pixel['query_id']),fill='yellow')
            if split == 'eval':
                crop = regions[name]['feet']
            else:
                q = [p for p in pixels if p['camera_name'] == name]
                crop = (max(0,min(p['x'] for p in q)-100),max(0,min(p['y'] for p in q)-100),
                        min(image.width,max(p['x'] for p in q)+101),min(image.height,max(p['y'] for p in q)+101))
            image.crop(crop).save(a.output / (split+'_'+name+'_source_queries.png'))
    assert sum(row['rays'] for row in prepared) <= 64
    (a.output / 'manifest.json').write_text(json.dumps(dict(requests=prepared, source_bindings=source_bindings,
        total_rays=sum(row['rays'] for row in prepared), GPU_trace_launched=False),indent=2)+'\n')
    print(json.dumps(prepared))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','native-base','clothing-queries','output']:
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
