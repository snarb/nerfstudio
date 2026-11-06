"""Export named boolean XYZ grids as a self-contained, offline slice inspector.

Input JSON: {"title": "...", "coordinate_frame": "accepted_floor_world",
 "layers": [{"name": "active", "archive": "/path/grid.npz", "key": "active",
             "bounds_key": "cascade_aabbs"}]}
All layers must have exactly the same cell boundaries and declared frame.
This visualizes measured/predicted states; it does not certify physical emptiness.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np


def load_layers(spec):
    if not spec.get('coordinate_frame'):
        raise ValueError('Declare the coordinate frame explicitly')
    layers = []
    reference = None
    for entry in spec['layers']:
        path = Path(entry['archive']).resolve()
        with np.load(path, allow_pickle=False) as archive:
            grid = archive[entry['key']]
            bounds = archive[entry.get('bounds_key', 'cascade_aabbs')]
            if grid.ndim == 4 and grid.shape[0] == 1:
                grid = grid[0]
            if grid.ndim != 3 or grid.dtype != np.bool_ or min(grid.shape) < 1:
                raise ValueError('Expected a boolean XYZ grid or a single cascade')
            if bounds.shape not in ((2, 3), (1, 6), (6,)):
                raise ValueError('Expected exactly one six-coordinate bounding box')
            bounds = bounds.astype(np.float64).reshape(2, 3)
            if not np.isfinite(bounds).all() or not (bounds[1] > bounds[0]).all():
                raise ValueError('Invalid bounds')
            if entry.get('coordinate_frame', spec['coordinate_frame']) != spec['coordinate_frame']:
                raise ValueError('Coordinate frames differ')
            if reference is not None:
                if grid.shape != reference[0] or not np.array_equal(bounds, reference[1]):
                    raise ValueError('Cell boundaries differ; explicitly resample before comparison')
            reference = (grid.shape, bounds.copy())
            layers.append(dict(name=entry['name'], archive=str(path), key=entry['key'],
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               occupied=int(grid.sum()),
                               bits=base64.b64encode(np.packbits(grid.ravel(order='C'), bitorder='big')).decode()))
    if not layers:
        raise ValueError('No layers')
    if len({row['name'] for row in layers}) != len(layers):
        raise ValueError('Layer names must be unique')
    landmarks = []
    for point in spec.get('landmarks', []):
        xyz = np.asarray(point['xyz'], dtype=np.float64)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError('Landmark requires three finite coordinates')
        if point.get('coordinate_frame', spec['coordinate_frame']) != spec['coordinate_frame']:
            raise ValueError('Landmark coordinate frame differs')
        inside = bool(((xyz >= reference[1][0]) & (xyz <= reference[1][1])).all())
        index = np.floor((xyz-reference[1][0])/(reference[1][1]-reference[1][0])*reference[0]).astype(int)
        landmarks.append(dict(label=point['label'], xyz=xyz.tolist(), inside=inside,
                              index=np.minimum(index, np.array(reference[0])-1).tolist() if inside else None,
                              provenance=point.get('provenance', 'Unspecified; not ground truth')))
    return dict(title=spec.get('title', 'Occupancy grid inspector'),
                coordinate_frame=spec['coordinate_frame'], shape=list(reference[0]),
                bounds=reference[1].tolist(), layers=layers, landmarks=landmarks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = load_layers(json.loads(args.spec.read_text()))
    template = Path(__file__).with_name('occupancy_grid_viewer.html').read_text()
    # Prevent labels or paths from terminating the JSON script element.
    encoded = json.dumps(payload, ensure_ascii=True).replace('<', '\\u003c')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace('GRID_PAYLOAD', encoded))
    receipt = {k: v for k, v in payload.items() if k != 'layers'}
    receipt['layers'] = [{k: v for k, v in row.items() if k != 'bits'} for row in payload['layers']]
    receipt['viewer_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'shape': payload['shape'],
                      'layers': [(x['name'], x['occupied']) for x in payload['layers']]}))


if __name__ == '__main__':
    main()
