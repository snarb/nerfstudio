"""Locate source TRAIN visibility supporting an isolated semantic-grid cell.

Reports shared interpolation support only. It assigns no new labels, physical
opacity, hidden-space truth, or historical optimizer causality.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from audit_marshawn_semantic_residual_exposure_cpu import nodes
from marshawn_frozen_semantic_head import FrozenActorGrid, tensor_digest
from prepare_marshawn_frozen_semantic_cpu import sha


def run(root, output):
    torch.set_num_threads(2)
    inventory = root / 'semantic_cell_candidates_cpu_v1/component_floor_inventory_v1/active_r0_low_cells.json'
    chosen = max(json.loads(inventory.read_text()), key=lambda r: r['CPU_head_center_p'])
    point = np.asarray(chosen['center'], np.float32)
    contract = json.loads((root / 'frozen_LC_semantic_head_plan_v1/head_input_contract.json').read_text())
    cp = root / 'frozen_LC_semantic_head_archives/recovery1_20261107_v1/runs/prior001_fresh2064/head.pt'
    if sha(cp) != 'bdb2cc0d8138e4c22c61b4d411227b272b95ceac9e8295d474eecc4a84aa98bc':
        raise ValueError('Exact original two-UNKNOWN head required')
    saved = torch.load(cp, map_location='cpu', weights_only=False)
    head = FrozenActorGrid(contract['bounds'], np.eye(4), 128, initial_probability=.01)
    head.load_state_dict(saved['head'], strict=True)
    head.requires_grad_(False)
    before = tensor_digest(head)
    bounds = head.bounds.numpy()
    kernel = np.zeros(128 ** 3, np.float64)
    for index, basis in nodes(point[None], bounds):
        kernel[int(index[0])] += float(basis[0])
    capture_path = root / 'frozen_LC_semantic_capture_plan_v4/capture/complete.json'
    capture = json.loads(capture_path.read_text())
    hashes = {str(p): sha(p) for p in [inventory, cp, capture_path]}
    rows = []
    offset = 0
    for record in capture['records']:
        count = record['rays']
        if not count:
            continue
        path = Path(record['archive'])
        if sha(path) != record['archive_sha256']:
            raise ValueError('Native contribution cache changed')
        hashes[str(path)] = record['archive_sha256']
        with np.load(path, allow_pickle=False) as arrays:
            xyz = arrays['xyz']
            overlap = np.zeros(len(xyz), np.float64)
            for index, basis in nodes(xyz, bounds):
                overlap += basis * kernel[index]
            overlap *= ((xyz >= bounds[0]) & (xyz <= bounds[1])).all(-1)
            ids = arrays['ray_indices'].astype(np.int64)
            mass = np.bincount(ids, weights=arrays['weight'] * overlap, minlength=count)
            for index in np.flatnonzero(mass > 1e-12):
                rows.append(dict(camera=record['camera'], local_query=int(index),
                    global_query=offset + int(index), pixel_xy=arrays['pixel_xy'][index].tolist(),
                    semantic_target=float(arrays['semantic_target'][index]),
                    confidence=float(arrays['label_weight'][index]),
                    native_visible_shared_node_exposure=float(mass[index])))
        offset += count
    if offset != 69888 or tensor_digest(head) != before or torch.cuda.is_initialized():
        raise ValueError('Immutable CPU support audit failed')
    frames = {r['physical_camera']: r for r in json.loads((root / 'data/transforms.json').read_text())['frames']}
    output.mkdir(parents=True, exist_ok=False)
    for label, target in [('positive', 1), ('negative', 0)]:
        selected = sorted([r for r in rows if r['semantic_target'] == target and r['confidence'] > 0],
                          key=lambda r: -r['native_visible_shared_node_exposure'])
        panel = Image.new('RGB', (1400, 540), '#171717')
        draw = ImageDraw.Draw(panel)
        draw.text((4, 4), label + ' source TRAIN support for isolated low cell; no new labels', fill='white')
        for i, row in enumerate(selected[:10]):
            path = root / 'data' / frames[row['camera']]['file_path']
            hashes[str(path)] = sha(path)
            x, y = row['pixel_xy']
            patch = Image.open(path).convert('RGB').crop((x-24, y-24, x+25, y+25)).resize((196, 196), Image.Resampling.NEAREST)
            ImageDraw.Draw(patch).rectangle((94, 94, 101, 101), outline='red')
            xx, yy = i % 5 * 280, 30 + i // 5 * 255
            panel.paste(patch, (xx, yy))
            draw.text((xx, yy+198), str({k: row[k] for k in ['camera', 'pixel_xy', 'global_query']}), fill='white')
            draw.text((xx, yy+225), 'w*b %.6g' % row['native_visible_shared_node_exposure'], fill='white')
        panel.save(output / (label + '_TRAIN_support.png'))
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Source changed: ' + path)
    report = dict(CPU_only=True, chosen_cell=chosen, point=point.tolist(),
        actual_CPU_probability=float(head(torch.from_numpy(point[None]))[0]),
        source_queries=sorted(rows, key=lambda r: -r['native_visible_shared_node_exposure']),
        source_hashes=hashes, new_labels=False, physical_opacity_or_empty_space_claim=False,
        historical_optimizer_causality_claim=False, head_unchanged=True)
    (output / 'support.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'shared_queries': len(rows), 'point_p': report['actual_CPU_probability']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
