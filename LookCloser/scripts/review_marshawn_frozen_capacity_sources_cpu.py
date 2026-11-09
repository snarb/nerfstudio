"""Table and calibrated source-only HD/RAW clusters for native-capacity failures."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from marshawn_actor_coordinates import raw_array_indices
from prepare_marshawn_frozen_semantic_cpu import sha, write


def review(campaign, capacity, output):
    report = json.loads(capacity.read_text())
    output.mkdir(parents=True, exist_ok=False)
    iso = campaign.parent/'two_actor_isolation'
    request_path = iso/'raw_native_sources/request.json'
    sources = {r['camera']: r for r in json.loads(request_path.read_text())['records']}
    frames = {r['physical_camera']: r for r in json.loads((campaign/'data/transforms.json').read_text())['frames']}
    source_manifest = campaign/'native_semantic_grid_20000/occlusion_v2/W64/support_manifest.json'
    support = {r['camera']: r for r in json.loads(source_manifest.read_text())['records']}
    rows, pages, tiles = [], [], []
    bindings = {str(p): sha(p) for p in [capacity, request_path, campaign/'data/transforms.json', source_manifest]}
    for name in sorted({r['camera'] for r in report['bad_positives']}):
        records = [r.copy() for r in report['bad_positives'] if r['camera'] == name]
        xy = np.array([r['pixel'] for r in records])
        sr, frame, source = support[name], frames[name], sources[name]
        with np.load(sr['source_archive']) as z:
            nominal = z['nominal_coverage']
        with np.load(sr['archive']) as z:
            unknown = z['unknown']
        semantic_path = campaign/'data/semantics'/f'{name}.npz'
        with np.load(semantic_path) as z:
            black, grey = z['black'], z['grey']
        raw_path = Path(source['image'])
        assert sha(raw_path) == source['image_sha256']
        hd_path = campaign/'data'/frame['file_path']
        for p in [Path(sr['source_archive']), Path(sr['archive']), semantic_path, hd_path, raw_path]:
            bindings[str(p)] = sha(p)
        hd = Image.open(hd_path).convert('RGB')
        raw = Image.open(raw_path).convert('RGB')
        raw_centers = raw_array_indices(xy+.5, frame)
        edges = cKDTree(xy).query_pairs(20, output_type='ndarray')
        if len(edges):
            graph = coo_matrix((np.ones(2*len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(len(xy), len(xy)))
            _, labels = connected_components(graph, directed=False)
        else:
            labels = np.arange(len(xy))
        for index, record in enumerate(records):
            x, y = record['pixel']
            record.update(category='actor_interior', semantic_actor='black' if black[y, x] else 'grey' if grey[y, x] else 'neither',
                native_nominal_coverage=float(nominal[y, x]), existing_UNKNOWN=bool(unknown[y, x]),
                RAW_array_center=raw_centers[index].tolist(), cluster=int(labels[index])+1,
                irreducible_conditional_NLL_lower_bound=float(-np.log(max(record['ratio'], 1e-30))),
                qualification='PENDING_SOURCE_REVIEW; genuine actor queries must remain, not filtered by capacity')
            rows.append(record)
        for group in np.unique(labels):
            ids = np.flatnonzero(labels == group)
            pixels = xy[ids]
            hd_box = np.r_[pixels.min(0)-24, pixels.max(0)+25].astype(int)
            centers = raw_centers[ids]-np.array(source['crop_xyxy'][:2])
            raw_box = np.r_[np.floor(centers.min(0)-95), np.ceil(centers.max(0)+96)].astype(int)
            h = hd.crop(tuple(hd_box)); hdraw = ImageDraw.Draw(h)
            raw_crop = raw.crop(tuple(raw_box)); rdraw = ImageDraw.Draw(raw_crop)
            for rank, index in enumerate(ids):
                x, y = xy[index]-hd_box[:2]
                hdraw.ellipse((x-2, y-2, x+2, y+2), outline='red', width=1)
                rx, ry = raw_centers[index]-np.array(source['crop_xyxy'][:2])-raw_box[:2]
                rdraw.ellipse((rx-5, ry-5, rx+5, ry+5), outline='red', width=2)
                rdraw.text((rx+6, ry-6), str(rank+1), fill='yellow')
            panel = Image.new('RGB', (880, 490), '#171717'); draw = ImageDraw.Draw(panel)
            lo = min(records[index]['ratio'] for index in ids)
            draw.text((5, 4), f'{name} cluster{group+1} n={len(ids)} cap/A min={lo:.6g}; SOURCE ONLY / NO LABEL EDITS', fill='white')
            for j, (title, tile) in enumerate([('HD', h), ('Native RAW (red=current positive query)', raw_crop)]):
                scale = min(430/tile.width, 430/tile.height)
                tile = tile.resize((max(1, round(tile.width*scale)), max(1, round(tile.height*scale))), Image.Resampling.NEAREST)
                panel.paste(tile, (j*440+(440-tile.width)//2, 50)); draw.text((j*440+4, 25), title, fill='white')
            page = output/f'{name}_cluster{group+1:02d}.jpg'; panel.save(page, quality=93)
            pages.append(dict(camera=name, cluster=int(group)+1, pixels=pixels.tolist(), page=str(page), page_sha256=sha(page), cap_ratio_min=lo))
            tile = panel.copy(); tile.thumbnail((440, 245)); tiles.append(tile)
    for first in range(0, len(tiles), 12):
        chunk = tiles[first:first+12]; sheet = Image.new('RGB', (4*440, 3*245), '#171717')
        for i, tile in enumerate(chunk): sheet.paste(tile, ((i%4)*440, (i//4)*245))
        sheet.save(output/f'overview_{first//12+1:02d}.jpg', quality=92)
    fields = ['camera', 'cluster', 'x', 'y', 'category', 'semantic_actor', 'native_nominal_coverage', 'native_alpha', 'insideboxcap', 'ratio', 'irreducible_conditional_NLL_lower_bound', 'existing_UNKNOWN', 'qualification']
    with (output/'capacity136.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for r in rows:
            row = {k: r[k] for k in fields if k not in ['x', 'y']}; row.update(x=r['pixel'][0], y=r['pixel'][1]); writer.writerow(row)
    write(output/'review.json', dict(records=rows, pages=pages, source_sha256=bindings, helper_sha256=sha(__file__),
        CPU_only=True, labels_changed=False, fitting_started=False,
        geometry_qualification='cap/A is frozen geometry/visibility capacity, not source-label validity. Keep genuine actors unchanged and report irreducible NLL; do not silently filter or normalize to cap.'))
    print(json.dumps(dict(rows=len(rows), cameras=len({r['camera'] for r in rows}), clusters=len(pages), output=str(output))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True); p.add_argument('--capacity', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); review(a.campaign, a.capacity, a.output)
