"""CPU-only source review proposal; does not produce accepted training targets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


POLYGONS = {
    'EA': [[790, 145], [825, 145], [825, 175], [790, 175]],
    'JA': [[1040, 178], [1075, 178], [1075, 210], [1040, 210]],
    'LA': [[1153, 185], [1170, 185], [1170, 215], [1153, 215]],
    'NA': [[920, 245], [944, 245], [944, 254], [920, 254]],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    c = args.campaign
    original = c / 'native_resolution_pair_100000/documentation/clear_background_polygon_proposal_v1/proposal.json'
    support_path = c / 'semantic_actor_candidate_v2_20261106/source/support_manifest.json'
    sources = {r['camera']: r for r in json.loads(support_path.read_text())['records']}
    ready = json.loads((c / 'data/ready.json').read_text())
    ready_rows = {r['camera']: r for r in ready['records']}
    transforms = c / 'data/transforms.json'
    data = json.loads(transforms.read_text())
    frames = {r['physical_camera']: r for r in data['frames'] if r['file_path'] in data['train_filenames']}
    box = np.asarray(json.loads((c / 'data/ready.json').read_text())['aabb'])
    assert not set(POLYGONS) & {'DC','HC','KA'}
    assert set(POLYGONS).issubset(frames)
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'proposal.json').exists(), 'Preserve reviewed generations'
    font = ImageFont.truetype('DejaVuSans.ttf', 15)
    records = []
    for camera in POLYGONS:
        source = sources[camera]; patch = ready_rows[camera]
        frame = frames[camera]
        rgb = c / 'data' / frame['file_path']
        archive = Path(source['archive'])
        semantic = c / 'data/semantics' / (camera + '.npz')
        assert sha(rgb) == patch['rgb_sha256']
        assert sha(archive) == source['archive_sha256']
        assert sha(semantic) == source['source_semantic_sha256'] == patch['semantic_sha256']
        with np.load(archive) as z:
            support, unknown, valid = z['support'], z['unknown'], z['valid']
        with np.load(semantic) as z:
            train_valid, w, band = z['valid'], z['W64'], z['band']
        image = Image.open(rgb).convert('RGB')
        polygon = POLYGONS[camera]
        raster = Image.new('1', image.size)
        ImageDraw.Draw(raster).polygon([tuple(p) for p in polygon], fill=1)
        polygon_mask = np.asarray(raster, dtype=bool)
        y, x = np.where(polygon_mask)
        m = np.asarray(frame['transform_matrix'])
        origin = m[:3, 3]
        direction = np.stack([(x + .5 - frame['cx']) / frame['fl_x'],
                              -(y + .5 - frame['cy']) / frame['fl_y'], -np.ones(len(x))], axis=1) @ m[:3, :3].T
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            first, second = (box[0] - origin) / direction, (box[1] - origin) / direction
            near = np.maximum(np.minimum(first, second).max(axis=1), .01)
            far = np.maximum(first, second).min(axis=1)
            floor_t = -origin[2] / direction[:, 2]
        minimum_z = np.minimum(origin[2] + near * direction[:, 2], origin[2] + far * direction[:, 2])
        conditions = dict(source_valid=valid[y, x].astype(bool), train_valid=train_valid[y, x].astype(bool),
                          outside_RAW64_support=~support[y, x].astype(bool), outside_UNKNOWN=~unknown[y, x].astype(bool),
                          W64_zero=w[y, x] == 0, positive_AABB_interval=far > near,
                          interval_z_above_point10=minimum_z > .10, floor_after_AABBfar=floor_t > far,
                          finite_geometry=np.isfinite(np.c_[direction, near, far, floor_t, minimum_z]).all(axis=1))
        keep = np.logical_and.reduce(list(conditions.values()))
        mask = np.zeros_like(polygon_mask)
        mask[y[keep], x[keep]] = True
        assert keep.any(), 'No eligible pixels in the manually proposed source polygon'
        np.savez_compressed(args.output / (camera + '_proposed_pixels.npz'), proposed_pixels=mask,
                            drawn_polygon=polygon_mask, x=x[keep], y=y[keep], origin=origin,
                            direction=direction[keep], near=near[keep], far=far[keep], floor_t=floor_t[keep])
        marked = image.copy()
        draw = ImageDraw.Draw(marked)
        draw.line([tuple(p) for p in polygon + polygon[:1]], fill='cyan', width=2)
        # Dots identify rejected pixel centers without hiding the source under a filled mask.
        for xx, yy in zip(x[~keep], y[~keep]):
            draw.point((int(xx), int(yy)), fill=(255, 80, 80))
        caption = camera + ' proposal: cyan polygon; red rejected by unchanged checks; SOURCE REVIEW PENDING'
        draw.rectangle((0, 0, image.width, 26), fill='black')
        draw.text((5, 4), caption, font=font, fill='white')
        marked.save(args.output / (camera + '_full_source_overlay.png'))
        xy = np.asarray(polygon)
        crop = (int(xy[:, 0].min()) - 35, max(26, int(xy[:, 1].min()) - 35),
                int(xy[:, 0].max()) + 36, int(xy[:, 1].max()) + 36)
        plain = image.crop(crop).resize(((crop[2] - crop[0]) * 4, (crop[3] - crop[1]) * 4), Image.Resampling.NEAREST)
        overlay = marked.crop(crop).resize(plain.size, Image.Resampling.NEAREST)
        contact = Image.new('RGB', (plain.width * 2, plain.height + 35), '#181818')
        contact.paste(plain, (0, 35)); contact.paste(overlay, (plain.width, 35))
        draw = ImageDraw.Draw(contact)
        draw.text((4, 4), camera + ' unmarked source, HD crop ' + str(crop), font=font, fill='white')
        draw.text((plain.width + 4, 4), 'Proposed polygon: ' + str(int(keep.sum())) + ' eligible pixels', font=font, fill='white')
        contact.save(args.output / (camera + '_source_zoom.png'))
        records.append(dict(camera=camera, polygon_HD_integer_vertices=polygon, rasterization='PIL integer polygon; ray centers x+.5,y+.5',
                            drawn_pixels=int(len(x)), proposed_pixels=int(keep.sum()), camera_position_floor=origin.tolist(), camera_azimuth_degrees=float(np.degrees(np.arctan2(origin[1]-box.mean(0)[1],origin[0]-box.mean(0)[0]))),
                            rejected_by_condition_nonexclusive={name: int((~value).sum()) for name, value in conditions.items()},
                            min_interval_floor_z=float(minimum_z[keep].min()), min_floor_after_far=float((floor_t - far)[keep].min()),
                            min_interval_length=float((far - near)[keep].min()), band_pixels=int(band[y[keep], x[keep]].sum()),
                            source_RGB=str(rgb), source_RGB_sha256=sha(rgb), source_archive=str(archive), source_archive_sha256=sha(archive),
                            source_semantic=str(semantic), source_semantic_sha256=sha(semantic),
                            proposal_archive_sha256=sha(args.output / (camera + '_proposed_pixels.npz')), crop_HD_xyxy=crop))
    receipt = dict(kind='TRAIN_ONLY_SOURCE_POLYGON_REVIEW_PROPOSAL', training_eligible=False, root_review_pending=True,
                   no_source_or_accepted_targets_changed=True, no_GPU_or_model_used=True,
                   existing7369_proposal_unchanged_sha256=sha(original), support_manifest_sha256=sha(support_path), transforms_sha256=sha(transforms),
                   ready_sha256=sha(c / 'data/ready.json'), helper_sha256=sha(Path(__file__)), AABB=box.tolist(),
                   total_proposed_pixels=sum(r['proposed_pixels'] for r in records), records=records,
                   limitations=['Conditional on approximate existing floor frame; no new independent depth measurement',
                                'Visible source floor only, no blanket clearance of source cache',
                                'All intervals retain minimum z>.10; no extension into shoe/floor contacts',
                                'All four additional source polygons require root review; none is an accepted target',
                                'DA and MA overviewed but not selected; KA explicitly excluded as EVAL'])
    (args.output / 'proposal.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({r['camera']: r['proposed_pixels'] for r in records}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
