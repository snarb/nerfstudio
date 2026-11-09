"""Existing calibrated native source context; no labels, alphaGT, or GPU."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from marshawn_actor_coordinates import raw_array_indices, raw_camera, RIG
from prepare_marshawn_frozen_semantic_cpu import sha, write


def run(c, output):
    output.mkdir(parents=True, exist_ok=False)
    source_path = c.parent/'two_actor_isolation/raw_native_sources/request.json'
    sources = {r['camera']: r for r in json.loads(source_path.read_text())['records']}
    frames = {r['physical_camera']: r for r in json.loads((c/'data/transforms.json').read_text())['frames']}
    fresh = c/'frozen_native_semantic_prior001_candidate_v1'; old = c/'frozen_native_semantic_render_recovery_v1'
    cases = [('HC_isolated_fleck', 'HC', [875, 602], 40), ('DC_gray_shoe_bag_edge', 'DC', [1097, 572], 35),
             ('KA_gray_changed_core_contact', 'KA', [1075, 270], 35), ('KA_gray_changed_core_bag', 'KA', [961, 485], 35)]
    bindings = {str(p): sha(p) for p in [source_path, c/'data/transforms.json', RIG/'images.txt', RIG/'cameras.txt']}
    rows, panels = [], []
    for name, cam, pixel, pad in cases:
        frame, source = frames[cam], sources[cam]
        raw_path = Path(source['image']); assert sha(raw_path) == source['image_sha256']
        K, distortion = raw_camera(cam)
        np.testing.assert_allclose(K, source['raw_OpenCV_K'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(distortion, source['distortion'], atol=1e-12, rtol=0)
        x, y = pixel
        box = [max(0, x-pad), max(0, y-pad), min(frame['w'], x+pad+1), min(frame['h'], y+pad+1)]
        raw_center = raw_array_indices(np.asarray([[x+.5, y+.5]]), frame)[0]
        raw_corners = raw_array_indices(np.asarray([[x, y], [x+1, y], [x+1, y+1], [x, y+1]], float), frame)
        raw_roi = raw_array_indices(np.asarray([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], float), frame)
        offset = np.asarray(source['crop_xyxy'][:2]); lower = np.floor(raw_roi.min(0)-offset-5).astype(int); upper = np.ceil(raw_roi.max(0)-offset+6).astype(int)
        paths = [c/'data'/frame['file_path'], raw_path, old/'renders'/cam/'identity_inside.png', fresh/'renders'/cam/'hard_theta05.png']
        images = [Image.open(paths[0]).convert('RGB').crop(box), Image.open(raw_path).convert('RGB').crop((*lower, *upper)),
                  Image.open(paths[2]).convert('RGB').crop(box), Image.open(paths[3]).convert('RGB').crop(box)]
        for j in [0, 2, 3]:
            draw = ImageDraw.Draw(images[j]); xx, yy = x-box[0], y-box[1]; draw.ellipse((xx-1, yy-1, xx+1, yy+1), outline='red', width=1)
        draw = ImageDraw.Draw(images[1]); poly = raw_corners-offset-lower
        draw.polygon([tuple(v) for v in poly], outline='red'); center = raw_center-offset-lower
        draw.ellipse((center[0]-3, center[1]-3, center[0]+3, center[1]+3), outline='yellow', width=1)
        panel = Image.new('RGB', (4*360, 430), '#171717'); draw = ImageDraw.Draw(panel)
        labels = ['HD SOURCE', 'NATIVE RAW source display', 'nohead box reference', 'p01 hard.5 physical']
        for j, (tile, label) in enumerate(zip(images, labels)):
            scale = min(350/tile.width, 355/tile.height); tile = tile.resize((round(tile.width*scale), round(tile.height*scale)), Image.Resampling.NEAREST)
            panel.paste(tile, (j*360+(360-tile.width)//2, 60)); draw.text((j*360+4, 35), label, fill='white')
        draw.text((4, 4), name+' HDpixel='+str(pixel)+'; red=geometricpixel footprint, NOT alphaGT / NO labels changed', fill='white')
        panel.save(output/(name+'.png')); panels.append(panel)
        with np.load(fresh/'renders'/cam/'physical_and_partition.npz') as z:
            values = {k: z[k][y, x].tolist() for k in ['identity_inside', 'hard_theta01', 'hard_theta05', 'soft_probability']}
        with np.load(c/'data/semantics'/f'{cam}.npz') as z:
            existing = {k: float(z[k][y, x]) for k in ['black', 'grey', 'core', 'band', 'W64']}
        rows.append(dict(name=name, camera=cam, pixel_xy=pixel, crop_xyxy=box, RAW_array_center=raw_center.tolist(), RAW_geometric_pixel_corners=raw_corners.tolist(),
            existing_EVAL_semantic_fields_diagnostic_only=existing, existing_model_RGBA=values,
            source_image=str(raw_path), source_image_sha256=source['image_sha256'], raw_display_contract=source['display'],
            potential_semantic_qualification='PENDING_SOURCE_REVIEW; visible source may be occluder, not empty behind. No EVAL label enters fitting.'))
        for path in paths: bindings[str(path)] = sha(path)
    sheet = Image.new('RGB', (1440, sum(p.height for p in panels)), '#171717'); y = 0
    for panel in panels: sheet.paste(panel, (0, y)); y += panel.height
    sheet.save(output/'all_native_source_contexts.jpg', quality=94)
    write(output/'source_context.json', dict(records=rows, source_sha256=bindings, helper_sha256=sha(__file__),
        coordinate_helper_sha256=sha(Path(__file__).with_name('marshawn_actor_coordinates.py')), CPU_only=True,
        no_model_queries=True, no_new_alpha_or_labels=True, no_whole_ray_negative_or_free_space_claim=True,
        scope='EVAL source visual diagnosis only. TRACE of actualEVAL density/contributors needs another separately authorized query; source photos alone do not identify hidden actor geometry.'))
    print(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--campaign', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.campaign, a.output)
