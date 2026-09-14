"""Attribute missing movie pixels to original geometry, later carving, or RGB eligibility.

Diagnostic only: black background is not automatically missing anatomy. Identical
camera rays compare saved stages, with input hashes and native evidence retained.
"""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth


def classify(original_depth, final_depth, source_ids):
    original = np.isfinite(original_depth) & (original_depth > 0)
    final = np.isfinite(final_depth) & (final_depth > 0)
    return dict(original_miss=~original, removed_surface=original & ~final,
                geometry_without_rgb=final & (source_ids == 255),
                supported_render=final & (source_ids != 255))


def diagnose(parent, output, frame):
    request = read(parent/'request.json')
    row = next(r for r in request['inventory'] if r['frame_id'] == frame)
    saved = parent/'frames'/frame
    rgb = np.asarray(Image.open(saved/'frame.png').convert('RGB'))
    source_ids = np.asarray(Image.open(saved/'source_ids.png'))
    if source_ids.shape != (1080, 1920):
        raise ValueError(f'Unexpected source ID shape {source_ids.shape}')
    root = output/frame
    binding = dict(parent_request_sha256=sha(parent/'request.json'),
                   render_sha256=sha(saved/'frame.png'), source_ids_sha256=sha(saved/'source_ids.png'),
                   script_sha256=sha(__file__), camera=row['camera'])
    stages = [('original', row['untrimmed_mesh']), ('final', row['mesh'])]
    depths = {}; panels = []; records = {}
    for name, path in stages:
        expected = row['untrimmed_mesh_sha256' if name == 'original' else 'mesh_sha256']
        if sha(path) != expected:
            raise ValueError('Changed input geometry')
        mesh = o3d.io.read_triangle_mesh(path)
        vertices = np.asarray(mesh.vertices); triangles = np.asarray(mesh.triangles)
        depth, ids, _ = camera_depth(scene_for(vertices, triangles), row['camera'])
        depths[name] = depth
        normals = np.cross(vertices[triangles[:, 1]]-vertices[triangles[:, 0]],
                           vertices[triangles[:, 2]]-vertices[triangles[:, 0]])
        normals /= np.linalg.norm(normals, axis=1)[:, None].clip(1e-12)
        light = np.asarray(row['camera']['transform_matrix'])[:3, 2]
        shade = (.2 + .8*np.abs(normals @ light))*230
        clay = np.zeros((*depth.shape, 3), np.uint8); hit = np.isfinite(depth)
        clay[hit] = shade[ids[hit], None].astype(np.uint8)
        panels.append(np.rot90(clay))
        records[name] = dict(path=path, sha256=expected, triangles=len(triangles),
                             vertices=len(vertices), bounds=[vertices.min(0).tolist(), vertices.max(0).tolist()])
    saved_depth = np.load(saved/'target_depth.npz')['depth']
    if not np.allclose(np.where(np.isfinite(depths['final']), depths['final'], 0), saved_depth, atol=1e-6):
        raise ValueError('Fresh final cast differs from saved depth')
    categories = {k: np.rot90(v) for k,v in classify(depths['original'], depths['final'], source_ids).items()}
    overlay = rgb.copy()
    overlay[categories['removed_surface']] = [255, 40, 0]
    overlay[categories['geometry_without_rgb']] = [0, 100, 255]
    # Keep original misses black/unchanged: they include genuine room background.
    root.mkdir(parents=True, exist_ok=True)
    if (root/'request.json').exists() and read(root/'request.json') != binding:
        raise ValueError('Immutable diagnosis mismatch')
    atomic_json(root/'request.json', binding)
    panel = Image.new('RGB', (2160, 990)); draw = ImageDraw.Draw(panel)
    for i,(title,img) in enumerate(zip(['RGB','Original clay','Final clay','RED removed / BLUE RGB missing'],[rgb,*panels,overlay])):
        panel.paste(Image.fromarray(img).resize((540,960)), (i*540,30))
        draw.text((i*540+3,5), title, fill='white')
    panel.save(root/'comparison.png')
    for name,img in zip(['original_clay','final_clay','attribution'],[*panels,overlay]):
        Image.fromarray(img).save(root/(name+'.png'))
    counts = {}
    for name,box in dict(head_context=[100,450,1050,1300], lower_context=[0,1150,1080,1920]).items():
        x0,y0,x1,y1=box
        counts[name] = {k:int(v[y0:y1,x0:x1].sum()) for k,v in categories.items()}
    np.savez_compressed(root/'evidence.npz', **categories)
    result = dict(frame_id=frame, stages=records, counts=counts,
        counts_are_context_not_anatomical_roi=True, missing_original_rays_include_true_background=True,
        image_quality_metrics_computed=False, visual_status='pending', previous_goal_turn='progress')
    atomic_json(root/'result.json', result)
    atomic_json(root/'complete.json', dict(hashes={p.name:sha(p) for p in root.iterdir() if p.is_file() and p.name!='complete.json'}))
    print(frame, counts, flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,default=Path('/mnt/data/dec5_elevated_camera_dynamic_150'))
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_temporal_missing_surface_diagnosis'))
    p.add_argument('--frames',nargs='+',default=['001033','001041'])
    a=p.parse_args()
    for frame in a.frames: diagnose(a.parent,a.output,frame)
