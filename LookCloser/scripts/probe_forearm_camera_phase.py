"""Screen unchanged camera-loop phases at actual damaged actor times."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from joint_temporal_texture import CALIBRATION, read, sha, atomic_json
from render_smooth_temporal_mesh_video import verify_request, calibration_pose
from probe_temporal_camera_phase import shifted_camera
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth

PARENT = Path('/mnt/data/dec5_phase30_early_texture_dynamic_150')
DEFAULT = Path('/mnt/data/dec5_forearm_camera_phase_probe')
FRAMES = ['001029', '001033', '001037', '001041']
OFFSETS = [0, 30, 60, 90, 120]


def run(output):
    request = verify_request(PARENT)
    calibration = read(CALIBRATION)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'request.json', dict(
        parent=str(PARENT), parent_request_sha256=sha(PARENT / 'request.json'),
        frames=FRAMES, offsets_relative_to_published=OFFSETS,
        actor_times_unchanged=True, original_meshes_unchanged=True,
        same_camera_loop=True, heldout_rgb_used=False, script_sha256=sha(__file__),
        scope='clay visibility screening; not accepted RGB video'))
    paths = {}
    for phase in OFFSETS:
        cameras = [shifted_camera(request['inventory'], i, phase, calibration) for i in range(150)]
        positions = np.array([np.array(calibration_pose(c, calibration, read(r['metadata']))['transform_matrix'])[:3, 3]
                              for c, r in zip(cameras, request['inventory'])])
        steps = np.linalg.norm(np.roll(positions, -1, axis=0) - positions, axis=1)
        if len(np.unique(positions, axis=0)) != 150 or steps.min() <= 0:
            raise ValueError('Static or duplicate camera')
        paths[str(phase)] = dict(cameras=cameras, camera_count=150,
                                step_max_min_ratio=float(steps.max() / steps.min()))
    atomic_json(output / 'camera_paths.json', paths)
    records = []
    for frame in FRAMES:
        index = next(i for i, r in enumerate(request['inventory']) if r['frame_id'] == frame)
        record = request['inventory'][index]
        if sha(record['mesh']) != record['mesh_sha256']:
            raise ValueError('Changed production mesh')
        mesh = o3d.io.read_triangle_mesh(record['mesh'])
        mesh.compute_triangle_normals()
        scene = scene_for(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
        normals = np.asarray(mesh.triangle_normals)
        folder = output / frame
        folder.mkdir()
        sheet = Image.new('RGB', (450 * len(OFFSETS), 984))
        draw = ImageDraw.Draw(sheet)
        for column, phase in enumerate(OFFSETS):
            camera = paths[str(phase)]['cameras'][index]
            depth, ids, _ = camera_depth(scene, camera)
            hit = np.isfinite(depth)
            rgb = np.zeros((*depth.shape, 3), np.uint8)
            shade = (.25 + .75 * np.abs(normals[ids[hit]] @ np.array(camera['transform_matrix'])[:3, 2])).clip(0, 1)
            rgb[hit] = np.rint(shade[:, None] * np.array([205, 210, 216])).astype(np.uint8)
            im = Image.fromarray(np.rot90(rgb))
            path = folder / f'offset_{phase:03d}.png'
            im.save(path)
            # Native crops are retained; no crop is applied to the camera model.
            im.crop((0, 500, 1080, 1300)).save(folder / f'offset_{phase:03d}_head.png')
            im.crop((0, 1300, 1080, 1920)).save(folder / f'offset_{phase:03d}_arm.png')
            sheet.paste(im.resize((450, 800), Image.Resampling.LANCZOS), (450 * column, 24))
            draw.text((450 * column + 4, 4), f'{frame} relative phase +{phase}', fill='white')
            records.append(dict(frame=frame, phase=phase, mesh_sha256=record['mesh_sha256'],
                                path=str(path), sha256=sha(path)))
        sheet.save(folder / 'overview.png')
        print('Completed', frame, flush=True)
    atomic_json(output / 'results.json', dict(records=records, visual_status='pending'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    run(parser.parse_args().output)
