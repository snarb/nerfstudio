"""CPU-only wide-path check of sealed fringe edits; no RGB or quality claim."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json
from bake_joint_temporal_mesh import camera_depth

ROOT = Path('/mnt/data/dec5_wide_fringe_geometry')
GEOMETRY = Path('/mnt/data/dec5_weak_fringe_replacement')
VIDEO = Path('/mnt/data/dec5_large_motion_choices_v3')
VARIANTS = ('diagonal_sweep', 'wide_oval', 'left_high_arc', 'right_high_arc_refined')


def load_scene(path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                        o3d.core.Tensor(triangles.astype(np.uint32)))
    return scene, np.asarray(mesh.triangle_normals), (vertices[triangles, 0] > -.03).all(1)


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    inputs = {str(GEOMETRY/'artifact_manifest.json'): sha(GEOMETRY/'artifact_manifest.json')}
    records = []
    for frame in ('001083', '001123'):
        original = read(GEOMETRY/frame/'request.json')
        paths = {'production': Path(original['source_mesh']),
                 'remove_only': GEOMETRY/frame/'remove_only/mesh.ply',
                 'replace': GEOMETRY/frame/'replace/mesh.ply'}
        assert sha(paths['production']) == original['source_mesh_sha256']
        for arm in ('remove_only', 'replace'):
            receipt = read(paths[arm].parent/'result.json')
            assert sha(paths[arm]) == receipt['hashes']['mesh.ply']
        scenes = {arm: load_scene(path) for arm, path in paths.items()}
        for p in paths.values(): inputs[str(p)] = sha(p)
        for variant in VARIANTS:
            request_path = VIDEO/variant/'request.json'
            request = read(request_path)
            entry = next(r for r in request['inventory'] if r['frame_id'] == frame)
            assert entry['mesh_sha256'] == original['source_mesh_sha256']
            inputs[str(request_path)] = sha(request_path)
            camera = entry['camera']; view = np.array(camera['transform_matrix'])[:3, 2]
            out = ROOT/frame/variant; out.mkdir(parents=True, exist_ok=True)
            depths = {}; images = {}; head = None
            for arm, (scene, normals, head_faces) in scenes.items():
                depth, ids, _ = camera_depth(scene, camera)
                hit = np.isfinite(depth) & (depth > 0)
                shade = np.zeros((*depth.shape, 3), np.uint8)
                value = np.clip(65 + 175*np.abs(normals[ids[hit]] @ view), 0, 255).astype(np.uint8)
                shade[hit] = value[:, None]
                depths[arm] = np.rot90(np.where(hit, depth, 0))
                images[arm] = np.rot90(shade)
                if arm == 'production':
                    mask = np.zeros(hit.shape, bool); mask[hit] = head_faces[ids[hit]]
                    head = np.rot90(mask)
            yy, xx = np.nonzero(head)
            box = (max(0, int(xx.min())-35), max(0, int(yy.min())-35),
                   min(1080, int(xx.max())+36), min(1920, int(yy.max())+36))
            width, height = box[2]-box[0], box[3]-box[1]
            panel = Image.new('RGB', (width*3, height+28))
            draw = ImageDraw.Draw(panel)
            for i, arm in enumerate(paths):
                im = Image.fromarray(images[arm]); im.save(out/(arm+'.png'))
                panel.paste(im.crop(box), (i*width, 28)); draw.text((i*width+4, 5), arm, fill='white')
            panel.save(out/'head_comparison.png')
            base_hit = depths['production'] > 0
            changes = {}
            for arm in ('remove_only', 'replace'):
                new_hit = depths[arm] > 0
                lost = base_hit & ~new_hit; gained = ~base_hit & new_hit
                deeper = base_hit & new_hit & (depths[arm] > depths['production']+.0001)
                closer = base_hit & new_hit & (depths[arm] < depths['production']-.0001)
                changes[arm] = dict(lost=int(lost.sum()), gained=int(gained.sum()),
                    deeper=int(deeper.sum()), closer=int(closer.sum()))
            np.savez_compressed(out/'depths.npz', **depths)
            record = dict(frame=frame, variant=variant, camera=camera, head_crop=box, changes=changes,
                input_mesh_hashes={a: sha(p) for a, p in paths.items()},
                request_sha256=sha(request_path), visual_status='pending',
                output_hashes={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name!='result.json'})
            atomic_json(out/'result.json', record); records.append(record)
            print(frame, variant, changes, flush=True)
    atomic_json(ROOT/'result.json', dict(records=records, inputs=inputs,
        script_sha256=sha(__file__), cpu_only=True, rgb_tested=False,
        counts_not_quality_metrics=True, production_changed=False, visual_status='pending'))


if __name__ == '__main__': run()
