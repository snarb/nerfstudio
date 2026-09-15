"""CPU clay anatomy/topology inspection of public MHR neutral, no DEC5 fit."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from prepare_mhr_head_prior import ROOT
from build_train_hair_semantics import read, sha, write


def main():
    import open3d as o3d
    import torch
    torch.set_num_threads(2)
    receipt = read(ROOT/'runtime.json')
    for name, digest in receipt['output_hashes'].items():
        assert sha(ROOT/name) == digest
    target = ROOT/'review'
    target.mkdir(exist_ok=False)
    mesh = o3d.io.read_triangle_mesh(str(ROOT/'neutral.ply'))
    mesh.compute_triangle_normals()
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    center = np.array([0., 154., 2.])
    poses = [('front', np.array([0., 154., 100.])),
             ('side', np.array([95., 154., 20.])),
             ('under_jaw', np.array([45., 110., 80.]))]
    panel = Image.new('RGB', (3*600, 624), (20, 20, 20))
    draw = ImageDraw.Draw(panel)
    yy, xx = np.mgrid[:600, :600]
    for index, (name, camera) in enumerate(poses):
        forward = center-camera; forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 1., 0.]); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        # Orthographic anatomical inspection, not a calibrated DEC5 render.
        origins = camera + ((xx-299.5)/600*48)[...,None]*right + ((299.5-yy)/600*48)[...,None]*up
        directions = np.broadcast_to(forward, origins.shape)
        hit = scene.cast_rays(o3d.core.Tensor(np.concatenate((origins, directions), 2).astype(np.float32)))
        depth, ids = hit['t_hit'].numpy(), hit['primitive_ids'].numpy()
        valid = np.isfinite(depth)
        rgb = np.full((600, 600, 3), 20, np.uint8)
        normals = np.asarray(mesh.triangle_normals)[ids[valid]]
        shade = np.clip(65+175*np.abs(normals@-forward), 0, 255).astype(np.uint8)
        rgb[valid] = shade[:,None]
        im = Image.fromarray(rgb); im.save(target/(name+'.png'))
        panel.paste(im, (600*index, 24)); draw.text((600*index+5, 5), name+' | neutral prior, not DEC5', fill='white')
    panel.save(target/'anatomy.png')
    edges = np.sort(t[:, [[0,1], [1,2], [2,0]]].reshape(-1,2), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    length = np.linalg.norm(v[unique[:,0]]-v[unique[:,1]], axis=1)
    head_edges = (v[unique,1]>135).all(1)
    # Inventory public model buffers to clarify where topology comes from.
    model = torch.jit.load(str(ROOT/'mhr_model.pt'), map_location='cpu')
    buffers = {name:list(value.shape) for name,value in model.named_buffers()}
    write(target/'manifest.json', dict(script_sha256=sha(__file__), runtime_sha256=sha(ROOT/'runtime.json'),
        model_buffers=buffers, topology_source='Pinned upstream mhr_face_mask.ply; not assumed same neutral pose',
        open_edges=int((count==1).sum()), nonmanifold_edges=int((count>2).sum()),
        head_edge_length_cm=dict(median=float(np.median(length[head_edges])),p99=float(np.percentile(length[head_edges],99)),maximum=float(length[head_edges].max())),
        files={str(p):sha(p) for p in sorted(target.glob('*.png'))},
        geometry_repair=False, visual_status='pending'))
    print('native neutral anatomy panels ready', flush=True)


if __name__ == '__main__':
    main()
