"""One-time train-only hand silhouette feasibility and volume proposal.

No movie/default mutation. Landmark predictions bound the experiment only;
they are not surface observations. Existing person masks exclude room, and a
fixed red-minus-blue test isolates warm hand skin against blue clothing.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json, project, cameras
from local_silhouette_volume import signed_pixels, combine_silhouettes, remove_box_caps

ROOT = Path('/mnt/data/dec5_hand_silhouette_volume')
MOVIE = Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150')
OBS = Path('/mnt/data/dec5_wrist_observations')
LANDMARKS = Path('/mnt/data/dec5_hand_landmark_triangulation')
FRAME = '001037'
SETTINGS = dict(portrait_roi=[0, 1400, 500, 1920], red_minus_blue_minimum=8,
                close_radius=3, minimum_views=3, margins_pixels=[0., 3.],
                voxel_spacing=.0004, landmark_bounds_padding=.025,
                minimum_component_pixels=500)


def stage():
    root = ROOT/FRAME
    root.mkdir(parents=True, exist_ok=False)
    movie = read(MOVIE/'request.json')
    entry = next(r for r in movie['inventory'] if r['frame_id'] == FRAME)
    obs = read(OBS/FRAME/'result.json')
    rows, _, metadata = cameras(FRAME)
    byname = {r['physical_camera']: r for r in rows}
    masksroot = Path(entry['source_masks']['root'])
    for file, key in [('masks.npz', 'masks_sha256'), ('cameras.json', 'cameras_sha256')]:
        assert sha(masksroot/file) == entry['source_masks'][key]
    masks = dict(zip(read(masksroot/'cameras.json'), np.load(masksroot/'masks.npz')['masks']))
    points = np.load(LANDMARKS/FRAME/'evidence.npz')
    bounds = points['points'][points['good']]
    pad = SETTINGS['landmark_bounds_padding']
    lower, upper = bounds.min(0)-pad, bounds.max(0)+pad
    dependencies = [MOVIE/'request.json', OBS/FRAME/'result.json', OBS/FRAME/'request.json',
                    LANDMARKS/FRAME/'evidence.npz', LANDMARKS/'result.json', metadata,
                    masksroot/'masks.npz', masksroot/'cameras.json', Path(entry['mesh'])]
    arrays = {}; records = []; panel = Image.new('RGB', (1500, 1120)); draw = ImageDraw.Draw(panel)
    for i, item in enumerate(obs['records']):
        name = item['camera']['physical_camera']; row = byname[name]
        for key in ['transform_matrix', 'fl_x', 'fl_y', 'cx', 'cy']:
            np.testing.assert_allclose(row[key], item['camera'][key], atol=1e-7, rtol=0)
        path = OBS/FRAME/(name+'.png')
        assert sha(path) == item['image_sha256'] and sha(row['file_path']) == item['source_sha256']
        dependencies += [path, Path(row['file_path'])]
        rgb = np.array(Image.open(path))
        person = np.rot90(masks[name]).astype(bool)
        x0,y0,x1,y1 = SETTINGS['portrait_roi']
        domain = np.zeros(rgb.shape[:2], bool); domain[y0:y1,x0:x1] = True
        warm = rgb[...,0].astype(int)-rgb[...,2].astype(int) > SETTINGS['red_minus_blue_minimum']
        skin = person & warm & domain
        labels, _ = ndimage.label(skin)
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        skin = labels == sizes.argmax()
        if skin.sum() < SETTINGS['minimum_component_pixels']:
            raise ValueError('No substantial hand skin component')
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        skin = ndimage.binary_fill_holes(skin) & domain & person
        native = np.rot90(skin,-1); native_domain = np.rot90(domain,-1)
        arrays[name+'_mask'] = native; arrays[name+'_domain'] = native_domain
        arrays[name+'_field'] = signed_pixels(native)
        # Original RGB is preserved; diagnostic overlay colors only contours.
        overlay = rgb.copy()
        edge = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3,3),np.uint8)) > 0
        overlay[edge] = [0,255,0]
        crop = Image.fromarray(overlay).crop((0,1400,500,1920))
        crop.save(root/(name+'_mask.png'))
        x,y = (i%3)*500, (i//3)*560
        panel.paste(crop,(x,y+30)); draw.text((x+4,y+6),name,fill='white')
        records.append(dict(camera=row, mask_pixels=int(skin.sum()), image_sha256=sha(path)))
    panel.save(root/'masks_native.png')
    np.savez_compressed(root/'silhouettes.npz', **arrays)
    request = dict(frame=FRAME, settings=SETTINGS, cameras=records, lower=lower.tolist(), upper=upper.tolist(),
                   dependencies={str(p):sha(p) for p in dependencies},
                   scripts={n:sha(Path(__file__).with_name(n)) for n in ['study_hand_silhouette_volume.py','local_silhouette_volume.py','joint_temporal_texture.py']},
                   heldout_used=False, target_camera_used_for_geometry=False, mesh_changed=False,
                   bounds_are_imprecise_landmark_proposal=True, surface_is_inferred_not_measured=True,
                   source_mesh=entry['mesh'], source_mesh_sha256=entry['mesh_sha256'],
                   silhouettes_sha256=sha(root/'silhouettes.npz'), visual_status='pending')
    atomic_json(root/'request.json', request)
    print('staged six silhouettes', [r['mask_pixels'] for r in records], flush=True)


def verify():
    q = read(ROOT/FRAME/'request.json')
    for p,h in q['dependencies'].items():
        if sha(p) != h: raise ValueError('Changed dependency '+p)
    for n,h in q['scripts'].items():
        if sha(Path(__file__).with_name(n)) != h: raise ValueError('Changed script '+n)
    assert sha(ROOT/FRAME/'silhouettes.npz') == q['silhouettes_sha256']
    return q


def volume():
    import open3d as o3d
    from skimage.measure import marching_cubes
    root = ROOT/FRAME; q = verify(); data = np.load(root/'silhouettes.npz')
    rows = [r['camera'] for r in q['cameras']]
    fields = [data[r['physical_camera']+'_field'] for r in rows]
    domains = [data[r['physical_camera']+'_domain'] for r in rows]
    lower, upper = np.array(q['lower']), np.array(q['upper']); spacing = SETTINGS['voxel_spacing']
    axes = [np.arange(a,b+spacing/2,spacing) for a,b in zip(lower,upper)]
    shape = tuple(len(a) for a in axes); actual_upper = np.array([a[-1] for a in axes])
    count = int(np.prod(shape)); outputs=[]
    for margin in SETTINGS['margins_pixels']:
        dest = root/('margin'+str(int(margin))); dest.mkdir(exist_ok=False)
        field = np.empty(count, np.float32); available = np.empty(count,np.uint8)
        for start in range(0,count,100000):
            end = min(start+100000,count)
            ijk = np.array(np.unravel_index(np.arange(start,end),shape)).T
            points = lower+ijk*spacing
            uv,z = project(points,rows)
            sdf,n,positive = combine_silhouettes(uv,z,fields,domains,SETTINGS['minimum_views'],margin)
            field[start:end] = sdf; available[start:end] = n
        occupancy = field.reshape(shape) >= 0
        labels, components = ndimage.label(occupancy)
        counts = np.bincount(labels.ravel())[1:]
        print('margin',margin,'voxels',int(occupancy.sum()),'components',components,flush=True)
        if not occupancy.any(): raise ValueError('Empty multiview silhouette volume')
        # Signed silhouette distances avoid aliasing from thresholded occupancy.
        v,t,_,_ = marching_cubes(field.reshape(shape),0,spacing=(spacing,)*3)
        v += lower
        keep = remove_box_caps(v,t,lower,actual_upper,spacing)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t[keep]))
        mesh.remove_unreferenced_vertices(); mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(dest/'hull.ply'),mesh)
        np.savez_compressed(dest/'field.npz',field=field.reshape(shape),available=available.reshape(shape),lower=lower,spacing=spacing)
        result = dict(margin_pixels=margin,grid_shape=list(shape),occupied_voxels=int(occupancy.sum()),
                      components=int(components),largest_voxel_components=sorted(counts.tolist(),reverse=True)[:10],
                      removed_box_faces=int((~keep).sum()),vertices=len(mesh.vertices),triangles=len(mesh.triangles),
                      mesh_sha256=sha(dest/'hull.ply'),field_sha256=sha(dest/'field.npz'),
                      silhouette_is_envelope_not_true_surface=True,production_mesh_changed=False)
        atomic_json(dest/'result.json',result); outputs.append(result)
    atomic_json(root/'volume_result.json',dict(request_sha256=sha(root/'request.json'),variants=outputs,visual_status='pending'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['stage','volume'])
    a=p.parse_args();{'stage':stage,'volume':volume}[a.action]()
