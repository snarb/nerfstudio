"""Posthoc missing-ray attribution for a frame-configured local MHR completion.

No mesh, fit, mask or gate is changed. Train-skin semantics only select a
diagnostic cohort; they are not ground-truth anatomy or a quality metric.
"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
from study_multiview_face_prior import read, save, sha, CROP
from transfer_local_mhr_prior import settings, verify
from review_local_mhr_transfer import camera_crops, enclosed_misses
from run_local_mhr_completion import require, prefix_check
from diagnose_mhr_admission_stages import stage_ids


def cohort(depth, crop, skin=None):
    """Return missing-ray cohort and the stricter eroded-skin subcohort."""
    depth = np.asarray(depth)
    x0, y0, x1, y1 = crop
    region = np.zeros(depth.shape, bool)
    region[y0:y1, x0:x1] = True
    if skin is None:
        selected = enclosed_misses(depth, crop)
        return selected, np.zeros_like(selected)
    require(skin.shape == depth.shape, 'Skin/depth lattice mismatch')
    missing = ~np.isfinite(depth) | (depth <= 0)
    support = skin >= 230
    return region & missing & support, region & missing & binary_erosion(support, iterations=5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--completion', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import open3d as o3d
    from admit_mhr_local_patch_depth import Scene2
    from bake_joint_temporal_mesh import camera_depth
    spec = read(a.spec); verify(spec); s = settings(spec)
    config = read(a.completion/'config.json')
    require(config['spec']['frame'] == spec['frame'], 'Wrong-time completion')
    audit = read(a.completion/'audit.json')
    require(audit['status'] == 'passed', 'Audited completion required')
    for name, digest in audit['inventory'].items():
        require(sha(a.completion/name) == digest, 'Changed audited artifact: '+name)
    prefix_check(config)
    source, views = camera_crops(spec)
    old = o3d.io.read_triangle_mesh(source['mesh'])
    oldscene = Scene2(np.asarray(old.vertices), np.asarray(old.triangles))
    arm = a.completion/'candidates/silhouette100'
    domain = np.load(arm/'domain_evidence.npz')
    prior = np.load(s['FINAL']/'fit.npz')
    triangles = prior['triangles']; neutral = prior['neutral']
    band = ((neutral[triangles, 1] >= 135) & (neutral[triangles, 1] <= 153)).all(1)
    safe = band & ~domain['unsafe_parent']
    raw = o3d.io.read_triangle_mesh(str(arm/'local_raw.ply'))
    proposals = np.load(arm/'proposal_evidence.npz')['proposals']
    folder = a.completion/'admission/silhouette100'
    admission = np.load(folder/'admission.npz')
    ids = stage_ids(len(proposals), admission['semantic_ids'], admission['strict'], admission['interpolated'],
        np.load(folder/'strict/evidence.npz')['retained_proposal_ids'],
        np.load(folder/'interpolated/evidence.npz')['retained_proposal_ids'])
    meshes = dict(whole_prior=(prior['vertices'], triangles), anatomical_band=(prior['vertices'], triangles[band]),
        safe_band=(prior['vertices'], triangles[safe]),
        local_before_centroid=(domain['subdivided_vertices'], domain['subdivided_triangles'][domain['local_before_centroid_gate']]))
    meshes.update({name: (np.asarray(raw.vertices), proposals[ix]) for name, ix in ids.items()})
    scenes = {name: Scene2(v, t) for name, (v, t) in meshes.items() if len(t)}
    require(not a.output.exists(), 'Use a fresh diagnostic root')
    a.output.mkdir(parents=True)
    records = []
    bindings = {str(path): sha(path) for path in [a.spec, a.completion/'config.json', a.completion/'audit.json',
        Path(source['mesh']), s['FINAL']/'fit.npz', arm/'domain_evidence.npz', arm/'proposal_evidence.npz',
        folder/'admission.npz', folder/'strict/evidence.npz', folder/'interpolated/evidence.npz', Path(__file__)]}
    for view, item in views.items():
        camera = item['camera']; d, _, _ = camera_depth(oldscene, camera)
        depth = np.rot90(np.where(np.isfinite(d), d, 0))
        skin = None
        if view != 'current_moving':
            path = s['HEAD']/'semantics'/(camera['physical_camera']+'.npz')
            skin = np.zeros(depth.shape, np.uint8)
            x0, y0, x1, y1 = CROP; skin[y0:y1, x0:x1] = np.load(path)['skin']
            bindings[str(path)] = sha(path)
        selected, eroded = cohort(depth, item['crop'], skin)
        y, x = np.nonzero(selected)
        pose = np.asarray(camera['transform_matrix'])
        ext = np.linalg.inv(pose @ np.diag([1., -1., -1., 1.])).astype(np.float32)
        k = np.array([[camera['fl_x'], 0, camera['cx']], [0, camera['fl_y'], camera['cy']], [0, 0, 1]], np.float32)
        rays = oldscene.create_rays_pinhole(o3d.core.Tensor(k), o3d.core.Tensor(ext), camera['w'], camera['h']).numpy()
        rays = o3d.core.Tensor(np.ascontiguousarray(np.rot90(rays)[y, x]))
        arrays = dict(portrait_xy=np.c_[x, y], eroded_skin=eroded[y, x])
        counts = {}
        for name in meshes:
            z = scenes[name].cast_rays(rays)['t_hit'].numpy() if len(x) and name in scenes else np.full(len(x), np.inf)
            hit = np.isfinite(z) & (z > 0)
            arrays[name+'_depth'] = z
            counts[name] = dict(hits=int(hit.sum()), eroded_skin_hits=int((hit & eroded[y, x]).sum()))
        np.savez_compressed(a.output/(view+'.npz'), **arrays)
        if skin is not None:
            imagepath = s['RGB']/spec['frame']/(camera['physical_camera']+'.png')
            im = Image.new('RGB', (1080, 1920)); im.paste(Image.open(imagepath), CROP[:2])
            im = np.array(im); im[y, x] = [255, 0, 255]
            im[eroded] = [0, 255, 255]
            native = Image.fromarray(im).crop(item['crop'])
            panel = Image.new('RGB', (native.width, native.height+24))
            panel.paste(native, (0, 24)); ImageDraw.Draw(panel).text((3, 4), 'missing rays: magenta skin / cyan eroded skin', fill='white')
            panel.save(a.output/(view+'.png')); bindings[str(imagepath)] = sha(imagepath)
        records.append(dict(view=view, camera=camera, crop=item['crop'], missing=int(selected.sum()),
            eroded_skin_missing=int(eroded.sum()), stages=counts))
        print(view, records[-1]['missing'], counts, flush=True)
    save(a.output/'result.json', dict(frame=spec['frame'], records=records, input_hashes=bindings,
        meshes_changed=False, target_used_posthoc_only=True, skin_is_predicted_not_ground_truth=True,
        moving_cohort='enclosed missing rays, not confirmed anatomical holes',
        train_cohort='unmasked production-mesh misses within fixed review crop and train skin confidence >=230/255',
        artifacts={f.name: sha(f) for f in a.output.iterdir() if f.is_file()}))


if __name__ == '__main__':
    main()
