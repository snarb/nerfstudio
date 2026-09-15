"""Opt-in texture backoff for verified unchanged original surface intersections.

An inferred patch may remove all texture sources for an otherwise unchanged
original target intersection. Reuse the baseline's train-only reprojection in
that exact case. Never fills a new surface, changes depth, averages RGB, or uses
GT. This is a confidence/visibility policy, not proof that old occlusion was right.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import open3d as o3d
from admit_mhr_local_patch_depth import Scene2
from bake_joint_temporal_mesh import camera_depth
from study_multiview_face_prior import read, save, sha


def selection(base_rgb, rgb, base_source, source, base_depth, depth, base_face, face,
              base_bary, bary, original_triangles):
    same = (face == base_face) & (face < original_triangles)
    same &= np.isfinite(base_depth) & np.isfinite(depth) & (depth > 0) & (base_depth > 0)
    difference = np.zeros_like(depth)
    np.subtract(depth, base_depth, out=difference, where=same)
    same &= abs(difference) <= 1e-7
    same &= np.max(abs(bary-base_bary), axis=-1) <= 1e-6
    return same & (rgb.max(-1) == 0) & (base_rgb.max(-1) > 0) & (source == 255) & (base_source != 255)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    baseline, candidate, output = (p.resolve() for p in (args.baseline,args.candidate,args.output))
    assert not output.exists() and len(output.parts)>=4
    for folder in [baseline,candidate]:
        assert output != folder and output not in folder.parents and folder not in output.parents
    requests=[]; results=[]; checked={}; images=[]; sources=[]; meshes=[]; raycasts=[]
    for folder in [baseline,candidate]:
        qpath=folder.parent.parent/'request.json'; receipt=read(folder/'complete.json')
        assert receipt['request_sha256']==sha(qpath)
        for name,h in receipt['hashes'].items():
            assert sha(folder/name)==h,name; checked[str(folder/name)]=h
        checked[str(qpath)]=sha(qpath); q=read(qpath); r=read(folder/'result.json')
        assert not r['target_rgb_read'] and not r['rgb_averaging']
        assert not set(r['source_cameras']) & {'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
        assert sha(r['mesh_path'])==r['mesh_sha256']; checked[r['mesh_path']]=r['mesh_sha256']
        mesh=o3d.io.read_triangle_mesh(r['mesh_path']); v=np.asarray(mesh.vertices); t=np.asarray(mesh.triangles)
        meshes.append((v,t)); requests.append(q); results.append(r)
        images.append(np.asarray(Image.open(folder/'prediction_native.png').convert('RGB')))
        sources.append(np.asarray(Image.open(folder/'source_ids.png')))
        raycasts.append(camera_depth(Scene2(v.astype(np.float32),t),r['camera']))
    for key in ['recipe','source_rows','profiles_sha256','exposure_sha256']:
        assert requests[0][key]==requests[1][key],key
    for key in ['camera','frame_id','source_cameras','fixed_exposure']:
        assert results[0][key]==results[1][key],key
    ov,ot=meshes[0]; v,t=meshes[1]
    np.testing.assert_array_equal(v[:len(ov)],ov); np.testing.assert_array_equal(t[:len(ot)],ot)
    bd,bi,bb=raycasts[0]; d,i,b=raycasts[1]
    mask=selection(images[0],images[1],sources[0],sources[1],bd,d,bi,i,bb,b,len(ot))
    rgb=images[1].copy(); source=sources[1].copy()
    rgb[mask]=images[0][mask]; source[mask]=sources[0][mask]
    np.testing.assert_array_equal(rgb[~mask],images[1][~mask])
    output.mkdir()
    Image.fromarray(rgb).save(output/'prediction_native.png')
    Image.fromarray(np.rot90(rgb)).save(output/'frame.png')
    Image.fromarray(source).save(output/'source_ids.png')
    np.savez_compressed(output/'evidence.npz',mask=mask,face_ids=i[mask],base_depth=bd[mask],depth=d[mask],
        base_bary=bb[mask],bary=b[mask],base_source=sources[0][mask],source_before=sources[1][mask])
    save(output/'result.json',dict(backoff_pixels=int(mask.sum()), baseline=str(baseline),candidate=str(candidate),
        input_hashes=checked, script_sha256=sha(__file__),
        original_prefix_exact=True, same_original_triangle_and_intersection_required=True,
        baseline_train_reprojection_reused=True, new_geometry_not_filled=True, target_depth_unchanged=True,
        masks_and_calibration_unchanged=True, no_rgb_averaging=True, target_rgb_used=False,
        experimental_visibility_backoff_not_physical_occlusion_proof=True,production_accepted=False,
        hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print('backoff pixels',int(mask.sum()),flush=True)


if __name__=='__main__': main()
