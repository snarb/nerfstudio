"""Attribute an unchanged front-under-chin hole, posthoc only, no gate changes."""
import numpy as np
from PIL import Image
from run_mhr_production_patch_control import ROOT, OUT, CANDIDATES, ARM, FRAME, builder
from admit_mhr_local_patch_depth import Scene2
from bake_joint_temporal_mesh import camera_depth
from study_multiview_face_prior import read, save, sha


def main():
    import open3d as o3d
    folder = OUT/'rgb/F004_E/baseline/frames'/FRAME
    receipt = read(folder/'complete.json')
    for name, digest in receipt['hashes'].items():
        assert sha(folder/name) == digest
    camera = read(folder/'result.json')['camera']
    image = np.asarray(Image.open(folder/'frame.png'))
    depth = np.rot90(np.load(folder/'target_depth.npz')['depth'])
    # Rectangle chosen after looking at the completed native diagnostic, never
    # supplied to fitting, reconstruction, patch proposals or admission.
    box = (670, 1135, 720, 1175)
    x0, y0, x1, y1 = box
    yy, xx = np.where((image[y0:y1, x0:x1].max(2) == 0) & (depth[y0:y1, x0:x1] == 0))
    yy, xx = yy+y0, xx+x0
    assert len(xx) > 0
    fit = np.load(builder.PRIOR/'fit.npz')
    v, t = fit['vertices'], fit['triangles']
    d, ids, bary = camera_depth(Scene2(v, t), camera)
    pd = np.rot90(d)[yy, xx]
    faces = np.rot90(ids)[yy, xx]
    uv = np.rot90(bary)[yy, xx]
    hit = np.isfinite(pd)
    points = (v[t[faces[hit]]] * np.column_stack((1-uv[hit].sum(1), uv[hit]))[..., None]).sum(1)
    domain_path = CANDIDATES/ARM/'domain_evidence.npz'
    domain = np.load(domain_path)
    closest = Scene2(domain['subdivided_vertices'], domain['subdivided_triangles']).compute_closest_points(
        o3d.core.Tensor(points.astype(np.float32)))
    nearest = closest['primitive_ids'].numpy()
    distance = np.linalg.norm(closest['points'].numpy()-points, axis=1)
    exact = distance < 1e-6
    kept = domain['retained']
    raw_lookup = np.full(len(kept), -1, int)
    raw_lookup[kept] = np.arange(kept.sum())
    raw = raw_lookup[nearest]
    admission_path = OUT/ARM/'admission.npz'
    admitted = np.load(admission_path)
    semantic = np.full(len(admitted['mask_support']), -1, int)
    semantic[admitted['semantic_ids']] = np.arange(len(admitted['semantic_ids']))
    available = exact & (raw >= 0)
    semantic_available = np.zeros(len(points), bool)
    semantic_available[available] = semantic[raw[available]] >= 0
    final = {}
    counts = {}
    for branch in ['strict', 'interpolated']:
        retained = np.load(OUT/ARM/branch/'evidence.npz')['retained_proposal_ids']
        final[branch] = int((available & np.isin(raw, retained)).sum())
        path = OUT/'rgb/F004_E'/branch/'frames'/FRAME/'target_depth.npz'
        rd = np.rot90(np.load(path)['depth'])
        counts[branch] = int((rd[yy, xx] <= 0).sum())
    stat = dict(baseline_missing=len(xx), remaining_missing=counts, prior_first_hits=int(hit.sum()),
        unsafe_first_hit_parent=int(domain['unsafe_parent'][faces[hit]].sum()),
        matching_safe_subdivision=int(exact.sum()),
        local_before_centroid=int((exact & domain['local_before_centroid_gate'][nearest]).sum()),
        centroid_rejected=int((exact & domain['local_before_centroid_gate'][nearest] & ~kept[nearest]).sum()),
        raw_proposal_available=int(available.sum()), semantic_admitted=int(semantic_available.sum()),
        final_first_hit_facet_available=final)
    dest = ROOT/'residual_hole'
    dest.mkdir(exist_ok=False)
    np.savez_compressed(dest/'evidence.npz', portrait_xy=np.c_[xx, yy], prior_depth=pd,
        prior_face=faces, prior_points=points, nearest_safe_facet=nearest,
        nearest_safe_distance=distance, raw_proposal_id=raw, semantic_available=semantic_available)
    bindings = [folder/'complete.json', folder/'result.json', builder.PRIOR/'fit.npz',
                domain_path, admission_path]
    for branch in ['strict', 'interpolated']:
        bindings.extend([OUT/ARM/branch/'evidence.npz', OUT/'rgb/F004_E'/branch/'frames'/FRAME/'complete.json'])
    save(dest/'result.json', dict(statistics=stat, posthoc_box=list(box),
        script_sha256=sha(__file__), input_hashes={str(p): sha(p) for p in bindings},
        evidence_sha256=sha(dest/'evidence.npz'), posthoc_only=True, gate_changed=False,
        note='First-hit facet attribution, not proof no deeper plausible surface exists.'))
    print(stat, flush=True)


if __name__ == '__main__':
    main()
