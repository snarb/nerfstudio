"""Frozen native comparisons plus posthoc attribution of missing-ray facets."""
from pathlib import Path
import numpy as np
from admit_mhr_silhouette_patch import OUT,CANDIDATES,PRIOR,ARM
from admit_mhr_local_patch_depth import Scene2
from study_multiview_face_prior import read,save,sha


def main():
    import open3d as o3d
    import diffusion_mesh_repair
    diffusion_mesh_repair.scene_for=Scene2
    import review_mhr_depth_admitted_patches as native
    import review_mhr_admission_branch_difference as branches
    proof=dict(script_sha256=sha(__file__),helpers={str(Path(module.__file__)):sha(module.__file__) for module in [native,branches]},
               native_scene_threads=2,only_review_paths_and_execution_threads_changed=True,target_used_posthoc_only=True)
    save(OUT/'review_wrapper.json',proof)
    native.review(OUT)
    branches.ROOT=OUT
    branches.main()
    folder=OUT/'facet_attribution';folder.mkdir(exist_ok=False)
    path=PRIOR/'locality/silhouette.npz';points=np.load(path)['points']
    domain=np.load(CANDIDATES/ARM/'domain_evidence.npz');v=domain['subdivided_vertices'];t=domain['subdivided_triangles'];keep=domain['retained']
    hit=Scene2(v,t).compute_closest_points(o3d.core.Tensor(points.astype(np.float32)));ids=hit['primitive_ids'].numpy()
    distance=np.linalg.norm(hit['points'].numpy()-points,axis=1);exact=distance<1e-6
    mapping=np.full(len(t),-1,int);mapping[keep]=np.arange(int(keep.sum()));raw=mapping[ids];available=raw>=0
    a=np.load(OUT/ARM/'admission.npz');lookup=np.full(len(a['mask_support']),-1,int);lookup[a['semantic_ids']]=np.arange(len(a['semantic_ids']))
    semantic=np.full(len(points),-1,int);semantic[available]=lookup[raw[available]];ok=semantic>=0
    strict=np.zeros(len(points),bool);interpolated=strict.copy();free=strict.copy()
    strict[ok]=a['strict'][semantic[ok]];interpolated[ok]=a['interpolated'][semantic[ok]];free[ok]=a['trusted_free'][:,semantic[ok]].any(axis=(0,2))
    final={}
    for branch in ['strict','interpolated']:
        retained=np.load(OUT/ARM/branch/'evidence.npz')['retained_proposal_ids']
        final[branch]=int((exact&available&np.isin(raw,retained)).sum())
    np.savez_compressed(folder/'evidence.npz',points=points,nearest_subdivided_triangle=ids,distance=distance,raw_id=raw,
        semantic_index=semantic,strict_initial=strict,interpolated_initial=interpolated,trusted_free=free)
    record=dict(points=len(points),matching_safe_band_points=int(exact.sum()),
        all_vertices_local=int((exact&domain['local_before_centroid_gate'][ids]).sum()),
        rejected_by_centroid_gap=int((exact&domain['local_before_centroid_gate'][ids]&~keep[ids]).sum()),
        matching_raw=int((exact&available).sum()),semantic_admitted=int((exact&ok).sum()),
        strict_initial=int((exact&strict).sum()),interpolated_initial=int((exact&interpolated).sum()),
        trusted_free_veto=int((exact&free).sum()),matching_final=final,
        note='Facet attribution at stored prior points; native missing-ray counts remain authoritative near numerical facet boundaries.')
    save(folder/'result.json',dict(record=record,script_sha256=sha(__file__),target_used_posthoc_only=True,
        input_hashes={str(p):sha(p) for p in [path,CANDIDATES/ARM/'domain_evidence.npz',OUT/ARM/'admission.npz']},
        production_accepted=False))
    print(record,flush=True)


if __name__=='__main__':main()
