"""Post-hoc attribution of rejected local facets near stored missing-ray points."""
from pathlib import Path
import argparse
import numpy as np
from build_train_hair_semantics import read, sha, write
from build_mhr_local_patch_candidates import PRIOR, ARMS


def main(candidates,admission):
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    dest=admission/'target_admission_diagnosis'; dest.mkdir(exist_ok=False)
    records=[]; inputs={}
    for arm in ARMS:
        qp=PRIOR/'probe_smooth025_smooth100_smooth400'/(arm+'.npz')
        mp=candidates/arm/'local_raw.ply'; ep=candidates/arm/'proposal_evidence.npz'; ap=admission/arm/'admission.npz'
        for p in [qp,mp,ep,ap]: inputs[str(p)]=sha(p)
        query=np.load(qp)['target_prior_points']; mesh=o3d.io.read_triangle_mesh(str(mp))
        v=np.asarray(mesh.vertices); proposals=np.load(ep)['proposals']; a=np.load(ap)
        scene=scene_for(v,proposals); cp=scene.compute_closest_points(o3d.core.Tensor(query.astype(np.float32)))
        ids=cp['primitive_ids'].numpy(); dist=np.linalg.norm(query-cp['points'].numpy(),axis=1)
        exact=dist<1e-6; ms=a['mask_support'][ids]; mo=a['mask_outside'][ids]
        lookup=np.full(len(proposals),-1,int); lookup[a['semantic_ids']]=np.arange(len(a['semantic_ids']))
        sem=lookup[ids]; available=sem>=0; vi=sem[available]
        strict=np.zeros(len(query),bool); prior=np.zeros(len(query),bool); free=np.zeros(len(query),bool)
        strict[available]=a['strict'][vi]; prior[available]=a['certified_prior'][vi]
        free[available]=a['trusted_free'][:,vi].any(axis=(0,2))
        record=dict(arm=arm,stored_prior_hit_points=len(query),matching_raw_facets=int(exact.sum()),
            matching_with_mask_support=int((exact&(ms>=2)).sum()),
            matching_without_mask_disagreement=int((exact&(mo==0)).sum()),
            matching_semantic_admitted=int((exact&available).sum()),
            matching_strict_admitted=int((exact&strict).sum()),
            matching_interpolation_certified=int((exact&prior).sum()),
            matching_semantic_with_trusted_free=int((exact&available&free).sum()),
            semantic_sample_votes=a['votes'][vi[exact[available]]].tolist())
        np.savez_compressed(dest/(arm+'.npz'),query=query,nearest_raw_proposal=ids,raw_distance=dist,
            mask_support=ms,mask_outside=mo,semantic_index=sem,strict=strict,certified_prior=prior,free=free)
        records.append(record); print({k:value for k,value in record.items() if k!='semantic_sample_votes'},flush=True)
    write(dest/'result.json',dict(arms=records,input_hashes=inputs,script_sha256=sha(__file__),
        target_used_posthoc_only=True,production_accepted=False,geometry_changed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--candidates',type=Path,default=Path('/mnt/data/dec5_mhr_local_patch_candidates'))
    p.add_argument('--admission',type=Path,default=Path('/mnt/data/dec5_mhr_local_patch_admission'))
    a=p.parse_args(); main(a.candidates,a.admission)
