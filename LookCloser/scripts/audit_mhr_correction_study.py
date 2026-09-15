"""Seal the bounded correction/proposal evidence, without claiming a repair."""
from pathlib import Path
import numpy as np
import open3d as o3d
from guard_mhr_anatomical_correction import all_pairs
from fit_mhr_guarded_correction import strict_pairs
from study_multiview_face_prior import read,save,sha

ROOT=Path('/mnt/data/dec5_mhr_correction_study')
PRIOR=Path('/mnt/data/dec5_mhr_anatomical_correction')
CANDIDATE=Path('/mnt/data/dec5_mhr_anatomical_candidates')


def main():
    assert not ROOT.exists();ROOT.mkdir()
    checked={}
    def check(path,digest):
        assert sha(path)==digest,str(path);checked[str(path)]=digest
    seal=read(PRIOR/'final_seal.json')
    assert seal['status']=='passed' and not seal['production_accepted']
    for p,h in seal['inventory'].items():check(PRIOR/p,h)
    for p,h in seal['checked_bindings'].items():check(p,h)
    request=read(CANDIDATE/'request.json')
    assert request['new_prior_recipe_explicit'] and not request['prior_compatibility_with_margin2_cli']
    for key in ['production_mesh','metadata','raw_mesh']:
        path=request['raw_mesh_not_used_as_base'] if key=='raw_mesh' else request[key]
        check(path,request[key+'_sha256'])
    q=read(CANDIDATE/'candidates/request.json')
    for p,h in q['input_hashes'].items():check(p,h)
    branch=CANDIDATE/'candidates'/q['arms'][0];r=read(branch/'result.json')
    assert r['request_sha256']==sha(CANDIDATE/'candidates/request.json')
    for p,h in r['hashes'].items():check(branch/p,h)
    original=o3d.io.read_triangle_mesh(request['production_mesh'])
    raw=o3d.io.read_triangle_mesh(str(branch/'local_raw.ply'))
    ov,ot=np.asarray(original.vertices),np.asarray(original.triangles)
    v,t=np.asarray(raw.vertices),np.asarray(raw.triangles)
    np.testing.assert_array_equal(v[:len(ov)],ov);np.testing.assert_array_equal(t[:len(ot)],ot)
    np.testing.assert_array_equal(t[len(ot):],np.load(branch/'proposal_evidence.npz')['proposals'])
    probe=read(CANDIDATE/'posthoc_probe/result.json');a=np.load(CANDIDATE/'posthoc_probe/evidence.npz')
    for p,h in probe['input_hashes'].items():check(p,h)
    check(CANDIDATE/'posthoc_probe/evidence.npz',probe['evidence_sha256'])
    assert probe['no_depth_admission_performed'] and probe['semantic_only_not_publishable']
    assert len(a['portrait_xy'])==30 and len(np.unique(a['portrait_xy'],axis=0))==30
    np.testing.assert_array_equal(a['semantic_keep'],(a['mask_support']>=2)&(a['mask_outside']==0))
    assert int(a['semantic_keep'].sum())==probe['semantic_pass']
    for name,record in probe['records'].items():
        assert int(np.isfinite(a[name+'_depth']).sum())==record['residual_hits']
        assert record['residual_hits']+record['residual_misses']==30
    # Reconstruct the last proposed endpoint from accepted displacement/factor.
    # This is a floating-point diagnostic, not an exact saved solver solution.
    f=np.load(PRIOR/'fit.npz');history=read(PRIOR/'result.json');factor=history['guards'][-1]['factor']
    previous=np.load(PRIOR/'iterates/039.npz')['vertices'];last=f['vertices']
    proposed=previous+(last-previous)/factor
    new=all_pairs(proposed,f['triangles'])-all_pairs(f['baseline'],f['triangles'])
    strict=strict_pairs(proposed,f['triangles'])-strict_pairs(f['baseline'],f['triangles'])
    pairs=np.array(sorted(new),int).reshape(-1,2)
    np.savez_compressed(ROOT/'last_proposal_diagnosis.npz',proposed=proposed,new_pairs=pairs,
        neutral_triangles=f['neutral'][f['triangles'][pairs]])
    solver_pairs=set(map(tuple,read(PRIOR/'qp_progress.json')['solves'][-1]['contact_pairs']))
    diagnosis=dict(reconstructed_from_iterates_not_exact_solver_array=True,factor=factor,
        new_all_pairs=[list(map(int,p)) for p in sorted(new)],new_strict_pairs=len(strict),
        pairs_already_in_proposal_contact_set=[list(map(int,p)) for p in sorted(new&solver_pairs)])
    video=Path('/mnt/data/dec5_cinematic_wide_spiral_6k_output_v1/presentation/video.mp4')
    check(video,'447bf60f513c495ea9af997007f529a9b414d54d27a995511ecacc36b024353e')
    for root in [PRIOR,CANDIDATE,Path('/mnt/data/dec5_mhr_anatomical_review')]:
        for p in root.rglob('*'):
            if p.is_file():checked[str(p)]=sha(p)
    check(Path(__file__).resolve(),sha(__file__))
    save(ROOT/'audit.json',dict(status='passed_bounded_evidence_audit_not_repair_acceptance',
        input_hashes=checked,original_prefix_exact=True,probe=probe['records'],last_proposal=diagnosis,
        prior_all_pair_guard_passed=True,depth_admission_performed=False,RGB_candidate_rendered=False,
        production_modified=False,video_unchanged=True,hole_fully_repaired=False,
        evidence_sha256=sha(ROOT/'last_proposal_diagnosis.npz')))
    print('checked',len(checked),'bindings; proposal',diagnosis,'coverage',probe['records'],flush=True)


if __name__=='__main__':main()
