"""Opt-in 001193 correction comparison on shared canonical vertex IDs."""
from pathlib import Path
import argparse
import inspect
import numpy as np
import review_mhr_guarded_corrections as review
from study_multiview_face_prior import read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.output.resolve()
    arms={'margin2':Path('/mnt/data/dec5_mhr_silhouette_convergence'),
          'reference':args.reference.resolve(),'candidate':args.prior.resolve()}
    assert not root.exists() and len(root.parts)>=4
    fits=[]
    for path in arms.values():
        assert (path/'result.json').exists(),f'Fit is not terminal: {path}'
        assert read(path/'protocol.json')['frame']=='001193'
        assert root!=path and root not in path.parents and path not in root.parents
        fits.append(np.load(path/'fit.npz'))
    for data in fits[1:]:
        for key in ['triangles','neutral']:np.testing.assert_array_equal(data[key],fits[0][key])
    common=np.logical_and.reduce([data['active'] for data in fits])
    assert common.any()
    source=inspect.getsource(review.main)
    old="v,t,active=data['vertices'],data['triangles'],data['active']"
    assert source.count(old)==1
    generated=source.replace(old,old+'\n        active=COMMON_ACTIVE')
    ns=dict(review.__dict__,ROOT=root,ARMS=arms,COMMON_ACTIVE=common)
    exec(compile(generated,'<shared_canonical_vertex_correction_review>','exec'),ns)
    ns['main']()
    save(root/'runtime_config.json',dict(arms={k:str(v) for k,v in arms.items()},output=str(root),
        common_vertex_ids=np.flatnonzero(common).tolist(),generated_source=generated,
        wrapper_sha256=sha(__file__),review_helper_sha256=sha(review.__file__),
        fit_inputs_unchanged=True,posthoc_ray_tests_not_admission=True))


if __name__=='__main__':main()
