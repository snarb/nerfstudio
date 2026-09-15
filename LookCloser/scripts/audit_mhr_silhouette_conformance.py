"""Replay the frozen 54-camera fit and evaluate fixed-availability cohorts."""
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
import fit_mhr_silhouette_conformance as fit
from study_multiview_face_prior import read, save, sha
from triangulate_face_prior import quantiles


def main():
    root = fit.ROOT
    assert not (root/'audit.json').exists()
    q = read(root/'protocol.json'); result=read(root/'result.json'); saved=np.load(root/'fit.npz')
    assert sha(fit.__file__) == q['script_sha256']
    assert sha(root/'protocol.json') == result['protocol_sha256']
    bindings={}
    def check(path,digest):
        assert sha(path)==digest,str(path);bindings[str(path)]=digest
    for p,h in q['input_hashes'].items():check(p,h)
    for p,h in q['helpers'].items():check(Path(fit.__file__).with_name(p),h)
    for p,h in result['hashes'].items():check(root/p,h)
    check(q['original_mesh'],q['original_mesh_sha256'])
    _,rows,masks,names,evidence,validation=fit.prepare()
    assert evidence==q['evidence']
    trainrows=[r for r,v in zip(rows,validation) if not v]
    assert [r['physical_camera'] for r in trainrows]==q['fit_cameras']
    assert not (set(q['fit_cameras']) & set(q['validation_cameras']))
    obs=np.load(fit.SOURCE/'anchors.npz'); take=~validation[obs['camera']]
    sdfs=[]
    for row in rows:
        mask=masks[names.index(row['physical_camera'])].astype(bool)
        sdfs.append((distance_transform_edt(~mask)-distance_transform_edt(mask)).astype(np.float32))
    fit.ROOT=root/'audit_replay';fit.ROOT.mkdir(exist_ok=False)
    try:
        replay,history=fit.optimize(saved['baseline'],saved['triangles'],saved['neutral'],
            obs['points'][take],obs['normals'][take],obs['neck'][take],trainrows,[s for s,v in zip(sdfs,validation) if not v])
    finally:fit.ROOT=root
    np.testing.assert_array_equal(replay,saved['vertices'])
    assert history==result['history']
    active=saved['active'];np.testing.assert_array_equal(replay[~active],saved['baseline'][~active])
    native=[]
    for vertices in [saved['baseline'][active],replay[active]]:
        grid=np.full((len(rows),len(vertices)),np.nan)
        for ci,(ids,values,_) in enumerate(fit.silhouette_samples(vertices,rows,sdfs)):grid[ci,ids]=values
        native.append(grid)
    before,after=native;common=np.isfinite(before)&np.isfinite(after)
    cohorts={}
    for split,selection in [('train',~validation),('validation',validation)]:
        for region,select in [('all',np.ones(active.sum(),bool)),('front',saved['neutral'][active,2]>=0),('back',saved['neutral'][active,2]<0)]:
            chosen=common&selection[:,None]&select[None,:]
            a,b=np.maximum(before[chosen]-2,0),np.maximum(after[chosen]-2,0)
            cohorts[split+'_'+region]=dict(samples=len(a),before_excess=quantiles(a),after_excess=quantiles(b),
                before_mean_excess=float(a.mean()),after_mean_excess=float(b.mean()),
                before_outside=int((a>0).sum()),after_outside=int((b>0).sum()))
    head_tri=saved['triangles'][(saved['neutral'][saved['triangles'],1]>140).all(1)]
    associations={}
    for arm in ['baseline','silhouette']:
        ids=saved[arm+'_nearest_triangle'];uv=saved[arm+'_nearest_uv'];bary=np.c_[1-uv.sum(1),uv]
        anatomical=np.sum(saved['neutral'][head_tri[ids]]*bary[:,:,None],axis=1)
        associated_active=saved['active'][head_tri[ids]].any(1)
        for split,selection in [('train',take),('validation',~take)]:
            chosen=selection&associated_active
            associations[arm+'_'+split+'_touching_active']=dict(count=int(chosen.sum()),
                distance=quantiles(saved[arm+'_distance'][chosen]),
                neutral_y=quantiles(anatomical[chosen,1]),
                actual_lower_band=int(((anatomical[:,1]>135)&(anatomical[:,1]<153)&selection).sum()))
    np.savez_compressed(root/'audit_evidence.npz',before_sdf=before,after_sdf=after,common_available=common,validation=validation)
    review=read(root/'review_v2/result.json')
    check(Path(__file__).with_name('review_mhr_silhouette_conformance.py'),review['script_sha256'])
    for p,h in review['input_hashes'].items():check(p,h)
    for item in review['files']:check(item['path'],item['sha256'])
    save(root/'audit.json',dict(status='passed',script_sha256=sha(__file__),checked_bindings=bindings,
        exact_fit_replay=True,fit_camera_count=54,reserved_camera_count=8,fit_anchor_count=int(take.sum()),
        reserved_anchor_count=int((~take).sum()),outside_active_exact=True,original_mesh_hash_verified=True,
        common_availability_cohorts=cohorts,associated_active_anchors=associations,
        evidence_sha256=sha(root/'audit_evidence.npz'),production_accepted=False))
    print('audit passed',len(bindings),'bindings; exact 54-camera replay',flush=True)


if __name__=='__main__':main()
