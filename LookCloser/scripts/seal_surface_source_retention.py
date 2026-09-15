"""Audit 16 matched source-selection renders and record the actual visual review."""
from pathlib import Path
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from review_jaw_repair_transfer import verified_image
import study_surface_ray_source_prior as initial
import transfer_surface_source_retention as transfer


def seal():
    root=initial.ROOT;hashes={};records=[]
    a=read(root/'supervisor_result.json');b=read(transfer.ROOT/'supervisor_result.json')
    assert a['all_six_passed'] and b['all_ten_passed']
    assert len(a['finished'])==6 and len(b['finished'])==10
    assert all(r['exit_code']==0 for r in a['finished']+b['finished'])
    cases=[(root/f/m,initial.baseline(f,initial.VIEW),f) for f in initial.FRAMES for m in initial.MODES]
    cases += [(transfer.ROOT/c/m,old,f) for c,(f,old) in transfer.CASES.items() for m in transfer.MODES]
    for out,old,frame in cases:
        q=initial.engine.verify_request(out);pq=read(old/'request.json')
        entry=next(r for r in q['inventory'] if r['frame_id']==frame)
        parent=next(r for r in pq['inventory'] if r['frame_id']==frame)
        assert entry==parent and len(q['inventory'])==1
        for key in ['profiles_sha256','exposure_sha256','calibration_sha256']:assert q[key]==pq[key]
        assert sha(entry['mesh'])==entry['mesh_sha256'] and sha(entry['metadata'])==entry['metadata_sha256']
        pred,pr=verified_image(out,frame);orig,orr=verified_image(old,frame)
        for key in ['camera','source_cameras','mesh_sha256','fixed_exposure']:assert pr[key]==orr[key]
        np.testing.assert_array_equal(np.load(out/'frames'/frame/'target_depth.npz')['depth'],np.load(old/'frames'/frame/'target_depth.npz')['depth'])
        assert not ((orig.max(2)>0)&(pred.max(2)==0)).any()
        receipt=read(out/'frames'/frame/'complete.json')
        for name,digest in receipt['hashes'].items():hashes[str(out/'frames'/frame/name)]=digest
        for path in [out/'request.json',out/'frames'/frame/'complete.json',old/'request.json',old/'frames'/frame/'complete.json',Path(entry['mesh'])]:hashes[str(path)]=sha(path)
        records.append(dict(output=str(out),frame=frame,depth_exact=True,no_new_black=True))
    viewed=[root/f/'review'/n for f in initial.FRAMES for n in ['jaw.png','overview.png']]
    viewed += [transfer.ROOT/c/'head.png' for c in transfer.CASES if c!='heldout']
    viewed += [transfer.ROOT/'heldout'/(m+'_heldout.png') for m in transfer.MODES]
    assert len(viewed)==10
    visual=dict(status='reviewed_local_texture_improvement_with_remaining_geometry_defects',
        inspected_images={str(p):sha(p) for p in viewed},
        relative_retention='Native late jaw line substantially weaker; a dotted residual remains.',
        ray_angle_alone='No material improvement to the diagnosed native jaw seam.',
        ray_plus_retention='Better late held-out LPIPS but lower PSNR; early moving lips look softer. Not promoted.',
        known_residuals=['crown hole','rough hair and shoulder boundary','small lipstick/hand membrane','dotted jaw seam'],
        temporal_flicker_evaluated=False,geometry_repaired=False,artifact_free_approval=False,production_promoted=False)
    atomic_json(root/'visual_review.json',visual)
    metrics=read(transfer.ROOT/'heldout/metrics.json')
    assert not metrics['full_frame_metrics'] and not metrics['loss_reported'] and metrics['depth_byte_equal']
    for row in metrics['rows']:
        assert all(np.isfinite(row[k]) for k in ['face_psnr','face_ssim','face_lpips'])
        assert sha(transfer.ROOT/'heldout'/row['mode']/'frames/001193/frame.png')==row['prediction_sha256']
    for p in viewed+[root/'comparison.json',transfer.ROOT/'comparison.json',transfer.ROOT/'heldout/metrics.json',root/'visual_review.json',root/'supervisor_result.json',transfer.ROOT/'supervisor_result.json']:
        hashes[str(p)]=sha(p)
    for path,digest in hashes.items():assert sha(path)==digest
    atomic_json(root/'artifact_manifest.json',dict(status='sixteen_controls_verified_not_production_promoted',records=records,hashes=hashes,
        script_sha256=sha(__file__),goal_complete=False,geometry_changed=False))
    print('Verified',len(records),'renders;',len(hashes),'artifact bindings; production unchanged',flush=True)


if __name__=='__main__':seal()
