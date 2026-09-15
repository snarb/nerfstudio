"""Replay interval grids and verify refined mask-boundary witnesses."""
from pathlib import Path
import argparse
from collections import Counter
import numpy as np
from scipy.ndimage import map_coordinates
from probe_mhr_residual_depth_intervals import MaskProbe,prepare,project
from run_local_mhr_completion import read,save,sha,require


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path);p.add_argument('--tests',type=Path);a=p.parse_args();root=a.output
    require(not (root/'final_seal.json').exists(),'Already sealed')
    request=read(root/'request.json');result=read(root/'result.json');require(result['request_sha256']==sha(root/'request.json'),'Request changed')
    producer=Path(__file__).with_name('probe_mhr_residual_depth_intervals.py');require(sha(producer)==request['script_sha256'],'Producer changed')
    checked={}
    for path,h in request['input_hashes'].items():require(sha(path)==h,'Changed input');checked[path]=h
    for name,h in result['outputs'].items():require(sha(root/name)==h,'Changed output')
    _,rows,masks,names,binding,_=prepare();require(binding==request['mask_binding'],'Mask override changed')
    import joint_temporal_texture as calibration
    _,metadata=calibration.geometry_paths(request['frame'])
    calibration_paths=[calibration.CALIBRATION,calibration.SOURCE/request['frame']/'transforms.json',metadata,Path(binding['depth_receipt']['transforms'])]
    for path in calibration_paths:checked[str(path)]=sha(path)
    save(root/'cameras_replay.json',dict(cameras=rows,input_hashes={str(p):sha(p) for p in calibration_paths},
        role='Exact 62-camera list replayed against every saved grid classification; supplementary provenance, no new inference input'))
    probe=MaskProbe(rows,masks,names);native_checks=0;max_sdf_difference=0.;class_disagreements=0
    for label in ['coarse','verification']:
        q=np.load(root/(label+'.npz'));points=q['prior_points'][:,None]+q['offsets'][None,:,None]*q['rays'][:,None,3:]
        av,b,s=probe.evaluate(points);shape=q['available'].shape
        np.testing.assert_array_equal(av.reshape(shape),q['available']);np.testing.assert_array_equal(b.reshape(shape),q['binary_outside'])
        np.testing.assert_array_equal(s.reshape(shape),q['signed_distance'])
        # Independent SDF interpolator (SciPy) and direct nearest-mask indexing.
        uv,z=project(points.reshape(-1,3),rows)
        for ci in range(len(rows)):
            ids=np.flatnonzero(av[ci]);p=uv[ci,ids];xx=np.rint(p[:,0]).astype(int);yy=np.rint(p[:,1]).astype(int)
            np.testing.assert_array_equal(~probe.masks[ci][yy,xx],b[ci,ids])
            independent=map_coordinates(probe.sdfs[ci],[p[:,1],p[:,0]],order=1,prefilter=False)
            max_sdf_difference=max(max_sdf_difference,float(np.max(abs(independent-s[ci,ids]),initial=0)))
            class_disagreements+=int(((independent>0)!=(s[ci,ids]>0)).sum());native_checks+=len(ids)
        print('grid replay',label,points.shape[:2],flush=True)
    require(class_disagreements==0,'SDF sign classification disagreement')
    q=np.load(root/'verification.npz');boundaries=[]
    for qi,row in enumerate(result['records']):
        for mode,entry in row['modes'].items():
            for ii,interval in enumerate(entry['intervals']):
                for side in ['left','right']:
                    bracket=interval[side+'_bracket'];values=[];cams=[]
                    for offset in bracket:
                        point=q['prior_points'][qi]+offset*q['rays'][qi,3:];av,b,s=probe.evaluate(point)
                        outside=b if mode=='binary' else s>0
                        values.append(bool(av.sum()>=2 and not outside.any()))
                        cams.append([rows[i]['physical_camera'] for i in np.flatnonzero(outside[:,0])])
                    if interval[side+'_truncated']:require(values==[True,True],'Truncated endpoint not feasible')
                    else:
                        require(bracket[1]-bracket[0]<=request['boundary_refinement_bracket'],'Unrefined boundary')
                        require(values==([False,True] if side=='left' else [True,False]),'Boundary does not bracket transition')
                    boundaries.append(dict(pixel=row['pixel'],cohort=row['cohort'],mode=mode,interval=ii,side=side,bracket=bracket,accepted=values,veto_cameras=cams))
    summary={}
    for cohort in ['residual','existing_hit_control']:
        rs=[r for r in result['records'] if r['cohort']==cohort];summary[cohort]={}
        for mode in ['binary','bilinear_sdf']:
            entries=np.array([np.mean(r['modes'][mode]['intervals'][0]['left_bracket']) for r in rs])
            active=Counter(c for b in boundaries if b['cohort']==cohort and b['mode']==mode and b['side']=='left' for c in b['veto_cameras'][0])
            summary[cohort][mode]=dict(rays=len(rs),feasible=sum(r['modes'][mode]['feasible'] for r in rs),
                zero_feasible=sum(r['modes'][mode]['zero_feasible'] for r in rs),entry_min_median_max=[float(entries.min()),float(np.median(entries)),float(entries.max())],
                entry_boundary_veto_counts=dict(active),zero_veto_counts=dict(Counter(c for r in rs for c in r['modes'][mode]['zero_veto_cameras'])))
    save(root/'boundary_audit.json',dict(status='passed',sample_camera_classifications=native_checks,independent_sdf_sign_disagreements=class_disagreements,
        independent_sdf_absolute_difference_max=max_sdf_difference,boundaries=boundaries,summary=summary,
        result_sha256=sha(root/'result.json'),script_sha256=sha(__file__),input_hashes=checked))
    if a.report is not None:
        require(a.tests is not None,'Tests required for final seal');visual=read(root/'visual_review.json')
        require(visual['reviewer']=='LLM' and not visual['geometry_changed'],'Visual gate absent')
        for path,h in visual['viewed_images'].items():require(sha(path)==h,'Reviewed image changed');checked[path]=h
        for path,h in visual.get('retained_failure_receipts',{}).items():require(sha(path)==h,'Failure evidence changed');checked[path]=h
        for name,h in request['helpers'].items():path=producer.with_name(name);require(sha(path)==h,'Changed helper');checked[str(path)]=h
        for path in [producer,Path(__file__),a.report,a.tests]:checked[str(path.resolve())]=sha(path)
        save(root/'final_seal.json',dict(status='passed',checked_bindings=checked,
            inventory={str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*')) if f.is_file()},
            target_scan_used_for_fit=False,geometry_changed=False,report_sha256=sha(a.report)))
    print(summary,flush=True)


if __name__=='__main__':main()
