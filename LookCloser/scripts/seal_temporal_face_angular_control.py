"""Seal the finite dynamic control, preserving limitations of the full goal."""
from datetime import datetime
from pathlib import Path
import numpy as np
from run_temporal_face_angular_control import ROOT,BASE,CLIPS,FRAMES,REUSE,verify
from review_temporal_face_angular_control import frame_pair
from build_train_hair_semantics import read,write,sha


def main():
    q=verify();out=ROOT/'artifact_manifest.json';assert not out.exists()
    supervisor=read(ROOT/'supervisor_result.json')
    assert supervisor['all_passed'] and not supervisor['not_started']
    assert len(supervisor['finished'])==33 and all(x['exit_code']==0 for x in supervisor['finished'])
    notes=read(ROOT/'visual_notes.json');assert not notes['normal_speed_playback_claimed'] and not notes['production_promoted']
    hashes={};meshes=[];camera_centers=[];source_hashes=[];summaries={}
    for inv in q['inventory']:
        f=inv['frame'];folder=Path(inv['output']);a,b,x,y,r=frame_pair(inv)
        br=read(BASE/'frames'/f/'result.json');meshes.append(br['mesh_sha256']);camera_centers.append(np.asarray(br['camera']['transform_matrix'])[:3,3].tolist())
        request=read(folder/'request.json')
        for p,h in request['input_hashes'].items():assert sha(p)==h;hashes[p]=h
        for n,h in r['hashes'].items():hashes[str(folder/n)]=h
        hashes[str(folder/'request.json')]=sha(folder/'request.json');hashes[str(folder/'result.json')]=sha(folder/'result.json')
        if f not in REUSE:
            state=read(ROOT/'states'/(f+'.json'));assert state['terminal'] and state['exit_code']==0
            assert state['result_sha256']==sha(folder/'result.json')
            inp=ROOT/'inputs'/f;sq=read(inp/'request.json');sc=read(inp/'complete.json');assert sc['request_sha256']==sha(inp/'request.json')
            assert len(sq['records'])==len(sc['outputs'])==62
            assert len({x['camera'] for x in sq['records']})==62
            assert not {x['camera'] for x in sq['records']}&{'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
            for record in sq['records']:
                assert sha(record['input_path'])==record['input_sha256']
                assert sha(record['source_path'])==record['source_sha256'];hashes[record['source_path']]=record['source_sha256']
            for record in sc['outputs']:assert sha(record['path'])==record['sha256']
            source_hashes.append(next(x['source_sha256'] for x in sq['records'] if x['camera']=='H004_C005_1210SZ'))
    assert len(set(meshes))==37 and len({tuple(c) for c in camera_centers})==37 and len(set(source_hashes))==33
    viewed=[]
    for clip in CLIPS:
        folder=ROOT/'review'/clip;review=read(folder/'result.json');assert [x['frame'] for x in review['records']]==CLIPS[clip]
        for p,h in review['input_hashes'].items():assert sha(p)==h;hashes[p]=h
        for p,h in review['images'].items():assert sha(p)==h;viewed.append(p)
        assert sha(folder/'matched.mp4')==review['video_sha256']
        records=review['records'];steps=[x for x in records if 'motion_diagnostic' in x]
        weights=np.array([x['appearance_diagnostic']['common_samples'] for x in steps])
        summaries[clip]=dict(rgb_changes_min_median_max=np.quantile([x['rgb_changes'] for x in records],[0,.5,1]).tolist(),
            newly_zero_rgb=sum(x['new_black'] for x in records),
            common_source_samples=sum(x['motion_diagnostic']['common_tracked_samples'] for x in steps),
            source_switches={k:sum(x['motion_diagnostic'][k+'_source_switches'] for x in steps) for k in ['baseline','candidate']},
            mean_rgb_step={k:float(np.average([x['appearance_diagnostic'][k]['mean_rgb_step'] for x in steps],weights=weights)) for k in ['baseline','candidate']})
    exceptions=read(ROOT/'exceptions/result.json');assert len(exceptions['new_black'])==8
    for p,h in exceptions['images'].items():assert sha(p)==h;viewed.append(p)
    assert len(viewed)==len(set(viewed))==26
    for f in ['001071','001085']:
        folder=ROOT/'renders'/f/'consensus';audit=read(folder/'independent_audit.json')
        assert audit['result_sha256']==sha(folder/'result.json') and audit['request_sha256']==sha(folder/'request.json')
    import json
    checks=[json.loads(x) for x in (ROOT/'checks.jsonl').read_text().splitlines()]
    assert all(not x['errors'] for x in checks) and not checks[-1]['active']
    seconds=(datetime.fromisoformat(checks[-1]['utc'])-datetime.fromisoformat(checks[0]['utc'])).total_seconds()
    for p in ROOT.rglob('*'):
        if p.is_file():hashes[str(p)]=sha(p)
    for name in ['run_temporal_face_angular_control.py','review_temporal_face_angular_control.py',
        'inspect_temporal_face_angular_exceptions.py','seal_temporal_face_angular_control.py',
        'review_temporal_source_retention.py','audit_face_angular_visibility.py']:
        p=Path(__file__).with_name(name).resolve();hashes[str(p)]=sha(p)
    report=Path(__file__).resolve().parents[1]/'experiments/dec5_temporal_face_angular_control.md';hashes[str(report)]=sha(report)
    write(out,dict(hashes=hashes,summaries=summaries,viewed_images=viewed,
        unique_actor_times=37,unique_camera_centers=37,new_workers=33,reused=4,supervised_seconds=seconds,
        audit_scope='All frame receipts/depth references and unchanged-source RGB; independent full ray/RGB stress audits at001071 and001085.',
        continuous_playback_claimed=False,production_promoted=False,full_goal='not achieved',geometry_changed=False))
    for p,h in read(out)['hashes'].items():assert sha(p)==h,p
    print('sealed',len(hashes),'hash bindings',len(viewed),'viewed images',round(seconds,1),'supervised seconds',flush=True)


if __name__=='__main__':main()
