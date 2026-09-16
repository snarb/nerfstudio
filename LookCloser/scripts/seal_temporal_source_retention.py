"""Seal the two reviewed temporal controls without changing production."""
from pathlib import Path
import argparse
import numpy as np
from study_temporal_source_retention import ROOT,BASE,CLIPS,FRAMES
from joint_temporal_texture import read,sha,atomic_json
from review_jaw_repair_transfer import verified_image
from review_temporal_source_retention import exact_geometry


def main(notes_path):
    assert not (ROOT/'artifact_manifest.json').exists()
    notes=read(notes_path);supervisor=read(ROOT/'supervisor_result.json')
    assert notes['actually_reviewed'] is True and notes['production_promoted'] is False
    assert supervisor['all_passed'] and len(supervisor['finished'])==48
    assert {r['frame'] for r in supervisor['finished']}==set(FRAMES)
    assert all(r['exit_code']==0 for r in supervisor['finished'])
    parent=read(BASE/'request.json');bindings={};required_images={};summaries={}
    for clip,frames in CLIPS.items():
        review=read(ROOT/'review'/clip/'result.json');required_images.update(review['images'])
        sums={k:0 for k in ['common_tracked_samples','baseline_source_switches','candidate_source_switches']}
        centers=[];mesh_hashes=[]
        for row in review['records']:
            f=row['frame'];assert f in frames
            a,ar=verified_image(BASE,f);b,br=verified_image(ROOT/f,f)
            exact_geometry(BASE/'frames'/f,ROOT/f/'frames'/f)
            q=read(ROOT/f/'request.json');source=next(r for r in parent['inventory'] if r['frame_id']==f)
            assert q['inventory']==[source]
            centers.append(np.array(source['camera']['transform_matrix'])[:3,3]);mesh_hashes.append(source['mesh_sha256'])
            assert row['new_black']==int(((a.max(2)>0)&(b.max(2)==0)).sum())
            for k,v in row.get('temporal_diagnostic',{}).items():sums[k]+=v
            for file,h in read(ROOT/f/'frames'/f/'complete.json')['hashes'].items():bindings[str(ROOT/f/'frames'/f/file)]=h
            for n,h in q['script_hashes'].items():
                path=Path(__file__).with_name(n);assert sha(path)==h,path;bindings[str(path)]=h
        assert len(review['records'])==24 and len(set(mesh_hashes))==24
        assert len(np.unique(np.array(centers),axis=0))==24
        summaries[clip]=dict(source_tracking=sums,
            changed_rgb_min_median_max=np.quantile([r['changed_rgb'] for r in review['records']],[0,.5,1]).tolist(),
            newly_black=sum(r['new_black'] for r in review['records']),
            camera_travel_normalized=float(np.linalg.norm(np.diff(centers,axis=0),axis=1).sum()),
            unique_actor_meshes=24,unique_camera_centers=24)
        for p,h in review['input_hashes'].items():assert sha(p)==h,p;bindings[p]=h
        video=ROOT/'review'/clip/'matched_diagnostic.mp4';assert sha(video)==review['video_sha256']
    black=read(ROOT/'review/new_black/result.json');required_images.update(black['images'])
    nose=read(ROOT/'nose_attribution/result.json')
    for n,h in nose['outputs'].items():assert sha(ROOT/'nose_attribution'/n)==h,n
    required_images[str(ROOT/'nose_attribution/nose.png')]=nose['outputs']['nose.png']
    assert nose['changed_selected_rgb']==0
    assert all(r['no_source']==r['no_depth']==0 for r in nose['records'])
    assert sum(s['newly_black'] for s in summaries.values())==len(black['records'])
    assert set(notes['reviewed_images'])==set(required_images),'Every generated panel must be inspected'
    for p,h in required_images.items():assert sha(p)==h,p
    for p in ROOT.rglob('*'):
        if p.is_file():bindings[str(p)]=sha(p)
    for p in [Path(__file__),Path(__file__).parents[1]/'experiments/dec5_temporal_source_retention.md',
              Path(__file__).parents[1]/'tests/test_temporal_source_retention.py',
              Path(__file__).with_name('review_temporal_source_retention.py'),
              Path(__file__).with_name('inspect_temporal_retention_black_pixels.py'),
              Path(__file__).with_name('diagnose_temporal_retention_nose.py')]:
        bindings[str(p)]=sha(p)
    atomic_json(ROOT/'artifact_manifest.json',dict(status='reviewed_control_not_promoted',
        summaries=summaries,visual_review=notes,hashes=bindings,
        full_goal_complete=False,delivered_6k_video_changed=False))
    print(summaries,flush=True);print('bindings',len(bindings),'reviewed panels',len(required_images),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--notes',type=Path,required=True)
    main(p.parse_args().notes)
