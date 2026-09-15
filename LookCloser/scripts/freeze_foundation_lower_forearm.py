"""Independent input/geometry/RGB audit and explicit negative visual gate."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import subprocess
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from review_jaw_repair_transfer import verified_image
from study_foundation_lower_forearm import ROOT,FRAME
from build_foundation_lower_forearm import MOVIE


def freeze():
    stage=read(ROOT/FRAME/'request.json');inputs={**stage['source_hashes'],**stage['rgb_receipt']['source_rgb_hashes']}
    assert sha(Path(__file__).with_name('study_foundation_lower_forearm.py'))==stage['script_sha256']
    for path,digest in inputs.items():assert sha(path)==digest
    inference=ROOT/FRAME/'inference';iq=read(inference/'request.json');done=read(inference/'complete.json')
    assert iq['staged_request_sha256']==sha(ROOT/FRAME/'request.json')
    assert done['request_sha256']==sha(inference/'request.json')
    engine=Path(__file__).with_name('infer_foundation_hand_stereo.py')
    assert sha(engine)==iq['script_sha256'];inputs[str(engine.resolve())]=sha(engine)
    for path,digest in read(ROOT/'geometry_adapter.json')['producers'].items():
        assert sha(path)==digest;inputs[path]=digest
    model=Path('/mnt/data/dec5_foundation_model_mirror/23-51-11/model_best_bp2.pth')
    assert sha(model)==iq['model_sha256'];inputs[str(model)]=iq['model_sha256']
    for pair in stage['pairs']:
        folder=Path(pair['directory'])
        for name,digest in pair['hashes'].items():assert sha(folder/name)==digest
        result=read(inference/folder.name/'complete.json')
        assert sha(inference/folder.name/'prediction.npz')==result['prediction_sha256']
    bias=read(ROOT/'bias/request.json')
    for path,digest in {**bias['scripts'],**bias['source_depth_hashes']}.items():
        assert sha(path)==digest;inputs[path]=digest
    q=read(ROOT/'foreground/request.json');r=read(ROOT/'foreground/result.json')
    assert r['request_sha256']==sha(ROOT/'foreground/request.json') and r['mesh_sha256']==sha(ROOT/'foreground/mesh.ply')
    assert r['guard_passed'] and len(r['rounds'][-1]['checks'])==124
    assert all(c['trusted_free']==0 for c in r['rounds'][-1]['checks'])
    source=o3d.io.read_triangle_mesh(q['source_mesh']);candidate=o3d.io.read_triangle_mesh(str(ROOT/'foreground/mesh.ply'))
    assert sha(q['source_mesh'])==q['source_mesh_sha256'];inputs[q['source_mesh']]=q['source_mesh_sha256']
    np.testing.assert_array_equal(np.asarray(candidate.vertices)[:len(source.vertices)],np.asarray(source.vertices))
    np.testing.assert_array_equal(np.asarray(candidate.triangles)[:len(source.triangles)],np.asarray(source.triangles))
    comparisons=[]
    for view in ['H004_A005_1210M6','E004_D005_1210L4','moving']:
        baseline=MOVIE if view=='moving' else ROOT/'rgb'/view/'baseline'
        a,ar=verified_image(baseline,FRAME);b,br=verified_image(ROOT/'rgb'/view/'candidate',FRAME)
        for key in ['camera','source_cameras','fixed_exposure']:assert ar[key]==br[key]
        assert b.shape==(1920,1080,3) and np.isfinite(b).all()
        comparisons.append(dict(view=view,upper_1200_rows_changed_pixels=int(np.any(a[:1200]!=b[:1200],axis=2).sum()),
            changed_rgb_pixels=int(np.any(a!=b,axis=2).sum())))
    inspected=[]
    for pair in stage['pairs']:
        folder=Path(pair['directory']);inspected += [folder/'rectification_review.png',inference/folder.name/'disparity_review.png']
    for name in ['H004_A005_1210M6','E004_D005_1210L4','moving']:
        inspected += [ROOT/'geometry_review'/(name+'.png'),ROOT/'rgb_review'/(name+'.png')]
    for name in ['B004_E005_1210VE','E004_E005_1210WX','F004_E005_1210FP']:
        inspected += [ROOT/'veto_diagnosis'/(name+'.png'),ROOT/'layer_diagnosis'/(name+'.png')]
    atomic_json(ROOT/'visual_review.json',dict(status='partial_forearm_recovery_not_accepted_for_video',
        inspected_images={str(p):sha(p) for p in inspected},
        rgb_verdicts=[dict(view=n,status='fail',notes=note) for n,note in [
            ('H004_A005_1210M6','Some lower forearm recovered, but black peppering, wrist cutout and hard texture boundary remain.'),
            ('E004_D005_1210L4','More arm surface, but scattered dark/blue defects and surface conflict increase in places.'),
            ('moving','Lower forearm gains surface; severe wrist gaps and dark scattered defects remain.')]],
        diagnosis='Four inspected B/E forearm samples project farther PM points onto blue clothing in four lower train views. Other skin-to-skin disagreements remain ambiguous.',
        no_blanket_depth_guard_override=True,full_video_approved=False,production_updated=False,
        montage_erratum='First layer montage could draw out-of-tile markers. Archived under layer_diagnosis_crop_unsafe; corrected variable-size crops were reinspected.'))
    atomic_json(ROOT/'audit.json',dict(comparisons=comparisons,source_vertices=len(source.vertices),source_triangles=len(source.triangles),
        candidate_vertices=len(candidate.vertices),candidate_triangles=len(candidate.triangles),
        source_arrays_preserved=True,final_native_guard_checks=124,full_frame_metrics=False,
        inspected_panel_count=len(inspected),new_rgb_renders=5,reused_moving_control=True))
    ps=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
    owned=[s for s in ps.splitlines() if any('python scripts/'+name in s for name in
        ['study_foundation_lower_forearm.py','build_foundation_lower_forearm.py','diagnose_lower_forearm_veto.py','diagnose_lower_forearm_layers.py']) and '/bin/bash' not in s]
    assert not owned
    atomic_json(ROOT/'terminal_check.json',dict(utc=datetime.now(timezone.utc).isoformat(),live_workers=owned,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
        free_bytes=shutil.disk_usage(ROOT).free,all_owned_jobs_terminal=True))
    hashes={str(p):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'};hashes.update(inputs)
    for name in ['study_foundation_lower_forearm.py','build_foundation_lower_forearm.py','diagnose_lower_forearm_veto.py',
                 'diagnose_lower_forearm_layers.py',Path(__file__).name]:
        path=Path(__file__).resolve().with_name(name);hashes[str(path)]=sha(path)
    for path in Path('/mnt/data').glob('dec5_foundation_lower_forearm_*.log'):
        if path.name.endswith('_freeze.log'):continue
        hashes[str(path)]=sha(path)
    atomic_json(ROOT/'artifact_manifest.json',dict(hashes=hashes,production_updated=False,full_video_approved=False,full_frame_metrics=False))
    print('Frozen',len(hashes),'hashes',comparisons,flush=True)


def verify():
    q=read(ROOT/'artifact_manifest.json')
    for path,digest in q['hashes'].items():
        if sha(path)!=digest:raise ValueError('Changed artifact: '+path)
    print('Rechecked',len(q['hashes']),'hashes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:verify()
    elif (ROOT/'artifact_manifest.json').exists():raise ValueError('Already frozen; use --check')
    else:freeze()
