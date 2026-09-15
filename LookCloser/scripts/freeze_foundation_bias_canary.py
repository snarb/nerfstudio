"""Freeze/recheck bias evidence and rejected foreground geometry without rollout."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import subprocess
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from stereo_anchor_bias import fit_and_validate
from study_foundation_anchor_bias import ROOT as BIAS
from build_foundation_consensus_patch import ROOT as EMPTY
from build_foundation_foreground_patch import ROOT as PATCH


def freeze():
    request=read(BIAS/'request.json'); result=read(BIAS/'result.json')
    assert result['request_sha256']==sha(BIAS/'request.json')
    inputs={**request['scripts'],**request['source_depth_hashes'],**result['dependencies']}
    for path,digest in inputs.items(): assert sha(path)==digest
    sources=[Path('/mnt/data/dec5_foundation_hand_stereo/001037'),Path('/mnt/data/dec5_foundation_wrist_stereo/001037')]
    for source in sources:
        for pair in read(source/'request.json')['pairs']:
            name=Path(pair['directory']).name; saved=read(BIAS/name/'result.json'); data=np.load(BIAS/name/'anchors.npz')
            assert sha(BIAS/name/'anchors.npz')==saved['anchors_sha256']
            assert (data['other_votes'][data['selected']]>=3).all()
            cal=np.load(Path(pair['directory'])/'calibration.npz')
            computed=fit_and_validate(data['rectified_uv'][data['selected']],data['predicted'],data['expected'],
                float(cal['cropped_intrinsic'][0,0]*cal['baseline']),float(cal['disparity_offset']))
            for key,value in computed.items(): assert value==saved[key], (name,key)
    assert read(EMPTY/'result.json')['retained_triangles']==0
    assert sha(EMPTY/'mesh.ply')==read(EMPTY/'request.json')['source_mesh_sha256']
    q=read(PATCH/'request.json'); p=read(PATCH/'result.json')
    assert p['guard_passed'] and p['retained_triangles']==6 and len(p['rounds'][-1]['checks'])==124
    assert all(r['trusted_free']==0 for r in p['rounds'][-1]['checks'])
    original=o3d.io.read_triangle_mesh(q['source_mesh']); candidate=o3d.io.read_triangle_mesh(str(PATCH/'mesh.ply'))
    np.testing.assert_array_equal(np.asarray(original.vertices),np.asarray(candidate.vertices)[:len(original.vertices)])
    np.testing.assert_array_equal(np.asarray(original.triangles),np.asarray(candidate.triangles)[:len(original.triangles)])
    inspected=[BIAS/name/'anchor_review.png' for name in ['F004_A_G004_A','E004_C_F004_C','G004_A_H004_A']]
    inspected += [PATCH/'review'/(name+'.png') for name in ['H004_A005_1210M6','moving']]
    atomic_json(BIAS/'visual_review.json',dict(status='bias_improvement_but_no_useful_surface_completion',
        inspected_images={str(p):sha(p) for p in inspected},
        notes=['F/A-G/A has mostly negative disparity residual; E/C-F/C mostly positive, with local opposite-sign patches.',
               'Weak G/A-H/A evidence is sparse and clustered; it was not promoted from a pooled anchor count.',
               'Native H/A and moving geometry comparisons retain the large wrist/forearm tear; 6 surviving faces are not a meaningful repair.',
               'No new textured RGB, face-quality metrics or video acceptance; other saved geometry panels were not visually certified.'],
        production_updated=False))
    ps=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
    owned=[s for s in ps.splitlines() if any('python scripts/'+n in s for n in
        ['study_foundation_anchor_bias.py','build_foundation_consensus_patch.py','build_foundation_foreground_patch.py']) and '/bin/bash' not in s]
    assert not owned,'Owned geometry job is still live'
    atomic_json(BIAS/'terminal_check.json',dict(utc=datetime.now(timezone.utc).isoformat(),live_workers=owned,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
        free_bytes=shutil.disk_usage(BIAS).free,all_owned_jobs_terminal=True))
    files={str(p):sha(p) for root in [BIAS,EMPTY,PATCH] for p in root.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    files.update(inputs)
    for name in ['stereo_anchor_bias.py','study_foundation_anchor_bias.py','build_foundation_consensus_patch.py',
                 'build_foundation_foreground_patch.py','review_foundation_bias_canary.py',Path(__file__).name]:
        path=Path(__file__).resolve().with_name(name);files[str(path)]=sha(path)
    for name in ['dec5_foundation_anchor_bias.log','dec5_foundation_consensus_patch.log',
                 'dec5_foundation_foreground_patch.log','dec5_foundation_bias_review.log','dec5_foundation_bias_tests.log']:
        path=Path('/mnt/data')/name;files[str(path)]=sha(path)
    atomic_json(BIAS/'artifact_manifest.json',dict(hashes=files,production_updated=False,
        no_new_rgb_or_video=True,no_full_frame_metrics=True,geometry_completion_accepted=False))
    print('Frozen',len(files),'hashes',flush=True)


def verify():
    q=read(BIAS/'artifact_manifest.json')
    for path,digest in q['hashes'].items():
        if sha(path)!=digest:raise ValueError('Changed artifact: '+path)
    print('Rechecked',len(q['hashes']),'hashes',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:verify()
    elif (BIAS/'artifact_manifest.json').exists():raise ValueError('Already frozen; use --check')
    else:freeze()
