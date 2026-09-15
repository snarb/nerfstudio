"""Freeze reviewed local geometry controls without promoting a new video."""
from pathlib import Path
import argparse
from joint_temporal_texture import read,sha,atomic_json
from study_subdivided_jaw_caps import OUT,SOURCE


def run(check=False):
    path=OUT/'artifact_manifest.json'
    if check:
        record=read(path)
        for name,digest in record['files'].items():
            if sha(name)!=digest:raise ValueError('Changed retained/input artifact: '+name)
        print('Verified',len(record['files']),'hashes');return
    if path.exists():raise ValueError('Already frozen; use --check')
    summary=read(OUT/'audit_summary.json')
    if summary['total_fresh_ray_checks']!=372:raise ValueError('Incomplete audit')
    for name,digest in summary['audits'].items():assert sha(name)==digest
    panels=[OUT/review/(view+'_'+kind+'.png') for review in ['review','review_large']
            for view in ['moving','F004_E005_1210FP'] for kind in ['head','detail']]
    atomic_json(OUT/'visual_review.json',dict(reviewer='main_agent',status='partial_improvement_not_artifact_free',
        inspected={str(p):sha(p) for p in panels},
        verdicts=dict(planar='no_improvement',curved='local_gain_defect_remains',large_curved='worse_than_small_curved'),
        geometry_promoted=False,new_video=False,notes='Residual isolated spot and ragged under-chin edge remain; no general head or hair repair.'))
    scripts=['subdivide_boundary_caps.py','study_subdivided_jaw_caps.py','review_subdivided_jaw_caps.py',
             'audit_subdivided_jaw_caps.py','freeze_subdivided_jaw_caps.py']
    external=[Path(__file__).with_name(n) for n in scripts]
    external.extend([Path(__file__).parents[1]/'experiments/dec5_curved_jaw_caps.md',
                     Path(__file__).parents[1]/'tests/test_subdivide_boundary_caps.py'])
    roots=[OUT,Path('/mnt/data/dec5_large_curved_jaw_caps')]
    for root in [OUT/'planar',OUT/'curved',roots[1]/'curved']:
        req=read(root/'001193/request.json');external.append(Path(req['source_mesh']))
        for name in req['scripts']:external.append(Path(__file__).with_name(name))
    for name in ['request.json','result.json','evidence.npz','mesh.ply']:external.append(SOURCE/'001193'/name)
    for review in ['review','review_large']:
        for row in read(OUT/review/'result.json')['records']:
            for name,digest in row['inputs'].items():assert sha(name)==digest;external.append(Path(name))
    files={str(p):sha(p) for root in roots for p in root.rglob('*') if p.is_file() and p!=path}
    files.update({str(p):sha(p) for p in external})
    atomic_json(path,dict(files=files,status='local_candidates_not_promoted',geometry_replay_and_native_guards=True))
    print('Frozen',len(files),'retained/input hashes')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');run(p.parse_args().check)
