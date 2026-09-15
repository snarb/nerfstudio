"""Bind the reviewed local prior and its remaining texture-coverage diagnosis."""
from pathlib import Path
import argparse
from joint_temporal_texture import read,sha,atomic_json
from study_poisson_jaw_completion import OUT,SOURCE,FRAME


def run(check=False):
    path=OUT/'artifact_manifest.json'
    if check:
        r=read(path)
        for name,digest in r['files'].items():
            if sha(name)!=digest:raise ValueError('Changed artifact: '+name)
        print('Verified',len(r['files']),'hashes');return
    if path.exists():raise ValueError('Already frozen; use --check')
    audit=read(OUT/'audit.json');assert audit['fresh_native_checks']==248
    ia=read(OUT/'interpolated'/FRAME/'audit.json');assert len(ia['native_ray_checks'])==124
    held=read(OUT/'interpolated/heldout/metrics.json');a,b=held['rows']
    for k in ['face_psnr','face_ssim','face_lpips','prediction_sha256']:assert a[k]==b[k]
    panels=[OUT/(v+'_raw_added.png') for v in ['moving','F004_E005_1210FP','D004_D005_1210LZ']]
    panels += [OUT/review/(v+'_'+detail+'.png') for review in ['review','review_interpolated']
               for v in ['moving','F004_E005_1210FP'] for detail in ['head','detail']]
    panels.append(OUT/'interpolated/heldout/comparison.png')
    atomic_json(OUT/'visual_review.json',dict(reviewer='main_agent',status='positive_local_mesh_step_not_artifact_free',
        inspected={str(p):sha(p) for p in panels},production_promoted=False,video_updated=False,
        notes='Interpolation reduces jaw spot; residual geometry misses and texture-source holes remain. Raw shell rejected; temporal transfer untested.'))
    scripts=['study_poisson_jaw_completion.py','guard_poisson_jaw_completion.py','review_poisson_jaw_completion.py',
             'audit_poisson_jaw_completion.py','local_surface_certificate.py','study_interpolated_poisson_jaw.py',
             'audit_interpolated_poisson_jaw.py','diagnose_poisson_texture_holes.py','freeze_poisson_jaw_completion.py']
    external=[Path(__file__).with_name(n) for n in scripts]
    external += [Path(__file__).parents[1]/'experiments/dec5_poisson_jaw_completion.md']
    external += [Path(__file__).parents[1]/'tests'/n for n in ['test_guard_poisson_jaw_completion.py','test_local_surface_certificate.py']]
    external += [SOURCE/n for n in ['request.json','result.json','mesh.ply','evidence.npz']]
    external.append(Path(read(SOURCE/'request.json')['source_mesh']))
    for review in ['review','review_interpolated']:
        for record in read(OUT/review/'result.json')['records']:
            for name,digest in record['inputs'].items():assert sha(name)==digest;external.append(Path(name))
    for name,digest in read(OUT/'admission/request.json')['scripts'].items():
        p=Path(__file__).with_name(name);assert sha(p)==digest;external.append(p)
    logs=['dec5_poisson_jaw_completion.log','dec5_poisson_jaw_guard.log','dec5_poisson_jaw_strict_rgb.log',
          'dec5_poisson_jaw_anchored_rgb.log','dec5_poisson_jaw_interpolated.log','dec5_poisson_jaw_interpolated_rgb.log',
          'dec5_poisson_jaw_audit.log','dec5_poisson_interpolation_audit.log','dec5_poisson_jaw_heldout.log',
          'dec5_poisson_jaw_heldout_metrics.log','dec5_poisson_jaw_texture_diagnosis.log']
    external += [Path('/mnt/data')/n for n in logs]
    files={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p!=path}
    files.update({str(p):sha(p) for p in external})
    atomic_json(path,dict(files=files,status='positive_local_pilot_not_video_publication',fresh_native_checks=372))
    print('Frozen',len(files),'hashes')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');run(p.parse_args().check)
