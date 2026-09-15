"""Freeze reviewed texture correction and the first same-recipe time transfer."""
from pathlib import Path
import argparse
from joint_temporal_texture import read,sha,atomic_json
from study_native_texture_footprint import OUT
from run_neighborhood_completion_transfer import ROOT


def run(check=False):
    path=OUT/'artifact_manifest.json';transfer=ROOT/'001195'
    if check:
        files=read(path)['files']
        for name,digest in files.items():
            if sha(name)!=digest:raise ValueError('Changed artifact: '+name)
        print('Verified',len(files),'hashes');return
    if path.exists():raise ValueError('Already frozen')
    review=read(OUT/'review/result.json');train=next(r for r in review['records'] if r['view']=='F004_E005_1210FP')
    assert train['black_with_geometry']==[16,0] and train['recovered_vs_GT_rgb_absolute_error_mean_8bit']==0
    assert all(r['depth_arrays_exact'] for r in review['records'])
    audit=read(transfer/'interpolated/001195/audit.json');assert len(audit['native_ray_checks'])==124
    panels=[OUT/'review'/(v+'_'+kind+'.png') for v in ['moving','F004_E005_1210FP'] for kind in ['head','detail']]
    panels += [OUT/'review/heldout_head.png',OUT/'heldout/comparison.png']
    panels += [transfer/'review'/(v+'_'+kind+'.png') for v in ['moving','F004_E005_1210FP'] for kind in ['head','detail']]
    atomic_json(OUT/'visual_review.json',dict(reviewer='main_agent',status='positive_local_transfer_not_artifact_free',
        inspected={str(p):sha(p) for p in panels},video_updated=False,
        notes='Texture-only black pixels recovered at 001193; geometry gain transfers to 001195; small geometric and broader video defects remain.'))
    names=['native_texture_footprint.py','study_native_texture_footprint.py','review_native_texture_footprint.py',
           'run_neighborhood_completion_transfer.py','review_neighborhood_completion_transfer.py','freeze_native_texture_footprint.py']
    external=[Path(__file__).with_name(n) for n in names]
    external += [Path(__file__).parents[1]/'experiments/dec5_native_texture_footprint.md',
                 Path(__file__).parents[1]/'tests/test_native_texture_footprint.py']
    for record in review['records']:
        external += [Path(p) for p in record['input_hashes']]
    for location in [OUT/'moving',OUT/'F004_E005_1210FP',OUT/'heldout']:
        req=read(location/'request.json')
        for name,digest in req['script_hashes'].items():
            p=Path(__file__).with_name(name);assert sha(p)==digest;external.append(p)
        for row in req['inventory']:
            p=Path(row['mesh']);assert sha(p)==row['mesh_sha256'];external.append(p)
    for name in ['request.json','result.json','mesh.ply','evidence.npz']:
        external.append(Path('/mnt/data/dec5_jaw_measured_mask_control/001195')/name)
    external += [Path('/mnt/data')/n for n in ['dec5_native_footprint_train.log','dec5_native_footprint_moving.log',
        'dec5_native_footprint_heldout.log','dec5_native_footprint_metrics.log','dec5_neighborhood_transfer_001195.log',
        'dec5_neighborhood_transfer_001195_rgb.log','dec5_neighborhood_transfer_001195_audit.log']]
    files={str(p):sha(p) for root in [OUT,transfer] for p in root.rglob('*') if p.is_file() and p!=path}
    files.update({str(p):sha(p) for p in external})
    atomic_json(path,dict(files=files,status='reviewed_local_method_not_video_publication'));print('Frozen',len(files),'hashes')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');run(p.parse_args().check)
