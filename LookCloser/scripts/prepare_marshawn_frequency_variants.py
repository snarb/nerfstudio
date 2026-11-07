"""Prepare two source-bound full TRAIN frequency queues; never launch or promote."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main(a):
    out = a.campaign/a.name
    if out.exists():
        raise ValueError('Preserve previous generation')
    ready = json.loads((a.data/'ready.json').read_text())
    transforms = json.loads((a.data/'transforms.json').read_text())
    records = [r for r in ready['records'] if r['camera'] not in ('DC','HC','KA')]
    if len(records)!=63 or len({r['camera'] for r in records})!=63:
        raise ValueError('Exactly63 unique TRAIN cameras required')
    if {r['image'] for r in records} != set(transforms['train_filenames']):
        raise ValueError('TRAIN split mismatch')
    out.mkdir()
    remote = Path(a.remote_root)/a.name
    remote_data = Path(a.remote_root)/a.data.relative_to(a.campaign)
    scripts = Path(__file__).resolve().parent
    for source, name in [(scripts/'pilot_marshawn_frequency_variants.py','worker.py'),
                         (scripts/'marshawn_gpu_isolation.py','gpu_isolation.py'),
                         (scripts.parent.parent/'nerfstudio/scripts/lookcloser_preprocess.py','preprocess.py')]:
        shutil.copyfile(source,out/name)
    shape, budget = ([1012,1920],1000) if a.resolution=='hd' else ([2024,3840],4000)
    base = json.loads((a.campaign/'resolution_4k_frequency_queue_4000_fleet/production_plan.json').read_text())
    base['jobs'] = []
    (out/'specs').mkdir()
    for row in records:
        camera = row['camera']
        image = a.data/row['image']
        if sha(image)!=row['rgb_sha256']:
            raise ValueError('Source image changed')
        for variant in ('lr_decay','threshold093'):
            name = a.name+'_'+camera+'_'+variant
            spec_name = camera+'_'+variant+'.json'
            inputs = {str(remote/f):sha(out/f) for f in ('worker.py','preprocess.py','gpu_isolation.py')}
            inputs.update({str(remote_data/f):sha(a.data/f) for f in ('ready.json','transforms.json',row['image'])})
            spec = dict(pilot_only=True, controller_owned_lease=True, data=str(remote_data),
                shape=shape, steps_per_level=budget, variants=[variant],
                output=str(Path(a.remote_root)/'jobs'/name/'run'),
                preprocess=str(remote/'preprocess.py'), gpu_isolation=str(remote/'gpu_isolation.py'),
                inputs_sha256=inputs, worker_sha256=sha(out/'worker.py'),
                images={camera:str(remote_data/row['image'])})
            write(out/'specs'/spec_name,spec)
            base['jobs'].append(dict(name=name, host='silver', dependencies=[],
                command=['{host_root}/python','{host_root}/'+a.name+'/worker.py',
                    '--plan','{host_root}/'+a.name+'/specs/'+spec_name,'--pinned-worker',
                    '--lease-receipt','{job_root}/receipt.json'],
                cwd='{host_root}/semantic_actor_candidate_v2_20261106/code/LookCloser',
                env=dict(PYTHONPATH='{host_root}/semantic_actor_candidate_v2_20261106/code',
                    OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1'),
                progress_files=['{job_root}/run/progress.json'], metrics_file='{job_root}/run/complete.json'))
    write(out/'controller_plan.json',base)
    write(out/'preparation.json',dict(prepared_only=True, launched=False, training_authorized=False,
        source_data=str(a.data),resolution=a.resolution,jobs=len(base['jobs']),
        controller_plan_sha256=sha(out/'controller_plan.json'),
        inventory={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        limitation='Two experimental map sets only. Selection of structural recipe, completed source audit, actual CUDA proof and separate dataset assembly required before NeRF use.'))
    print(json.dumps(dict(prepared=str(out),jobs=len(base['jobs']),launched=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('campaign','data'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--resolution',choices=('hd','4k'),required=True)
    p.add_argument('--remote-root',default='/home/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106')
    main(p.parse_args())
