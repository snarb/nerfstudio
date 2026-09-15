"""Matched TSDF activation transfer on dev3; keep local video GPU independent.

One staged time per launch. Historical camera/JPEG recipe, one shared fresh
62-view PatchMatch run, two fusion domains. No production/default mutation.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

LOCAL = Path('/mnt/data/dec5_full_block_transfer')
REMOTE = Path('/fsx/oregon/dec5_full_block_transfer')
CAMPAIGN = Path('/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150')
FRAMES = ('000995', '000997')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def immutable(path, value):
    # Normalize tuples from captured stage commands to their JSON representation.
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists(): assert read(path) == value, 'Immutable request mismatch'
    else: write(path, value)


def prepare(frame):
    root = LOCAL/frame; root.mkdir(parents=True, exist_ok=True)
    code = root/'code'; code.mkdir(exist_ok=True)
    sources = {p.name: p for p in (CAMPAIGN/'config/reconstruct_code').glob('*.py')}
    # Both fusion arms use the same current implementation; only the flag differs.
    sources['fuse_depth_tsdf_mesh.py'] = Path(__file__).with_name('fuse_depth_tsdf_mesh.py')
    for name, path in sources.items():
        dest = code/name
        if dest.exists(): assert sha(dest) == sha(path)
        else: shutil.copyfile(path, dest)
    source = Path('/mnt/data/dec5_5a3_nerfstudio_exr_1920x1080')/frame
    calibration = CAMPAIGN/'config/calibration/transforms.json'
    assert sha(calibration) == '79a91edfd8b441df1ff229839e2cc5f0b861ebe3fd626f40b04280d76a5f3900'
    (root/'logs').mkdir(exist_ok=True)
    with (root/'logs/convert.log').open('a') as stream:
        subprocess.run([sys.executable, str(code/'convert_exr_nerfstudio_to_jpeg.py'),
            '--input', str(source), '--output', str(root/'jpeg65'), '--middle-gray', '.18',
            '--exposure-mode', 'per-image', '--exposure-percentile', '70', '--quality', '98', '--resume'],
            check=True, stdout=stream, stderr=subprocess.STDOUT,
            env=dict(os.environ, OPENCV_IO_ENABLE_OPENEXR='1'))
    sys.path.insert(0, str(code))
    from colmap_patchmatch_tsdf_campaign_common import stage_fixed_calibration_dataset
    if not (root/'data/staging_manifest.json').exists():
        stage_fixed_calibration_dataset(source, root/'jpeg65', calibration, root/'data')
    staged = read(root/'data/staging_manifest.json')
    historical_path = Path('/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/frames')/frame/'staging_manifest.json'
    old = {r['physical_camera']: r for r in read(historical_path)['conversion_rows']}
    fresh = {r['physical_camera']: r for r in staged['conversion_rows']}
    assert set(old) == set(fresh) and len(old) == 63
    for name, row in fresh.items():
        assert row['sha256'] == old[name]['sha256'] == sha(row['output'])
        assert row['exposure_gain'] == old[name]['exposure_gain']
    request = dict(frame=frame, source_transforms_sha256=sha(source/'transforms.json'),
        source_images={r['file_path']: sha(source/r['file_path']) for r in read(source/'transforms.json')['frames']},
        historical_staging_sha256=sha(historical_path), exact_historical_jpeg_count=63,
        code={n: sha(code/n) for n in sources}, controller_sha256=sha(__file__),
        data_hashes={str(p.relative_to(root/'data')): sha(p) for p in (root/'data').rglob('*') if p.is_file()},
        calibration_sha256=sha(calibration), source_count=12, remote_root=str(REMOTE/frame),
        controlled_variable='TSDF per-view vs full bounded block-union integration',
        masks_used=False, eval_rgb_used_for_geometry=False, production_changed=False,
        texture_ingest='Not rendered here; video fixed profiles/global exposure unchanged')
    immutable(root/'request.json', request)
    subprocess.run(['ssh','ubuntu@dev3','mkdir','-p',str(REMOTE/frame)], check=True)
    for name in ('code', 'data'):
        subprocess.run(['rsync','-rL',str(root/name)+'/',f'ubuntu@dev3:{REMOTE/frame/name}/'], check=True)
    subprocess.run(['rsync','-r',str(root/'request.json'),f'ubuntu@dev3:{REMOTE/frame}/'], check=True)
    subprocess.run(['rsync','-r',__file__,f'ubuntu@dev3:{REMOTE}/controller.py'], check=True)
    print(frame, 'staged exact historical63, transferred to dev3', flush=True)


def stage(command, root, name):
    path = root/'stages'/(name+'.json')
    if path.exists():
        r = read(path); assert r['command'] == command and r['request_sha256'] == sha(root/'request.json')
        for p, digest in r['hashes'].items(): assert sha(root/p) == digest
        return
    log = root/'logs'/(name+'.log'); log.parent.mkdir(exist_ok=True)
    start = time.time()
    with log.open('a') as stream:
        worker = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            record = dict(utc=datetime.now(timezone.utc).isoformat(), stage=name,
                controller_pid=os.getpid(), worker_pid=worker.pid, worker_alive=worker.poll() is None,
                elapsed_seconds=time.time()-start, free_bytes=shutil.disk_usage(root).free,
                geometric_maps=len(list((root/'pipeline/dense/stereo/depth_maps').glob('**/*.geometric.bin'))),
                photometric_maps=len(list((root/'pipeline/dense/stereo/depth_maps').glob('**/*.photometric.bin'))),
                gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True),
                log_tail=log.read_text(errors='replace')[-1200:])
            write(root/'progress.json', record)
            with (root/'checks.jsonl').open('a') as checks: checks.write(json.dumps(record)+'\n')
            if worker.poll() is not None: break
            try: worker.wait(timeout=30)
            except subprocess.TimeoutExpired: pass
        assert worker.returncode == 0, 'Failed stage; preserve workspace: '+str(log)
    if name.startswith('patchmatch-'):
        files=list((root/'pipeline/dense/stereo/depth_maps').glob('**/*.'+name.split('-')[-1]+'.bin'))
        assert len(files)==62
    elif name.startswith('fuse-'):
        mesh=Path(command[command.index('--output')+1]); files=[mesh,mesh.with_suffix('.json')]
    else:
        files=[p for p in (root/'pipeline').rglob('*') if p.is_file() and p.suffix in ('.json','.txt','.cfg','.gz')]
    write(path,dict(command=command,request_sha256=sha(root/'request.json'),seconds=time.time()-start,
        hashes={str(p.relative_to(root)):sha(p) for p in files}))
    print(root.name,name,'complete',flush=True)


def remote(frame):
    root=REMOTE/frame
    with (REMOTE/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU occupied'
        q=read(root/'request.json'); assert sha(__file__)==q['controller_sha256']
        for name,digest in q['code'].items(): assert sha(root/'code'/name)==digest
        for name,digest in q['data_hashes'].items(): assert sha(root/'data'/name)==digest
        os.environ.update(PYTHONPATH=f'/home/ubuntu/repos/nerfstudio:{root}/code',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',OPENCV_IO_ENABLE_OPENEXR='1')
        sys.path.insert(0,str(root/'code'))
        import run_colmap_patchmatch_tsdf as runner
        build=subprocess.run(['/usr/local/bin/colmap','-h'],capture_output=True,text=True,check=True)
        runner.validate_colmap_build(build.stdout+build.stderr,allow_unverified=False)
        immutable(root/'binary.json',dict(build=build.stdout+build.stderr,sha256=sha('/usr/local/bin/colmap')))
        commands=[]
        runner.run_stage=lambda name, command, **kwargs: commands.append((name,command)) or {'name':name}
        with redirect_stdout(io.StringIO()):
            runner.main(['--data',str(root/'data'),'--output-dir',str(root/'pipeline'),
                '--source-count','12','--colmap-bin','/usr/local/bin/colmap','--dry-run'])
        immutable(root/'commands.json',commands)
        for name, command in commands:
            if name in {'texture-subset','raycast-mesh','hard-texture-render','fuse-tsdf'}: continue
            stage(command,root,name)
        import numpy as np
        from import_colmap_mvs_depth_dataset import read_colmap_dense_array
        depths=sorted((root/'pipeline/dense/stereo/depth_maps').glob('**/*.geometric.bin'))
        assert len(depths)==62; coverage=[]
        for p in depths:
            d=read_colmap_dense_array(p); assert d.shape==(1080,1920,1) and np.isfinite(d).all()
            coverage.append(float((d>0).mean())); assert coverage[-1]>0
        write(root/'depth_qc.json',dict(maps=62,shape=[1080,1920],coverage_mean=float(np.mean(coverage)),coverage_min=min(coverage),hashes={str(p.relative_to(root)):sha(p) for p in depths}))
        fuse=next(c for n,c in commands if n=='fuse-tsdf')
        for name, extra in [('fuse-original',[]),('fuse-full-block',['--tensor-full-block-integration'])]:
            (root/name).mkdir(exist_ok=True); command=list(fuse)
            command[command.index('--output')+1]=str(root/name/'mesh.ply');stage(command+extra,root,name)
        retained=[root/'depth_qc.json',root/'binary.json',root/'commands.json']
        retained += [p for name in ('fuse-original','fuse-full-block') for p in (root/name).iterdir() if p.is_file()]
        write(root/'complete.json',dict(request_sha256=sha(root/'request.json'),hashes={str(p.relative_to(root)):sha(p) for p in retained},visual_status='pending',production_changed=False))
        print(frame,'paired geometry complete; review pending',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','remote'))
    parser.add_argument('--frame',required=True,choices=FRAMES);args=parser.parse_args()
    (prepare if args.action=='prepare' else remote)(args.frame)
