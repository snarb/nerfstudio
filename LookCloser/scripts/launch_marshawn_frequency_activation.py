"""Launch a separately released HD frequency-fit queue from the local SSH origin."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(release_path):
    root = release_path.resolve().parent
    release = json.loads(release_path.read_text())
    activation = json.loads((root/'activation.json').read_text())
    if release.get('reviewer') != 'root' or release.get('approved_map_fits') is not True:
        raise ValueError('Explicit map-fit release required')
    if release.get('activation_sha256') != sha(root/'activation.json'):
        raise ValueError('Activation generation changed')
    if activation['jobs'] != 126 or activation['NeRF_training_authorized'] or activation['assembly_authorized']:
        raise ValueError('Only the reviewed126 map fits')
    bindings = json.loads((root/'bindings.json').read_text())
    if sha(root/'bindings.json') != activation['bindings_sha256']:
        raise ValueError('Bindings changed')
    plan = json.loads((root/'controller_plan.json').read_text())
    if sha(root/'controller_plan.json') != activation['controller_plan_sha256']:
        raise ValueError('Controller plan changed')
    verifier = """import json,sys,hashlib,pathlib,subprocess,shutil
request=json.load(sys.stdin)
for name,digest in request['files'].items():
 assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==digest,name
actual={int(row.split(',')[0]):row.split(',')[1].strip()for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
for index,uuid in request['GPU_UUIDs'].items():assert actual[int(index)]==uuid
assert shutil.disk_usage(request['root']).free>=10*2**30
print(json.dumps({'verified_files':len(request['files']),'GPU_UUIDs':request['GPU_UUIDs']}))
"""
    proofs = {}
    for host, files in bindings['hosts'].items():
        cfg = plan['hosts'][host]
        command = [cfg['python'], '-c', verifier]
        if cfg.get('address'):
            import shlex
            command = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',cfg['address'],shlex.join(command)]
        payload = dict(files=files,GPU_UUIDs=activation['GPU_UUIDs'][host],root=str(Path(cfg['root']).parent))
        result = subprocess.run(command,input=json.dumps(payload),text=True,capture_output=True,check=True)
        proofs[host] = json.loads(result.stdout)
    controller = root/'transport/controller.py'
    if sha(controller) != activation['controller_sha256']:
        raise ValueError('Dispatcher changed')
    record = root/'launch_verification.json'
    if not record.exists():record.write_text(json.dumps(dict(release_sha256=sha(release_path),proofs=proofs),indent=2)+'\n')
    subprocess.run([sys.executable,str(controller),'launch','--root',str(root/'controller'),
                    '--manifest',str(root/'controller_plan.json')],check=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    main(parser.parse_args().release)
