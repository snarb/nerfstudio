from pathlib import Path
import json,hashlib,time
root=Path('/cache/ubuntu/lookcloser/emergency_workstations_20261111')
verified=set()
while True:
    required=[]
    for name in ['ws07','ws08']:
        inventory=json.loads((root/name/'source_inventory.json').read_text())
        for record in inventory['records']:
            if '/frames/' not in record['path'] or '/model/' not in record['path']:
                continue
            key=(name,record['path'])
            required.append(key)
            if key in verified:
                continue
            path=root/name/'filesystem'/record['path'].lstrip('/')
            if not path.is_file() or path.stat().st_size!=record['size']:
                continue
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8388608),b''):
                    digest.update(chunk)
            if digest.hexdigest()==record['sha256']:
                verified.add(key)
    report=dict(time=time.time(),verified=len(verified),required=len(required),complete=set(required)<=verified,files=[dict(host=n,path=p) for n,p in sorted(verified)])
    target=root/'model_verification.json'
    temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2))
    temporary.replace(target)
    if report['complete']:
        break
    time.sleep(30)
