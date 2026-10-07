#!/usr/bin/env python3
"""Wait for audited preprocessing, then launch exactly one supervised visual gate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True)
    args=p.parse_args()
    request=json.loads(args.request.read_text())
    data=Path(request['data']);root=data.parent
    while not (data/'frequency_ready.json').exists():
        if time.time()>=request['campaign_deadline']:
            raise SystemExit('Campaign budget exhausted before training')
        print(f"waiting_frequency={len(list((data/'lookcloser_frequencies').glob('*.pt')))}/162",flush=True)
        time.sleep(30)
    with (root/'logs/data_ready_audit.log').open('w') as log:
        subprocess.run([sys.executable,str(Path(__file__).with_name('audit_luster_opacity_data.py')),
                        '--data',str(data)],stdout=log,stderr=subprocess.STDOUT,check=True)
    raise SystemExit(subprocess.call([sys.executable,str(Path(__file__).with_name('run_luster_opacity.py')),
                                     'supervise','--request',str(args.request)]))
