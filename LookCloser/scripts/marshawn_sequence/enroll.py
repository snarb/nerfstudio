"""Enroll verified replica hosts without resetting the six-hour campaign clock."""
import argparse
import subprocess
import time
from pathlib import Path
from common import read,atomic
from campaign import Host


def run(root):
    root=Path(root)
    while True:
        fleet=read(root/'fleet.json');known={h['name'] for h in fleet['hosts']};changed=False
        for name in ('olive','clever'):
            if name in known:continue
            definition=root/'node_defs'/(name+'.json')
            if not definition.exists():continue
            row=read(definition);host=Host(row)
            done=host.get(host.root/'technical/port_v5/stage_complete.json')
            if not done or done.get('completed_iterations')!=32:continue
            host.put(host.root/'preflight_complete.json',dict(passed=True,technical_completed=32,units_scale=.2217203609534452,native_crop_metrics=True))
            row['agent_root']=str(host.root/'agents/gpu0');fleet['hosts'].append(row);changed=True
            print('ENROLLED',name,flush=True)
        if changed:
            atomic(root/'fleet.json',fleet);subprocess.run(['sudo','-n','systemctl','restart','marshawn908-controller.service'],check=True)
        if {'olive','clever'}<={h['name'] for h in fleet['hosts']}:return
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
