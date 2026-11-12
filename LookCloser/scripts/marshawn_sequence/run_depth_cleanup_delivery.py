"""Resume the inference-only GC cleanup delivery without touching training units."""
import argparse
import json
import subprocess
import time
from pathlib import Path


def main(root, output):
    root, output = Path(root), Path(output)
    tools = root/'tools'
    python = json.loads((root/'settings.json').read_text())['python']
    def run(command):
        print(time.time(), 'START', command, flush=True)
        subprocess.run(command, check=True)
        print(time.time(), 'DONE', flush=True)
    batch = [python, str(tools/'depth_cleanup_batch.py'), '--root', str(root), '--output', str(output)]
    run(batch+['--stage','raw'])
    run([python,str(tools/'prepare_depth_temporal.py'),'--output',str(output),
         '--name','temporal_support_v3','--minimum-component','10000','--preserve-lower-border','--minimum-height','120'])
    for name, radius, feather in [('depth_soft_v3',8,6),('depth_strict_v3',3,5)]:
        run(batch+['--stage','clean','--name',name,'--temporal','--support-name','temporal_support_v3','--band','.15',
                   '--fade','.10','--radius',str(radius),'--feather',str(feather)])
    run([python,str(tools/'assemble_depth_cleanup.py'),'--output',str(output),'--suffix','_v3'])


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();main(args.root,args.output)
