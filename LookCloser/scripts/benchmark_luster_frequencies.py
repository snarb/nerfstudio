"""Measure one CUDA worker against completed four-worker fits, with identical RGB."""
import argparse
import json
from pathlib import Path
import time
from prepare_luster_frequencies import fit_one
from prepare_luster_video import write


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    source=args.root/'frames/000473/data';out=args.root/'preflight/frequency_serial_benchmark'
    (out/'images').mkdir(parents=True,exist_ok=True);(out/'lookcloser_frequencies').mkdir(exist_ok=True)
    meta=json.loads((source/'transforms.json').read_text());results=[]
    for camera in [11,97,151,36]:
        row=next(r for r in meta['frames'] if r['camera_id']==camera)
        target=out/row['file_path']
        if not target.exists():target.hardlink_to(source/row['file_path'])
        start=time.monotonic();stem=fit_one(str(out),row)
        current=json.loads((out/'lookcloser_frequencies'/f'{stem}.receipt.json').read_text())
        prior=json.loads((source/'lookcloser_frequencies'/f'{stem}.receipt.json').read_text())
        results.append(dict(camera=camera,wall_seconds=time.monotonic()-start,serial_seconds=current['seconds'],
                            parallel_worker_seconds=prior['seconds'],identical_map=current['sha256']==prior['sha256']))
        write(out/'results.json',results);print(json.dumps(results[-1]),flush=True)


if __name__=='__main__':main()
