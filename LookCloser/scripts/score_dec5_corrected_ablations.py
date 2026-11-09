#!/usr/bin/env python3
"""Collect sealed, matched corrected-baseline ablations without refitting."""
from pathlib import Path
import argparse,json,subprocess,sys,time
from run_dec5_sequence import read,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    rows=[];parents=set()
    for arm in ['control','hair_mix','frequency_gate01']:
        root=a.root/('corrected_'+arm);training=root/'frames/000901'/('training_corrected_'+arm)
        while not (root/'complete.json').exists():time.sleep(15)
        request=read(training/'request.json');parents.add(request['experimental_parent_sha256'])
        if len(parents)!=1:raise ValueError('Ablations use different parent checkpoints')
        output=root/'roi_metrics.json'
        if not output.exists():
            command=[sys.executable,str(Path(__file__).with_name('score_dec5_sequence_pilot.py')),
                '--frame-root',str(root/'frames/000901'),'--training',str(training),
                '--reference',str(a.root/'opacity_diagnosis15188/reference.npz'),'--output',str(output)]
            with (root/'roi_score.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows.append(dict(arm=arm,**read(output)['rows'][0]))
        write(a.root/'corrected_ablations_summary.json',dict(rows=rows,finished=len(rows)==3,parent_sha256=next(iter(parents))))


if __name__=='__main__':main()
