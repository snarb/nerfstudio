#!/usr/bin/env python3
"""Compare frozen depth statistics without further field queries or training."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
from run_dec5_sequence import read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    r = a.root
    scripts = Path(__file__).parent
    base = r / 'geometry/control'
    rows = [dict(statistic='q1000', metrics=read(base/'metrics.json')['aggregate'])]
    for statistic in ['expected', 'median']:
        missing = []
        for role in ['source_0100', 'target_0100']:
            for record in read(base/role/'complete.json')['records']:
                with np.load(base/role/(record['camera']+'.npz')) as values:
                    if statistic not in values.files:
                        missing.append(role+'/'+record['camera'])
        if missing:
            rows.append(dict(statistic=statistic, skipped='Not cached; no additional field queries',
                             missing_maps=missing))
            continue
        out = r/'depth_statistics'/statistic
        out.mkdir(parents=True, exist_ok=True)
        def run(stage, script, *args):
            with (out/(stage+'.log')).open('w') as log:
                subprocess.run([sys.executable, str(scripts/script), *map(str,args)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        prediction = out/'prediction'
        if not (prediction/'complete.json').exists():
            if prediction.exists():
                raise RuntimeError('Incomplete render requires inspection: '+str(prediction))
            run('render', 'render_dec5_native_head_benchmark.py',
                '--source-rgb-data', r/'native_rgb/data', '--source-cameras', r/'source8_hd.json',
                '--source-depth', base/'source_0100', '--semantics', r/'frame/semantics',
                '--target-depth', base/'target_0100', '--fallback', r/'fallback4k',
                '--cameras', r/'target4k.json', '--depth-key', statistic,
                '--expected-train-count', '64', '--output', prediction)
        metrics = out/'metrics.json'
        if not metrics.exists():
            run('score', 'score_dec5_hair.py', '--root', r/'benchmark_scene',
                '--prediction', prediction, '--output', metrics,
                '--boundary-radius', '5', '--include-person-roi')
        rows.append(dict(statistic=statistic,metrics=read(metrics)['aggregate']))
    write(r/'depth_statistics/comparison.json', dict(results=rows,
          frozen_checkpoint=True, identical_source_rgb_and_fallback=True,
          minimum_opacity=.1, no_additional_training=True, no_depth_ground_truth=True))


if __name__ == '__main__':
    main()
