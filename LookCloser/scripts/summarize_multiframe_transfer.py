"""Separate CSV and worst-frame gate for a single global transfer recipe."""
import argparse
import csv
from pathlib import Path
import numpy as np
from luster_hybrid_common import read,write


def summarize(root,candidate):
    inventory=read(root/'preflight/completed_frames.json')['frames']
    frames=[];rows=[];pending=[]
    for item in inventory:
        frame=item['frame'];directory=root/'frames'/f'{frame:06d}'/'evaluation'
        bp=directory/'baseline_metrics.json';hp=directory/f'{candidate}_metrics.json'
        if not bp.exists() or not hp.exists():
            pending.append(frame);continue
        baseline=read(bp);hybrid=read(hp)
        assert baseline['protocol_sha256']==hybrid['protocol_sha256']
        assert baseline['device']==hybrid['device']
        assert baseline['resolution']==hybrid['resolution']=='native'
        protocol=read(directory/'protocol.json')
        assert set(map(int,protocol['cameras']))=={7,12,95,150}
        expected={(int(c),region) for c,e in protocol['cameras'].items()
                  for region in e['boxes']}
        base_scope={(v['camera'],v['region']) for v in baseline['rows']}
        hybrid_scope={(v['camera'],v['region']) for v in hybrid['rows']}
        assert base_scope==expected, 'Full baseline protocol coverage is required'
        missing=sorted(expected-hybrid_scope)
        unexpected=sorted(hybrid_scope-expected)
        assert len(hybrid_scope)==len(hybrid['rows']), 'Duplicate hybrid metric rows'
        b={(v['camera'],v['region']):v for v in baseline['rows'] if v['status']=='measured'}
        coverage={(v['camera'],v['region']):v for v in hybrid['coverage']}
        regressions=[]
        for h in hybrid['rows']:
            if h['status']!='measured':continue
            key=h['camera'],h['region'];ref=b[key]
            row=dict(frame=f'{frame:06d}',split=item['split'],candidate=candidate,
                     camera=h['camera'],region=h['region'],device=hybrid['device'],
                     protocol_sha256=hybrid['protocol_sha256'])
            for metric in ['lpips','psnr','ssim']:
                row['baseline_'+metric]=ref[metric];row['hybrid_'+metric]=h[metric]
            row['lpips_reduction_percent']=100*(1-h['lpips']/max(ref['lpips'],1e-12))
            row['preserved_fraction']=coverage[key]['preserved_fraction']
            rows.append(row)
            if h['lpips']>ref['lpips']+1e-6:
                regressions.append(dict(camera=h['camera'],region=h['region'],
                                        delta=h['lpips']-ref['lpips']))
        if hybrid['validation_mean'] is None:
            pending.append(frame);continue
        old=baseline['validation_mean']['lpips'];new=hybrid['validation_mean']['lpips']
        for metrics in [baseline,hybrid]:
            person={v['camera']:v['lpips'] for v in metrics['rows']
                    if v['region']=='person' and v['status']=='measured'}
            assert {12,95,150}<=set(person)
            assert np.isclose(metrics['validation_mean']['lpips'],
                              np.mean([person[c] for c in [12,95,150]]),rtol=0,atol=1e-10)
        gain=100*(1-new/old)
        visual=directory/f'{candidate}_visual_review.json'
        visual_status=read(visual)['status'] if visual.exists() else 'pending'
        frames.append(dict(frame=frame,baseline_lpips=old,hybrid_lpips=new,
            reduction_percent=gain,meets_30_percent=gain>=30,
            regressions=regressions,visual_review=visual_status,
            missing_regions=missing,unexpected_regions=unexpected,
            metric_scope_complete=not missing and not unexpected and
                set(b)=={(v['camera'],v['region']) for v in hybrid['rows']
                         if v['status']=='measured'}))
    out=root/'evaluation'
    if rows:
        with (out/f'{candidate}_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(candidate=candidate,frames=frames,pending=pending,
        goal_achieved=bool(frames) and not pending and all(
            f['meets_30_percent'] and f['metric_scope_complete'] and
            not f['regressions'] and f['visual_review']=='accepted'
            for f in frames),
        minimum_frame_reduction_percent=min((f['reduction_percent'] for f in frames),default=None),
        scope='Native heldout-person mean in each frame separately; cam007 and spatial regions checked for regressions; visual review required')
    write(out/f'{candidate}_summary.json',report)
    print(dict(completed=len(frames),pending=len(pending),
               minimum_frame_reduction_percent=report['minimum_frame_reduction_percent'],
               goal_achieved=report['goal_achieved']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidate',required=True);a=p.parse_args();summarize(a.root,a.candidate)
