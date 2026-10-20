import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from luster_hybrid_common import write,read
from summarize_multiframe_transfer import summarize


def test_partial_screen_cannot_pass_full_goal(tmp_path):
    directory=tmp_path/'frames/000545/evaluation';directory.mkdir(parents=True)
    write(tmp_path/'preflight/completed_frames.json',
          {'frames':[{'frame':545,'split':'development'}]})
    (tmp_path/'evaluation').mkdir()
    protocol={'cameras':{str(c):{'boxes':{'person':[0,0,10,10],'detail':[1,1,5,5]}}
                         for c in [7,12,95,150]}}
    write(directory/'protocol.json',protocol)
    def row(region,value,camera=12):
        return dict(camera=camera,region=region,status='measured',lpips=value,psnr=30,ssim=.9,
                    pixels=100 if region=='person' else 16)
    baseline=dict(resolution='native',device='cpu',protocol_sha256='same',
                  rows=[row(r,v,c) for c in [7,12,95,150] for r,v in [('person',.1),('detail',.2)]],
                  validation_mean={'lpips':.1})
    hybrid_rows=[row(r,v,c) for c in [7,12,95,150] for r,v in [('person',.06),('detail',.1)]
                 if (c,r)!=(12,'detail')]
    hybrid={**baseline,'rows':hybrid_rows,
            'validation_mean':{'lpips':.06},
            'coverage':[dict(camera=r['camera'],region=r['region'],preserved_fraction=.2)
                        for r in hybrid_rows]}
    write(directory/'baseline_metrics.json',baseline)
    write(directory/'v_metrics.json',hybrid)
    write(directory/'v_visual_review.json',{'status':'accepted'})
    summarize(tmp_path,'v')
    result=read(tmp_path/'evaluation/v_summary.json')
    assert not result['goal_achieved']
    assert result['frames'][0]['missing_regions']==[[12,'detail']]
    hybrid['rows'].append(row('detail',.1))
    hybrid['coverage'].append(dict(camera=12,region='detail',preserved_fraction=.1))
    write(directory/'v_metrics.json',hybrid)
    summarize(tmp_path,'v')
    assert read(tmp_path/'evaluation/v_summary.json')['goal_achieved']
    import csv
    with (tmp_path/'evaluation/v_metrics.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert rows[0]['support_pixels']=='100'
    assert rows[0]['support_fraction']=='1.0'
    import pytest
    hybrid['rows'][0]['pixels']=99
    write(directory/'v_metrics.json',hybrid)
    with pytest.raises(AssertionError,match='identical support'):
        summarize(tmp_path,'v')
