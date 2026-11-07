"""CPU checks for the failure modes specific to independent frame scheduling."""
import copy, tempfile
from pathlib import Path
from marshawn_video_campaign import create, jobs, retry_failed
from marshawn_video_train import owned, selectors
from marshawn_video_metrics import actor_visibility,aggregate_visible
from marshawn_tuning_metrics import DOMAIN,aggregate_actor_metrics
import marshawn_tuning_controller as fleet


def run():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);plan=dict(schema=1,frames=[1660,1665],retry_base_seconds=0,pipeline_sha256='0'*64,
            hosts={'host':dict(python='/python',controller='/controller',root='/campaign',gpus=[0,1])})
        fleet.atomic(root/'input.json',plan);_,state=create(root,root/'input.json')
        assert [j['name'] for j in jobs(plan,state)]==['frame001660_a00','frame001665_a00']
        item=state['frames']['1660'];old=item['name'];state['jobs'][old].update(status='failed',finished=0)
        assert retry_failed(plan,state,100)==[],'Missing remote proof must not retry'
        remote=dict(supervisor_alive=False,worker_alive=True)
        state['host_snapshots']={'host':dict(jobs={old:remote})}
        assert retry_failed(plan,state,100)==[],'Surviving optimizer must not duplicate'
        remote['worker_alive']=False;state['host_snapshots']['host']['error']='SSH timeout'
        assert retry_failed(plan,state,100)==[],'SSH UNKNOWN must not become dead'
        del state['host_snapshots']['host']['error']
        assert retry_failed(plan,state,100)==[1660]
        assert item['name']=='frame001660_a01' and item['host']=='host'
        assert state['jobs'][old]['status']=='failed','Keep immutable previous attempts'
        changed=jobs(plan,state)[0]
        assert '--frame' in changed['command'] and '1660' in changed['command']
        fleet.atomic(root/'state.json',state);_,restored=create(root,root/'input.json')
        assert restored==state,'Controller restart must adopt same attempt'
        try:owned(root/'frames/001660/train/checkpoints',root/'frames/001665/train/checkpoints/step-1.ckpt')
        except ValueError:pass
        else:raise AssertionError('Foreign frame checkpoint accepted')
        try:owned(root/'checkpoints',root/'checkpoints/../../foreign.ckpt')
        except ValueError:pass
        else:raise AssertionError('Checkpoint traversal accepted')
        rows=[dict(step=1,metrics=dict(lpips=.10,psnr=30)),
              dict(step=2,metrics=dict(lpips=.1005,psnr=31)),
              dict(step=3,metrics=dict(lpips=.12,psnr=31.05))]
        selected=selectors(rows)
        assert selected['actor_lpips']['step']==1,'Raw LPIPS must not use historical .001 tie'
        assert selected['actor_psnr07']['step']==2
        class Mask:
            def __init__(self,value):self.value=value
            def any(self):return self.value
        absent=dict(camera='DC',crops={'black':None},visibility={'black':'absent'})
        try:actor_visibility(absent,'black',Mask(False))
        except ValueError:pass
        else:raise AssertionError('Unreviewed empty mask became ABSENT')
        absent['visibility_review']={'black':dict(reviewed=True,evidence='source-bound review record')}
        assert actor_visibility(absent,'black',Mask(False))=='absent'
        invisible=dict(camera='DC',crops={'black':None},visibility={'black':'unknown'})
        try:actor_visibility(invisible,'black',Mask(False))
        except ValueError:pass
        else:raise AssertionError('UNKNOWN actor contributed a validation metric')
        metrics=[dict(view=v,actor=a,domain=DOMAIN,psnr=30+i,ssim=.8+i*.01,lpips=.2-i*.01,validation_loss=.1)
                 for i,(v,a) in enumerate(( (v,a) for v in ['DC','HC','KA'] for a in ['black','grey']))]
        full=aggregate_visible(metrics,dict(validation=['DC','HC','KA']))
        assert full['metrics']==aggregate_actor_metrics(metrics)['metrics'],'Complete-pair metrics must be identical'
        partial=aggregate_visible(metrics[1:],dict(validation=['DC','HC','KA']))
        assert partial['visible_pair_count']==5 and partial['visible_cameras_by_actor']['black']==['HC','KA']
        assert partial['metrics']['lpips']==sum(a['lpips'] for a in partial['by_actor'].values())/2
        try:aggregate_visible([m for m in metrics if m['actor']=='black'],dict(validation=['DC','HC','KA']))
        except ValueError:pass
        else:raise AssertionError('Actor without visible monitor accepted')
    print('PASS: cold-frame ownership, SSH-UNKNOWN/live-worker retry fences, restart adoption, raw LPIPS selector')


if __name__=='__main__':run()
