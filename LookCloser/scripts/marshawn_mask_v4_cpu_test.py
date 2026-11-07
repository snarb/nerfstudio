"""Queue timeout regression: healthy multi-wave native jobs never fall back."""
from marshawn_sam3_assist_controller_v4 import native_queue_inactive,can_fallback


def main():
    state={'jobs':{},'cameras':{}}
    for i in range(52):
        name='view%d'%i;state['cameras'][name]={'name':name,'collected':False}
        state['jobs'][name]={'status':'running' if i<4 else 'pending'}
    for t in (0,61,120,600,3600):
        assert not native_queue_inactive(state,t,60)
        assert not can_fallback(state['jobs']['view20'],False)
    for i in range(4):
        state['jobs']['view%d'%i]={'status':'complete','supervisor_alive':True} # stale terminal receipt must not hold forever
        state['cameras']['view%d'%i]['collected']=True
    assert not native_queue_inactive(state,4000,60)
    assert not native_queue_inactive(state,4059,60)
    assert native_queue_inactive(state,4060,60)
    assert can_fallback(state['jobs']['view20'],True)
    assert not can_fallback({'status':'pending','host':'silver'},True),'Claimed/delayed launch cannot duplicate locally'
    state['jobs']['view4']={'status':'unknown','worker_alive':False}
    assert not native_queue_inactive(state,10000,60),'UNKNOWN owner must remain adopted'
    state['jobs']['view4']={'status':'orphaned','worker_alive':True}
    assert not native_queue_inactive(state,20000,60),'Surviving child cannot be duplicated'
    state['jobs']['view4']={'status':'complete'}
    assert not native_queue_inactive(state,30000,60),'Fresh uncollected progress resets inactivity'
    print('PASS:52-view waves beyond60s stayqueued; real idle fallback; UNKNOWN/orphans/claimed jobs protected')


if __name__=='__main__':main()
