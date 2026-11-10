"""One CPU preparation/verified native transfer ahead of each remote GPU queue.

Native pixels and training parameters are unchanged. Source producers own their
existing frame lease; the receiving native-transfer lease serializes extraction.
"""
import shlex
import time
from common import atomic


def next_frame(queue,state):
    active=state.get('active')
    if not active or not active.startswith('frame'):return None
    for job in queue.get('jobs',[]):
        if job['id'].startswith('frame') and job['id'] not in state.get('jobs',{}):
            return int(job['id'][5:])
    return None


def prefetch(root,hosts):
    from campaign import Host
    checks=[]
    for host in hosts:
        if host.row.get('production_disabled'):continue
        row=dict(host=host.row['name'])
        try:
            settings=host.get(host.root/'settings.json')
            if not settings.get('source_owners'):continue
            frame=next_frame(host.get(host.agent_root/'queue.json',{}),host.get(host.agent_root/'agent_state.json',{}))
            if frame is None:continue
            row['frame']=frame
            # The source cache can already have been released after prefetch.
            # Check the receiver first, so we never regenerate that cache.
            transferred=host.get(host.root/'frames'/f'{frame:06d}'/'native/transfer_complete.json')
            if transferred:
                if transferred.get('frame')!=frame:raise ValueError('Prefetch frame receipt differs')
                row['status']='native_ready';checks.append(row);continue
            owner=Host(settings['source_owners'][0 if frame<=1111 else 1])
            receipt=owner.root/'prepared'/f'{frame:06d}'/'producer_complete.json'
            if owner.get(receipt):
                tools=settings.get('production_tools',settings['tools'])
                args=[settings['python'],tools+'/native_transfer.py','--root',str(host.root),'--frame',str(frame),'--nonblocking']
                command='nohup env CUDA_VISIBLE_DEVICES= PYTHONPATH='+shlex.quote(settings.get('dependency_path',''))+' '+shlex.join(args)
                command+=' >> '+shlex.quote(str(host.root/('prefetch_native_%06d.log'%frame)))+' 2>&1 < /dev/null &'
                host.command(command);row['status']='native_transfer_requested'
            else:
                source=owner.get(owner.root/'settings.json')
                tools=source.get('production_tools',source['tools'])
                args=[source['python'],tools+'/producer.py','--root',str(owner.root),'--frame',str(frame),'--nonblocking']
                command='nohup env CUDA_VISIBLE_DEVICES= PYTHONPATH='+shlex.quote(source['dependency_path'])+' '+shlex.join(args)
                command+=' >> '+shlex.quote(str(owner.root/('producer_%06d.log'%frame)))+' 2>&1 < /dev/null &'
                owner.command(command);row['status']='requested_or_already_running'
        except Exception as error:
            # Optional overlap must never interrupt training or terminal accounting.
            row.update(status='unavailable',error=str(error)[-1200:])
        checks.append(row)
    atomic(root/'prefetch_status.json',dict(time=time.time(),cpu_only=True,at_most_one_future_frame_per_remote_queue=True,checks=checks))
