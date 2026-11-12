"""Continue durable work stealing after the sparse24 barrier is complete."""
import argparse
import copy
import fcntl
import json
import shlex
import time
from pathlib import Path

from common import atomic, read
from retiring_host import RetiringHost
import priority_scheduler as scheduler

ALL_FRAMES=set(range(908,1269))


def plan(queues,states,roots,unavailable=(),ready_frames=(),local_receivers=()):
    previous=scheduler.PRIORITY
    try:
        scheduler.PRIORITY=ALL_FRAMES
        prepared=copy.deepcopy(queues);ready=set(ready_frames);local=set(local_receivers);moves=[]
        # Only local consumers can use canonical native data without a transfer.
        # Keep each live local donor's sole prepared next job on that donor.
        for receiver in prepared:
            if receiver not in local or receiver in unavailable:continue
            if any(scheduler.frame(j) in ready for j in scheduler.pending(prepared[receiver],states[receiver])):continue
            candidates=[]
            for donor,q in prepared.items():
                if donor==receiver:continue
                tasks=[j for j in scheduler.pending(q,states[donor]) if scheduler.frame(j) in ready]
                if donor in local and donor not in unavailable and len(tasks)<2:continue
                if tasks:candidates.append((donor in unavailable,len(tasks),donor,tasks[-1]))
            if not candidates:continue
            _,_,donor,job=max(candidates,key=lambda x:x[:3])
            template=prepared[receiver]['jobs'][0]
            prepared[donor]['jobs'].remove(job)
            prepared[receiver]['jobs'].append(scheduler.relocate(job,template,roots[receiver]))
            moves.append(dict(frame=scheduler.frame(job),donor=donor,receiver=receiver,reason='verified_local_native'))
        result,extra=scheduler.plan(prepared,states,roots,unavailable)
        for name in local:
            if name in result:
                result[name]['jobs'].sort(key=lambda j:0 if j['id'] in states[name].get('jobs',{}) else 1 if scheduler.frame(j) in ready else 2)
        return result,moves+extra
    finally:
        scheduler.PRIORITY=previous


def observe(host):
    queue=host.get(host.agent_root/'queue.json')
    state=host.get(host.agent_root/'agent_state.json')
    if not queue or not state:raise ValueError('Incomplete fleet observation: '+host.row['name'])
    for job in state.get('jobs',{}).values():
        if job.get('status')=='complete' and job.get('job',{}).get('completion'):
            host.get(Path(job['job']['completion']))
    # An explicitly stopped workstation executor is no longer a GPU receiver.
    # Refresh its ledger first; activation retains only SHA-backed completions.
    if host.enabled and not host.offline:
        status=host.command('systemctl is-active marshawn-sequence-agent.service || true').strip()
        if status in ('inactive','failed'):
            host.activate()
            queue=host.get(host.agent_root/'queue.json')
            state=host.get(host.agent_root/'agent_state.json')
    return queue,state


def tick(root):
    if not (root/'sparse24_complete.json').exists():
        raise ValueError('Sparse24 barrier must complete before dense scheduling')
    hosts={r['name']:RetiringHost(r,root) for r in read(root/'fleet.json')['hosts'] if not r.get('production_disabled')}
    scheduler.recover(root,hosts)
    observed={n:observe(h) for n,h in hosts.items()}
    queues={n:q for n,(q,s) in observed.items()};states={n:s for n,(q,s) in observed.items()}
    ready={f for f in ALL_FRAMES if (root/'frames'/f'{f:06d}/native/transfer_complete.json').exists() and (root/'frames'/f'{f:06d}/native/source_cache_released.json').exists()}
    local={n for n,h in hosts.items() if h.row.get('local') and h.root==root}
    new,moves=plan(queues,states,{n:h.root for n,h in hosts.items()},unavailable={n for n,h in hosts.items() if h.offline},ready_frames=ready,local_receivers=local)
    changes={n:dict(before=queues[n],after=new[n]) for n in hosts if queues[n]!=new[n]}
    if changes:
        for n in changes:hosts[n].put(hosts[n].agent_root/'SCHEDULING',dict(time=time.time(),owner='dense_scheduler'))
        time.sleep(6)
        try:
            for n in changes:
                h=hosts[n];now=h.get(h.agent_root/'agent_state.json')
                if h.get(h.agent_root/'queue.json')!=queues[n] or now.get('jobs')!=states[n].get('jobs'):
                    raise ValueError('Concurrent queue/worker transition; retry next tick')
        except Exception:
            for n in changes:hosts[n].command('rm -f '+shlex.quote(str(hosts[n].agent_root/'SCHEDULING')))
            raise
        atomic(root/'priority_queue_transaction.json',dict(time=time.time(),changes=changes,moves=moves,committed=False,owner='dense_scheduler'))
        scheduler.recover(root,hosts)
        with (root/'dense_queue_history.jsonl').open('a') as stream:stream.write(json.dumps(dict(time=time.time(),moves=moves))+'\n')
    completed={int(k[5:]) for s in states.values() for k,v in s.get('jobs',{}).items() if k.startswith('frame') and v.get('status')=='complete'}
    status=dict(time=time.time(),completed=len(completed),total=361,offline=[n for n,h in hosts.items() if h.offline],moves=moves,active={n:s.get('active') for n,s in states.items()},all_completed=completed==ALL_FRAMES)
    atomic(root/'dense_scheduler_status.json',status)
    return status['all_completed']


def run(root,once=False):
    root=Path(root)
    with (root/'priority_scheduler.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if tick(root) or once:return
            except Exception as error:
                with (root/'dense_scheduler_errors.jsonl').open('a') as stream:stream.write(json.dumps(dict(time=time.time(),error=repr(error)))+'\n')
                if once:raise
            time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args();run(args.root,args.once)
