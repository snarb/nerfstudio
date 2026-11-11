"""Continue durable work stealing after the sparse24 barrier is complete."""
import argparse
import fcntl
import json
import shlex
import time
from pathlib import Path

from common import atomic, read
from retiring_host import RetiringHost
import priority_scheduler as scheduler

ALL_FRAMES=set(range(908,1269))


def plan(queues,states,roots,unavailable=()):
    previous=scheduler.PRIORITY
    try:
        scheduler.PRIORITY=ALL_FRAMES
        return scheduler.plan(queues,states,roots,unavailable)
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
    new,moves=plan(queues,states,{n:h.root for n,h in hosts.items()},unavailable={n for n,h in hosts.items() if h.offline})
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
