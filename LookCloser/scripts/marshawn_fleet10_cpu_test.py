"""CPU ownership/assignment regression for the ten physical training slots."""
import copy, json
import marshawn_fleet10 as fleet10


def run():
    plan = dict(hosts={'silver': {'gpus': list(range(8))}, 'olive': {'gpus': [0]}, 'dev3': {'gpus': [0]}},
                host_preference=['olive', 'dev3', 'silver'])
    original = dict(frames={}, jobs={})
    for i, frame in enumerate(range(1660, 1710, 5)):
        name = 'frame%06d_a00' % frame
        original['frames'][str(frame)] = dict(host='silver', name=name, attempt=0)
        original['jobs'][name] = dict(status='pending')
        if i < 3:
            original['jobs'][name].update(status='running', host='silver', gpu=i, spec_sha256=str(i),
                                          worker={'pid': 100+i}, supervisor={'pid': 200+i})
    before = copy.deepcopy(original); state = fleet10.inherited_state(plan, original)
    assert original == before
    for frame in ['1660', '1665', '1670']:
        name = state['frames'][frame]['name']
        assert state['jobs'][name] == original['jobs'][name]
    snapshots = {h: dict(gpus={str(g): dict(processes=[], used_mib=0, global_lease_free=True) for g in row['gpus']})
                 for h, row in plan['hosts'].items()}
    assignments = []
    for frame in range(1675, 1710, 5):
        host, gpu = fleet10.select_slot(plan, state, assignments, snapshots)
        assignments.append(dict(frame=frame, host=host, gpu=gpu))
    assert assignments[0]['host'] == 'olive' and assignments[1]['host'] == 'dev3'
    ownership = {('silver', i) for i in range(3)} | {(a['host'], a['gpu']) for a in assignments}
    assert len(ownership) == 10 and fleet10.select_slot(plan, state, assignments, snapshots) is None
    # Unknown CUDA owners and a held global lease can never be claimed.
    blocked = copy.deepcopy(snapshots)
    blocked['olive']['gpus']['0']['processes'] = [{'pid': 999}]
    blocked['dev3']['gpus']['0']['global_lease_free'] = False
    assert fleet10.select_slot(plan, state, [], blocked) == ('silver', 3)
    print(json.dumps(dict(passed=True, active_worker_records_preserved=True, original_ledger_unchanged=True,
                          distinct_slots=len(ownership), first_two_new_hosts=[a['host'] for a in assignments[:2]],
                          unknown_owners_and_held_leases_excluded=True)))


if __name__ == '__main__':
    run()
