"""CPU request-validation and inherited-lease regression for MAT/map fairness."""
from pathlib import Path
import fcntl, json, subprocess, sys, tempfile, time
import marshawn_future_sg_frequency_v3 as broker
import marshawn_sam3_assist as io


def run():
    with tempfile.TemporaryDirectory() as temporary:
        V = Path(temporary); alias = V / 'operations/sam3_assist/v5/frame001680/props_alias_ready.json'
        io.write(alias, dict(frame=1680, records={str(i): {} for i in range(66)},
                            current_ALL66_native_aliases_verified=True, source_ops_sha256='qualified-code'))
        request = V / 'operations/advanced_inputs/001680/mat_request.json'
        row = dict(kind='MAT_NATIVE_GPU_REQUEST_V1', status='pending', frame=1680, requested_slot=4,
                   request_nonce='cpu-regression', raw_inputs_ready=True, source_ops_sha256='qualified-code',
                   raw_alias_receipt_path=str(alias), raw_alias_receipt_sha256=io.sha(alias))
        io.write(request, row); config = dict(mat_source_ops_sha256=['qualified-code'])
        materials = broker.material_requests(V, config)
        assert [r['gpu'] for r in materials] == [4]
        assert broker.eligible_gpus(list(range(8)), False, False, [4]) == [0, 1, 2, 3, 5, 6, 7]
        assert broker.eligible_gpus(list(range(8)), True, False, [4]) == [0, 2, 3]
        assert broker.eligible_gpus(list(range(8)), False, True, [4]) == []
        row['requested_slot'] = 7; io.write(request, row)
        assert broker.material_requests(V, config)[0]['gpu'] == 7
        assert set(broker.eligible_gpus(list(range(8)), False, False, [7])) - set(range(5)) == {5, 6}
        io.write(alias, dict(frame=1680, records={}))
        try:
            broker.material_requests(V, config)
            raise AssertionError('Changed aliases accepted')
        except ValueError:
            pass
        row['status'] = 'complete'; io.write(request, row)
        assert not broker.material_requests(V, config)
        # A completed field/request releases scheduling capacity; source inputs
        # are still independently checked by the unchanged MAT worker.
        path = V / 'lookcloser_tuning_gpu_TEST_4.lock'
        fit_lock = path.open('a+'); fcntl.flock(fit_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fit = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(.20)'], pass_fds=(fit_lock.fileno(),))
        fit_lock.close()  # crash-surviving fit owns the inherited descriptor
        mat = path.open('a+'); start = time.monotonic(); other_maps = 0
        while True:
            try:
                fcntl.flock(mat, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                # These other physical slots stay eligible while slot4 drains.
                assert set(broker.eligible_gpus(list(range(8)), False, False, [4])) - {0, 1, 2} == {3, 5, 6, 7}
                other_maps += 1
                if time.monotonic() - start > 2:
                    raise AssertionError('MAT could not take the retired fit lease')
                time.sleep(.01)
        elapsed = time.monotonic() - start; fit.wait(); mat.close()
        assert elapsed >= .15 and other_maps > 0 and fit.returncode == 0
    print(json.dumps(dict(passed=True, source_alias_hash_guard=True, MAT_slot4_drains_naturally=True,
                          inherited_fit_lease_retained=True, MAT_acquired_after_seconds=round(elapsed, 3),
                          other_slot_progress_events=other_maps)))


if __name__ == '__main__':
    run()
