"""Exercise the real per-campaign lock with two processes and stubbed CUDA."""
import multiprocessing as mp
from pathlib import Path
import sys
import time
import types

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))


def worker(root,barrier,active,peak):
    import guarded_transfer_inference as guard
    guard.telemetry=lambda:dict(total_mib=97887,used_mib=8000,free_mib=89887,own_mib=0)
    def render(args):
        with active.get_lock():
            active.value+=1
            peak.value=max(peak.value,active.value)
        time.sleep(.2)
        with active.get_lock():active.value-=1
    sys.modules['torch']=types.SimpleNamespace(set_num_threads=lambda n:None,
        cuda=types.SimpleNamespace(set_per_process_memory_fraction=lambda *args:None))
    sys.modules['occupancy_native_luster_hybrid']=types.SimpleNamespace(run=render)
    barrier.wait(timeout=10)
    guard.run(types.SimpleNamespace(root=root,chunk=1,cameras=[7],occupancy_only=True,aabb_only=False))


def test_two_own_inferences_queue_without_overlapping(tmp_path):
    context=mp.get_context('fork')
    active=context.Value('i',0);peak=context.Value('i',0);barrier=context.Barrier(2)
    processes=[]
    for frame in [544,549]:
        root=tmp_path/'frames'/str(frame)
        (root/'logs').mkdir(parents=True)
        (root/'context.json').write_text('{}')
        process=context.Process(target=worker,args=(root,barrier,active,peak))
        processes.append(process);process.start()
    for process in processes:
        process.join(timeout=15)
        try:assert process.exitcode==0
        finally:
            if process.is_alive():process.terminate();process.join()
    assert peak.value==1 and active.value==0
