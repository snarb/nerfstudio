import multiprocessing as mp
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))


def worker(root,barrier,active,peak):
    import source_surface_confidence_luster_hybrid as module
    def build(*args):
        with active.get_lock():
            active.value+=1;peak.value=max(peak.value,active.value)
        time.sleep(.2)
        with active.get_lock():active.value-=1
        return root/'source_surface_confidence/cam008'
    module._source_confidence=build
    barrier.wait(timeout=10)
    assert module.source_confidence(root,8,{})==root/'source_surface_confidence/cam008'


def test_same_source_build_is_serialized(tmp_path):
    context=mp.get_context('fork');barrier=context.Barrier(2)
    active=context.Value('i',0);peak=context.Value('i',0)
    children=[context.Process(target=worker,args=(tmp_path,barrier,active,peak)) for _ in range(2)]
    for child in children:child.start()
    for child in children:
        child.join(timeout=15)
        try:assert child.exitcode==0
        finally:
            if child.is_alive():child.terminate();child.join()
    assert peak.value==1 and active.value==0
