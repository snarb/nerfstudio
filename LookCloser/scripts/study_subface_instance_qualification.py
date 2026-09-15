"""Same semantic near qualification on the already verified subdivision control.

Only explicit paths/base binding change. Subdivision, mask and depth thresholds
are not refitted. The immutable coarse and subdivision studies are untouched.
"""
import argparse
from copy import deepcopy
from pathlib import Path

from joint_temporal_texture import read, sha, atomic_json
import study_instance_qualified_free_surface as worker

ROOT = Path('/mnt/data/dec5_subface_instance_qualified_free_surface')
PARENT = Path('/mnt/data/dec5_subface_free_space')


def configure():
    worker.ROOT = ROOT
    worker.PARENT = PARENT
    original_read = read

    def bound_read(path):
        value = original_read(path)
        if Path(path).resolve() == (PARENT/'000995/request.json').resolve():
            value = deepcopy(value)
            control = PARENT/'refined/000995'
            request, result = read(control/'request.json'), read(control/'result.json')
            assert result['control'] == 'refined'
            assert result['request_sha256'] == sha(control/'request.json')
            assert request['study_request_sha256'] == sha(PARENT/'000995/request.json')
            assert result['hashes']['mesh.ply'] == sha(control/'mesh.ply')
            value.update(original_production_mesh=value['mesh'],
                original_production_mesh_sha256=value['mesh_sha256'],
                mesh=str(control/'mesh.ply'), mesh_sha256=result['hashes']['mesh.ply'],
                subdivision_control_result_sha256=sha(control/'result.json'),
                semantic_parent_adapter_sha256=sha(ROOT/'adapter.json'))
            value['scripts'][str(Path(__file__).resolve())] = sha(__file__)
        return value
    worker.read = bound_read


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['geometry','prepare','render'])
    p.add_argument('--view');a=p.parse_args()
    if a.stage=='geometry':
        assert not ROOT.exists();ROOT.mkdir()
        atomic_json(ROOT/'adapter.json',dict(script_sha256=sha(__file__),
            worker_sha256=sha(worker.__file__),
            only_read_override=str(PARENT/'000995/request.json'),
            numerical_rules_changed=False, original_production_modified=False))
        configure();worker.geometry()
    else:
        import review_measured_free_surface as workflow
        workflow.ROOT=ROOT
        if a.stage=='prepare':workflow.prepare('000995')
        else:
            assert a.view in workflow.VIEWS
            workflow.render('000995',a.view)
