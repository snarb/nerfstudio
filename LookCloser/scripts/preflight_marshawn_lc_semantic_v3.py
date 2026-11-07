"""CPU replay of actual worker import order from the packaged working directory."""
import argparse
import json
import os
from pathlib import Path
import sys

import run_marshawn_lc_semantic_capture_v3 as worker


def main(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU import preflight requires empty CUDA_VISIBLE_DEVICES')
    m=json.loads(a.inputs.read_text());root=a.inputs.parent
    if Path.cwd().resolve() != (root/'code').resolve():
        raise ValueError('Replay must run from actual package working directory')
    helpers,branch=[Path(m[k])for k in ['helpers','branch']]
    isolation=worker.absolute('_lc_capture_gpu_isolation',helpers/'marshawn_gpu_isolation.py')
    sys.path[:0]=[str(helpers),str(branch/'code'),str(branch/'code/LookCloser/scripts')]
    import numpy as np
    import torch
    adapter=worker.absolute('_reviewed_lc_native_capture',root/'code/marshawn_lc_semantic_capture.py')
    torch.set_num_threads(2)
    late=worker.absolute('_lc_capture_tree_digest',helpers/'train_marshawn_semantic_late_distortion.py')
    import marshawn_semantic_occupancy as semantic
    import nerfstudio.utils.eval_utils as eu
    from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
    if torch.cuda.is_initialized():raise ValueError('CPU preflight initialized CUDA')
    bindings=worker.project_import_bindings(m,root)
    synthetic={name:getattr(sys.modules.get(name),'__file__',None)for name in ['torch.ops','torch.classes']}
    if 'torch.ops' in bindings or 'torch.classes' in bindings:
        raise ValueError('Virtual Torch namespaces incorrectly classified as source files')
    worker.write(root/'CPU_actual_worker_import_bindings.json',bindings)
    worker.write(root/'CPU_actual_worker_import_order.json',dict(passed=True,input_spec_sha256=worker.sha(a.inputs),
        cwd=str(Path.cwd()),actual_project_helper_modules=len(bindings),synthetic_modules=synthetic,
        replay='isolation -> Torch/NumPy -> native adapter -> tree digest -> semantic -> eval_utils -> trainer RNG -> byte proof',
        CUDA_initialized=False,model_constructed=False,checkpoint_loaded=False,GPU_started=False,
        deliberately_not_executed='GPU lease/device/resource allocation/model setup are runtime-only gates; CPU checks import order and binding logic only.'))
    print(json.dumps(dict(passed=True,actual_imported_files=len(bindings),synthetic_modules=synthetic,CUDA_initialized=False,model_constructed=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True)
    main(p.parse_args())
