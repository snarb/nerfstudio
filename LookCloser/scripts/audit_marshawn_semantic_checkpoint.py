"""Read-only schema2 replay: fixed semantic density support stays active in all profiles."""
import argparse,json,sys,os
from pathlib import Path
import numpy as np


def main(a):
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    sys.path.insert(0,str(a.code_root));sys.path.insert(0,str(a.code_root/'LookCloser/scripts'));sys.path.insert(0,str(Path(__file__).resolve().parent))
    # These replacements exist only in this independent audit process. The
    # frozen model/worker/snapshot files and running training are never changed.
    import marshawn_tuning_occupancy as old
    import marshawn_semantic_occupancy as semantic
    import audit_marshawn_tuning_checkpoint as base
    from marshawn_occupancy_audit import dilate_one
    import nerfstudio.utils.eval_utils as eval_utils
    original_setup=eval_utils.eval_setup
    def bound_setup(*args,**kwargs):
        result=original_setup(*args,**kwargs);model=result[1].model
        if model.config.adaptive_coarse_step_size!=.00625 or not getattr(model.field,'_semantic_admission_installed',False):raise ValueError('Actual semantic inference/coarse-step path mismatch')
        record=json.loads(a.receipt.read_text())
        if model.semantic_receipt!=record['semantic_receipt']:raise ValueError('Saved/live semantic source mismatch')
        (a.output/'actual_parameter_binding.json').write_text(json.dumps(dict(actual_coarse_step=.00625,semantic_density_gate_installed=True,semantic_receipt=model.semantic_receipt),indent=2)+'\n')
        return result
    eval_utils.eval_setup=bound_setup
    original_loader=old.load_snapshot;original_profiles=base.actual_profiles
    old.load_snapshot=semantic.load_snapshot
    def profiles(native,ema,radius,step,warmup,*,receipt,checkpoint):
        core,active,saved_ema,_,record=semantic.load_snapshot(receipt,checkpoint)
        with np.load(Path(receipt).with_suffix('.npz'))as z:admission=z['admission'].copy()
        if not np.array_equal(native,active)or not np.array_equal(np.asarray(ema).reshape(core.shape),saved_ema)or radius!=record['radius']or step!=record['checkpoint_step']:raise ValueError('Live semantic model differs from actual saved core/active/EMA')
        if bool(step<warmup)!=record['binary_warmup']:raise ValueError('Warmup mismatch')
        result=dict(native=active,base=core&admission,halo1=dilate_one(core)&admission,all_cells=np.ones_like(active))
        return result,dict(record,cpu_reconstruction_used=False,source_receipt=str(receipt),halo1_available=not record['binary_warmup'],all_profiles_keep_fixed_density_admission=True,base_is_native_core_intersect_admission=True,halo_is_native_core_dilated_then_intersect_admission=True)
    base.actual_profiles=profiles
    try:base.trace(a)
    finally:old.load_snapshot=original_loader;base.actual_profiles=original_profiles;eval_utils.eval_setup=original_setup

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','output','queries','receipt','checkpoint','config','code-root'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--memory-cap-gib',type=float,default=10.)
    a=p.parse_args();a.skip_wholegrid=True;a.allow_readonly_overlap=True;a.expected_sh='true';a.expected_density='trunc_exp';a.expected_projection='legacy';main(a)
