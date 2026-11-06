"""CPU safeguards for source discovery and numerical prior gate decisions."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np


spec=importlib.util.spec_from_file_location('native4k_executor',Path(__file__).parents[1]/'scripts/run_marshawn_native_4k_technical.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def source_discovery_binds_selected_runtime_and_retains_all_controls(tmp_path,case):
    production=tmp_path/'production';rows=[]
    for step in [20000,25000,30000]:
        ck=production/('step-%09d.ckpt'%step);ck.parent.mkdir(exist_ok=True);ck.write_bytes(str(step).encode());rows.append(dict(step=step,checkpoint=str(ck),checkpoint_sha256=runner.sha(ck),renders=str(production/'renders'/str(step)),metrics=dict(psnr=30,ssim=.8,lpips=.05)))
    selected=rows[1];write(production/'state.json',dict(selection=selected,evaluations=rows,plateau=dict(stopped=True)));write(production/'complete.json',dict(selection=selected));write(production/'runtime/step-000025000.json',dict(dynamic_num_rays_per_batch=6800));grid=production/'grids/step-000025000_grid.npz';grid.parent.mkdir();grid.write_bytes(b'grid');write(grid.with_suffix('.json'),dict(checkpoint_sha256=selected['checkpoint_sha256'],grid_archive_sha256=runner.sha(grid)))
    found=runner.discover_source(production,{})
    assert found['step']==25000 and len(found['all_HD_checkpoint_controls'])==3
    Path(rows[2]['checkpoint']).write_bytes(b'changed')
    with case.assertRaises(AssertionError):runner.discover_source(production,{})


def prior_gate_accepts_tolerance_and_rejects_geometry_gradient_change(tmp_path,case):
    reference=tmp_path/'reference';candidate=tmp_path/'candidate';reference.mkdir();candidate.mkdir();meta=dict(parameter_names=['density'],parameter_shapes=[[2]],step=35001,values={'free':.01})
    for path in [reference,candidate]:write(path/'prior_probe.json',meta)
    np.savez(reference/'prior_gradients.npz',free_gradient=np.array([.01,.02],np.float32));np.savez(candidate/'prior_gradients.npz',free_gradient=np.array([.01,.02+1e-7],np.float32));limits=dict(gradient_rtol=1e-4,gradient_atol=1e-6,value_rtol=1e-5,value_atol=1e-7)
    assert runner.compare_priors(reference,candidate,limits)['values_close']
    np.savez(candidate/'prior_gradients.npz',free_gradient=np.array([.01,.021],np.float32))
    with case.assertRaises(AssertionError):runner.compare_priors(reference,candidate,limits)


def runtime_release_rejects_changed_bound_helper(tmp_path,case):
    path=tmp_path/'helper.py';path.write_text('original');bindings={str(path):runner.sha(path)};runner.verify_bindings(bindings);path.write_text('changed')
    with case.assertRaises(ValueError):runner.verify_bindings(bindings)


class ExecutorChecks(unittest.TestCase):
    def run_case(self,function):
        with tempfile.TemporaryDirectory() as directory:function(Path(directory),self)
    def test_source_selection(self):self.run_case(source_discovery_binds_selected_runtime_and_retains_all_controls)
    def test_prior_gate(self):self.run_case(prior_gate_accepts_tolerance_and_rejects_geometry_gradient_change)
    def test_hash_binding(self):self.run_case(runtime_release_rejects_changed_bound_helper)


if __name__=='__main__':unittest.main()
