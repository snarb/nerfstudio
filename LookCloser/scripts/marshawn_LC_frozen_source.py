"""Independent LC semantics loader; source restoration copied from reviewed capture.

No source optimizer, occupancy callback, RGB target or mask is loaded for fitting.
Physical evaluation is separately authorized. Imports/inputs are explicitly bound.
"""
import json
import os
from pathlib import Path
import sys
from prepare_marshawn_frozen_semantic_cpu import sha, write


def load_source(spec, output, lease, release):
    import importlib.util
    source_plan, source_release = Path(spec['LC_source_plan']), Path(spec['LC_source_release'])
    m = json.loads(source_plan.read_text())
    worker = source_plan.parent/'code'/spec['LC_source_worker_name']
    if sha(worker) != m['code_sha256'][worker.name]: raise ValueError('Reviewed source loader changed')
    sys.path.insert(0,str(source_plan.parent/'code'))
    loader = importlib.util.spec_from_file_location('_direct_LC_reviewed_capture_loader',worker)
    source_runner = importlib.util.module_from_spec(loader); sys.modules[loader.name] = source_runner; loader.loader.exec_module(source_runner)
    source_runner.authorize(source_plan,source_release,Path(m['campaign']))
    binding_contract = dict(m, bound_inputs=dict(m['bound_inputs'], **release['bound_files']))
    helpers, branch, archive = [Path(m[k]) for k in ['helpers','branch','source_archive']]
    absolute = source_runner.absolute
    isolation = absolute('_direct_LC_physical_isolation',helpers/'marshawn_gpu_isolation.py')
    sys.path[:0] = [str(helpers),str(branch/'code'),str(branch/'code/LookCloser/scripts')]
    import numpy as np
    import torch
    if torch.__version__ != m['torch_version'] or torch.version.cuda != m['cuda_version']:
        raise ValueError('Unreviewed source runtime')
    identity = isolation.verify_cuda(lease['expected_physical_uuid'],torch)
    if identity['actual_physical_uuid'].lower() != m['expected_physical_UUID'].lower():
        raise ValueError('Source field must use reviewed exact physical UUID')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(m['allocator_cap_gib']*2**30/torch.cuda.get_device_properties(0).total_memory)
    import threading
    guard_done=threading.Event()
    threading.Thread(target=source_runner.monitor_resources,args=(m,lease['expected_physical_uuid'],output,guard_done),daemon=True).start()
    # Identical source eval loader/config callback to reviewed native_A, with
    # inference mode suppressing image/mask caching. No optimizer constructed.
    late = absolute('_lc_capture_tree_digest', helpers/'train_marshawn_semantic_late_distortion.py')
    import marshawn_semantic_occupancy as semantic
    import nerfstudio.utils.eval_utils as eu
    from nerfstudio.engine.trainer import _capture_rng_state, _restore_rng_state
    write(output/'import_binding_before_model.json',source_runner.project_import_bindings(binding_contract,source_plan.parent))
    cp = Path(m['local_checkpoint'])
    payload = torch.load(cp, map_location='cpu', weights_only=False)
    keys = ['pipeline', 'optimizers', 'schedulers', 'scalers', 'rng_state', 'tuning_runtime', 'actual_occupancy_capture']
    serialized = {k: late.tree_digest(payload[k]) for k in keys}
    if serialized != m['expected_serialized_states_sha256']:
        raise ValueError('Selected seven source states differ from CPU-qualified checkpoint')
    if payload['step'] != m['source_step']:
        raise ValueError('Wrong exact source step')
    def configure(cfg):
        cfg.eval_checkpoint = cp
        for dotted,expected in m['expected_config_paths'].items():
            actual=cfg
            for key in dotted.split('.'):actual=getattr(actual,key)
            if str(actual) != expected:
                raise ValueError('Materialized config path differs: '+dotted)
        return cfg
    cfg, pipe, actual_checkpoint, step = eu.eval_setup(Path(m['local_config']), test_mode='inference', update_config_callback=configure)
    model = pipe.model
    pipe.eval(); pipe.requires_grad_(False); model.current_train_step = step
    if Path(actual_checkpoint).resolve() != cp.resolve() or step != m['source_step'] or late.tree_digest(pipe.state_dict()) != serialized['pipeline']:
        raise ValueError('Strict full native pipeline restoration failed')
    c = model.config
    if (c.adaptive_coarse_step_size != .0015625 or c.corrected_arm_allocator or c.max_steps_per_ray != 1024 or
            c.occupancy_dilation_radius != 0 or c.occupancy_eval_dilation_radius != 0 or
            c.transmittance_threshold != 0 or c.adaptive_fixed_fallback_samples_per_ray != 0 or
            c.background_color != 'black' or model.field.density_activation != 'trunc_exp' or
            not model.field.correct_sh_directions or not getattr(model.field, '_semantic_admission_installed', False)):
        raise ValueError('Selected native source recipe differs')
    core, active, ema, _, grid_receipt = semantic.load_snapshot(archive/'grid.json', cp)
    with np.load(archive/'grid.npz', allow_pickle=False) as z:
        admission = z['admission'].copy()
    for expected, actual in [(active, model.occupancy_grid.binaries), (ema, model.occupancy_grid.occs.reshape(ema.shape)),
                             (core, model.semantic_learned_core), (admission.reshape(model.semantic_admission.shape), model.semantic_admission)]:
        if not np.array_equal(expected, actual.detach().cpu().numpy()):
            raise ValueError('Source actual core/EMA/admission/active mismatch')
    if model.config.feature_reweighting_strength != m['source_FR'] or model.field.feature_reweighting_strength != m['source_FR']:
        raise ValueError('Loaded source FR switch state must already equal0.3')
    model.field._get_discrete_feature_weights(torch.arange(model.field.num_levels, device=model.device, dtype=torch.float32).unsqueeze(-1))
    if model.adaptive_sampler.get_sigma_fn(torch.empty(0, 3, device=model.device), torch.empty(0, 3, device=model.device)) is not None:
        raise ValueError('Unexpected native eval density/opacity pruning callback')
    before = late.tree_digest(pipe.state_dict())
    lut = late.tree_digest((model.field._feature_weight_lut, model.field._feature_weight_lut_strength))
    admission_digest = late.tree_digest((model.semantic_admission, model.semantic_lookup.state_dict()))
    normalize = model.field._normalize_positions
    _restore_rng_state(payload['rng_state'])
    rng_before = late.tree_digest(_capture_rng_state())
    def assert_unchanged():
        checks = dict(pipeline=late.tree_digest(pipe.state_dict()) == before,
            LUT=late.tree_digest((model.field._feature_weight_lut,model.field._feature_weight_lut_strength)) == lut,
            RNG=late.tree_digest(_capture_rng_state()) == rng_before,
            admission=late.tree_digest((model.semantic_admission,model.semantic_lookup.state_dict())) == admission_digest,
            normalization_method=model.field._normalize_positions is normalize,
            FR=model.config.feature_reweighting_strength == m['source_FR'] and model.field.feature_reweighting_strength == m['source_FR'],
            seven_serialized_states={k:late.tree_digest(payload[k]) for k in keys} == serialized,
            source_checkpoint=sha(cp) == m['source_archive_sha256']['checkpoint.ckpt'])
        if not all(checks.values()): raise ValueError('Source changed: '+str(checks))
        source_runner.project_import_bindings(binding_contract,source_plan.parent)
        return checks
    write(output/'source_initial.json',dict(source_plan_sha256=sha(source_plan),source_release_sha256=sha(source_release),pipeline_digest=before,
        source_step=step,source_checkpoint_sha256=sha(cp),serialized_states=serialized,LUT_sha256=lut,RNG_sha256=rng_before,world_to_box=m['world_to_box']))
    assert_unchanged()
    FHD_proof = source_runner.prove_source_FHD(pipe,m['original_source_FHD'],output,model,torch)
    assert_unchanged()
    return dict(model=model,pipeline=pipe,manifest=m,checkpoint=cp,pipeline_digest=before,
        assert_unchanged=assert_unchanged,source_runner=source_runner,FHD_proof=FHD_proof,guard_done=guard_done)
