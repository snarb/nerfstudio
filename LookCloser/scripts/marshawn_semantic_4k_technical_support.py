"""Strict technical-only 4K source adapter and observational runtime proof hooks."""
import copy
import hashlib
import json
from pathlib import Path
import time

CONTRACT = None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    import os
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def verify_files(root,files):
    for name,digest in files.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe source inventory path')
        if sha(Path(root)/name)!=digest:raise ValueError('Source mutation: '+name)

def validate_contract(c):
    if c['scope']!='TECHNICAL_ONLY_4K_R0_W64' or c['smoke_steps']!=1057 or c['pause_after']!=1024:
        raise ValueError('Only cold0→1024 +32 resume diagnostic allowed')
    if c['production_authorized'] is not False or c['READY_creation_allowed'] is not False:
        raise ValueError('No production/READY authorization')
    if Path(c['geometry_data']).resolve()==Path(c['data']).resolve():raise ValueError('Distinct HD geometry and4K data required')
    if c['mapped_support_sha256']!='d60baa9186a63f3769204b93cb46dcfd2ea16e3d41a72ca788934fe19bd2659f':raise ValueError('Only original W64pad1 geometry')
    return c

def initialize(contract_path,spec_path,smoke_steps):
    global CONTRACT
    c=validate_contract(read(contract_path));release=read(c['release'])
    if Path(c['contract_path']).resolve()!=Path(contract_path).resolve():raise ValueError('Wrong contract identity')
    if release.get('technical_authorized') is not True or release.get('production_authorized') is not False:
        raise ValueError('Explicit technical-only release required')
    if release.get('contract_sha256')!=sha(contract_path) or release.get('spec_sha256')!=sha(spec_path):raise ValueError('Unbound technical release')
    if smoke_steps!=1057 or sha(spec_path)!=c['spec_sha256']:raise ValueError('Wrong smoke/spec')
    for path,digest in c['artifacts'].items():
        if sha(path)!=digest:raise ValueError('Bound artifact changed: '+path)
    verify_files(c['code_root'],read(c['code_inventory']))
    spec=read(spec_path)
    if spec['data']!=c['data'] or spec['output']!=c['output'] or spec.get('technical_pause_after')!=1024 or spec.get('technical_start_step')!=0:
        raise ValueError('Unreviewed data/output/technical phase')
    expected=dict(objective='actor',sampling='body80',width=64,radius=0,distortion_final=.01,frequency_projection_mode='legacy',seed=42,density_activation='trunc_exp',correct_sh_directions=True)
    for key,value in expected.items():
        if type(spec.get(key)) is not type(value) or spec[key]!=value:raise ValueError('Wrong unchanged recipe '+key)
    CONTRACT=c


def bind_config(cfg):
    if CONTRACT is None:raise ValueError('Technical contract not initialized')
    cfg.pipeline.model.semantic_data=CONTRACT['geometry_data']
    return cfg


def validate_data(data):
    from marshawn_semantic_4k_derivation import inspect,validate_map_audit
    c=CONTRACT
    if c is None or Path(data).resolve()!=Path(c['data']).resolve():raise ValueError('Wrong source generation')
    verify_files(data,read(c['source_inventory']))
    verify_files(c['candidate_root'],read(c['map_inventory']))
    expected=read(c['derivation_receipt']);actual=inspect(c['geometry_data'],data,c['source_inventory'],c['semantic_manifest'])
    for k in ('source_inventory_sha256','geometry_source_hashes','semantic_manifest_sha256','support','splits','eval9bytes_and_crops_identical','native_RAW_widths_preserved'):
        if actual[k]!=expected[k]:raise ValueError('Derivation changed: '+k)
    cpu=read(c['cpu_proof'])
    if cpu['status']!='CPU_CACHE_AND_SAMPLING_PASS' or cpu['map_inventory_sha256']!=sha(c['map_inventory']) or cpu['source_inventory_sha256']!=sha(c['source_inventory']) or cpu['aggregate_sha256']!=sha(c['aggregate']):raise ValueError('CPU cache proof/source mismatch')
    if actual['support']['mapped_bool_sha256']!=c['mapped_support_sha256']:raise ValueError('HD geometry support changed')
    validate_map_audit(read(c['aggregate']),actual['splits']['train'])
    if (Path(data)/'ready.json').exists():raise ValueError('Technical generation must not create READY')
    return read(Path(data)/'source_metadata.json')


def tree_digest(value):
    import torch
    import numpy as np
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str(('tensor',str(x.dtype),tuple(x.shape))).encode());h.update(x.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str(('numpy',x.dtype.str,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for key in sorted(x,key=lambda v:(type(v).__name__,repr(v))):visit(key);visit(x[key])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode())
            for v in x:visit(v)
        else:h.update((type(x).__name__+':'+repr(x)).encode())
    visit(value);return h.hexdigest()

def adam_steps(trainer):
    return {name:sorted({float(v['step'])for v in opt.state_dict()['state'].values()if 'step'in v})for name,opt in trainer.optimizers.optimizers.items()}

def begin(trainer,sampler,state,output,resume):
    import torch
    import numpy as np
    from nerfstudio.engine.trainer import _capture_rng_state
    from marshawn_semantic_occupancy import validate_policy
    model=trainer.pipeline.model;pipe=trainer.pipeline;output=Path(output)
    actual=dict(pipeline=pipe.state_dict(),optimizers={k:v.state_dict()for k,v in trainer.optimizers.optimizers.items()},schedulers={k:v.state_dict()for k,v in trainer.optimizers.schedulers.items()},scalers=trainer.grad_scaler.state_dict(),rng_state=_capture_rng_state(),tuning_runtime=copy.deepcopy(state),actual_occupancy_capture=getattr(model,'_tuning_occupancy_capture',None))
    support_sha=hashlib.sha256(model.semantic_admission.detach().cpu().numpy().tobytes()).hexdigest()
    if support_sha!=CONTRACT['mapped_support_sha256']:raise ValueError('Nonpersistent semantic support reload mismatch')
    assert model.__class__.__name__=='SemanticAdmissionModel' and hasattr(model,'semantic_lookup')
    ids=pipe.datamanager.train_image_dataloader.cached_collated_batch['image_idx'].tolist()
    import random
    if ids!=random.Random(42).sample(range(63),63):raise ValueError('Actual complete-cache order mismatch')
    proof=dict(resume=resume is not None,support_sha256=support_sha,cache_ids=ids,fullstate_sha256={k:tree_digest(v)for k,v in actual.items()},Adam=adam_steps(trainer),RNG_preserved=True)
    if resume is not None:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if saved['step']!=1024 or trainer._start_step!=1025:raise ValueError('Exact1024→1025 resume required')
        for key,digest in proof['fullstate_sha256'].items():
            if tree_digest(saved[key])!=digest:raise ValueError('Full-state resume mismatch: '+key)
        if not all(v and min(v)>0 for v in proof['Adam'].values()):raise ValueError('Nonempty Adam required')
        cap=actual['actual_occupancy_capture'];npy=lambda t:t.detach().cpu().numpy()
        validate_policy(npy(cap['core']),npy(cap['active']),cap['radius'],cap['binary_warmup'],npy(model.semantic_admission)[None])
        if not np.array_equal(npy(model.semantic_learned_core),npy(cap['core'])) or not np.array_equal(npy(model.occupancy_grid.binaries),npy(cap['active'])):raise ValueError('Schema2 core/active mismatch')
        proof['schema2_actual_core_active_verified']=True
        proof['EMA_pipeline_sha256']=tree_digest(model.occupancy_grid.occs)
        del saved
    atomic(output/('resume_fullstate_proof.json'if resume else'cold_setup_proof.json'),proof)
    if tree_digest(_capture_rng_state())!=proof['fullstate_sha256']['rng_state']:raise ValueError('Proof consumed RNG')
    del actual
    telemetry=[];phase='resume'if resume else'cold';original_iteration=trainer.train_iteration
    def observed_iteration(step):
        before=adam_steps(trainer);result=original_iteration(step);after=adam_steps(trainer)
        updated=any(v and (not before[k]or min(v)>min(before[k]))for k,v in after.items())
        checked=step<64 or step>=1024
        finite=None
        if checked:
            grads=[p.grad for p in model.parameters()if p.grad is not None]
            finite=bool(grads)and all(bool(torch.isfinite(g).all())for g in grads)
            if updated and not finite:raise FloatingPointError('Actual Adam update has nonfinite combined gradients')
        telemetry.append(dict(step=step,Adam=after,Adam_updated=updated,combined_gradients_checked=checked,combined_gradients_finite=finite,scale=float(trainer.grad_scaler.get_scale())))
        if checked or step%128==0:atomic(output/(phase+'_Adam_telemetry.json'),telemetry)
        return result
    trainer.train_iteration=observed_iteration
    grid=model.freq_grid;original_update=grid.update_step
    def observed_update(step,positions,rendered_depth,focals,patch_f2d):
        if step!=1024:raise ValueError('Unexpected frequency update in bounded technical phase')
        before=grid.grid.detach().clone();rng=tree_digest(_capture_rng_state())
        result=original_update(step,positions,rendered_depth,focals,patch_f2d)
        after=grid.grid.detach();indices=grid.grid_to_indices(positions);flat=indices[:,0]*grid.resolution**2+indices[:,1]*grid.resolution+indices[:,2];unique=torch.unique(flat)
        finite=torch.isfinite(positions).all(-1)&torch.isfinite(rendered_depth.reshape(-1));inside=((positions>=grid.aabb_min_buf)&(positions<=grid.aabb_max_buf)).all(-1)&finite;admitted=inside&model.semantic_lookup(positions)
        admitted_cells=torch.unique(flat[admitted]);levels=after.reshape(-1)[unique]
        f3d=patch_f2d*(focals/(rendered_depth+1e-6))
        if not torch.isfinite(after).all():raise FloatingPointError('Nonfinite native frequency grid')
        report=dict(step=step,projection_mode=pipe.config.frequency_projection_mode,pre_sha256=tree_digest(before),post_sha256=tree_digest(after),pre_histogram=torch.bincount(before.long().flatten(),minlength=16).tolist(),post_histogram=torch.bincount(after.long().flatten(),minlength=16).tolist(),samples=len(positions),finite_samples=int(finite.sum()),inside_AABB_samples=int(inside.sum()),admitted_samples=int(admitted.sum()),unique_touched_cells=len(unique),touched_maxlevel_cells=int((levels==15).sum()),touched_maxlevel_fraction=float((levels==15).float().mean()),unique_admitted_touched_cells=len(admitted_cells),admitted_touched_maxlevel_cells=int((after.reshape(-1)[admitted_cells]==15).sum()),admitted_touched_maxlevel_fraction=float((after.reshape(-1)[admitted_cells]==15).float().mean())if len(admitted_cells)else None,changed_cells=int((before!=after).sum()),raw_f3d_over8192_fraction=float((f3d>8192).float().mean()),legacy_outside_positions_still_clamped=True,occupancy_grid_not_changed_by_observer=True,RNG_unchanged_by_grid_math_and_observer=rng==tree_digest(_capture_rng_state()))
        if not report['RNG_unchanged_by_grid_math_and_observer']:raise ValueError('Frequency observer changed RNG')
        atomic(output/'frequency_step1024.json',report);return result
    grid.update_step=observed_update
