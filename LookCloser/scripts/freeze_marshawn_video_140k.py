"""Clone the reviewed video worker with an exact 140000-iteration budget.

Only termination, budget provenance and display configuration change. The
parent campaign, model, optimizer schedule and per-iteration math stay intact.
"""
from pathlib import Path
import ast, hashlib, json, shutil, time

PARENT=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107')
DEST=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2)+'\n')

BUDGET='''"""Exact attempted-iteration budget, shared by production and boundary tests."""
LIMIT = 140000

def validate_limit(spec):
    value = spec.get('hard_max_iterations')
    if type(value) is not int or value != LIMIT:
        raise ValueError('Production recipe requires exactly140000 iterations')
    return value

def check_step(step, limit=LIMIT):
    if type(step) is not int or not 0 <= step < limit:
        raise ValueError('Attempt outside the hard iteration budget: '+str(step))

def completed(step, limit=LIMIT):
    return step + 1 >= limit
'''

def freeze(parent=PARENT,dest=DEST):
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    old_seal=json.loads((parent/'code_seal.json').read_text())
    for name,digest in old_seal['files_sha256'].items():
        if sha(parent/'code'/name)!=digest:raise ValueError('Parent changed: '+name)
    shutil.copytree(parent/'code',dest/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    scripts=dest/'code/LookCloser/scripts'
    target=scripts/'train_marshawn_semantic_admission.py'
    original=target.read_text();modified=original;edits=[]
    def replace(before,after):
        nonlocal modified
        if modified.count(before)!=1:raise ValueError('Non-unique cap edit: '+before[:80])
        modified=modified.replace(before,after);edits.append((before,after))
    replace('The manual loop has no iteration ceiling and','The production loop has exactly140000 attempted iterations and')
    replace('from marshawn_tuning_metrics import HISTORICAL_EVAL_STEPS, PlateauTracker',
            'from marshawn_tuning_metrics import HISTORICAL_EVAL_STEPS, PlateauTracker\nfrom marshawn_iteration_budget import validate_limit, check_step, completed')
    replace("'technical_memory_cap_gib'}","'technical_memory_cap_gib', 'hard_max_iterations'}")
    replace("    for path in ('data', 'output'):","    validate_limit(spec)\n    for path in ('data', 'output'):")
    replace('    # Used by framework display/config only; our while-loop has NO step ceiling.\n    cfg.max_num_iterations = sys.maxsize',
            "    # The manual-loop guard is authoritative; framework display agrees.\n    cfg.max_num_iterations = validate_limit(spec)")
    replace("    chosen = choose_checkpoint(state['evaluations'])\n", "    limit = validate_limit(spec)\n    if step + 1 > limit or (smoke_steps is None and step + 1 != limit):\n        raise ValueError('Invalid terminal iteration count')\n    chosen = choose_checkpoint(state['evaluations'])\n")
    replace("else 'plateau',\n                technical_smoke=smoke_steps is not None, final_step=step, plateau=state['plateau'],", "else 'fixed_iteration_budget_complete',\n                technical_smoke=smoke_steps is not None, final_step=step, completed_iterations=step+1, hard_max_iterations=limit, plateau=state['plateau'],")
    replace("technical_smoke=smoke_steps is not None, final_step=step,\n                plateau=state['plateau'],", "technical_smoke=smoke_steps is not None, final_step=step, completed_iterations=step+1, hard_max_iterations=limit,\n                plateau=state['plateau'],")
    replace("    spec = validate_spec(json.loads(spec_path.read_text()))", "    spec = validate_spec(json.loads(spec_path.read_text()))\n    hard_limit = validate_limit(spec)")
    replace("    tracker = PlateauTracker.from_dict(state['plateau'])\n    if tracker.stopped or (resume is not None and smoke_steps is not None and latest['step'] + 1 >= smoke_steps):", "    tracker = PlateauTracker.from_dict(state['plateau'])\n    if resume is not None and latest['step'] + 1 > hard_limit:\n        raise ValueError('Cannot resume an over-budget checkpoint')\n    if resume is not None and (completed(latest['step'], hard_limit) or (smoke_steps is not None and latest['step'] + 1 >= smoke_steps)):")
    replace('                actual_iteration_limit=None, source_sha256=', '                actual_iteration_limit=hard_limit, source_sha256=')
    replace('    while True:\n        trainer.step = step', '    while True:\n        check_step(step, hard_limit)\n        trainer.step = step')
    replace('        if gate or smoke_done:\n            result = evaluate', '        hard_done = completed(step, hard_limit)\n        if gate or smoke_done or hard_done:\n            result = evaluate')
    replace('            if not smoke_done:\n                tracker.observe', '            if gate and not smoke_done and not tracker.stopped:\n                tracker.observe')
    replace('        terminal = tracker.stopped or smoke_done', '        terminal = hard_done or smoke_done')
    restored=modified
    for before,after in reversed(edits):
        if restored.count(after)!=1:raise ValueError('Cap reversal ambiguous')
        restored=restored.replace(after,before)
    if ast.dump(ast.parse(restored))!=ast.dump(ast.parse(original)):raise ValueError('Training parity reversal failed')
    target.write_text(modified);(scripts/'marshawn_iteration_budget.py').write_text(BUDGET)
    adapter=scripts/'marshawn_video_train.py';old_adapter=adapter.read_text();new_adapter=old_adapter
    before="    receipt = dict(schema=1, frame=frame, status='complete', technical=technical,\n        final_step=result['final_step'], selectors=exported,"
    after="    if not technical and (result.get('completed_iterations') != 140000 or result.get('hard_max_iterations') != 140000):\n        raise ValueError('Production completion must contain exactly140000 iterations')\n    receipt = dict(schema=1, frame=frame, status='complete', technical=technical,\n        completed_iterations=result['completed_iterations'], hard_max_iterations=140000,\n        final_step=result['final_step'], selectors=exported,"
    assert new_adapter.count(before)==1;new_adapter=new_adapter.replace(before,after)
    load_before="    root = Path(frame_root).resolve(); receipt = read(root / 'frame_completion.json')\n"
    load_after=load_before+"    if receipt.get('completed_iterations') != 140000 or receipt.get('hard_max_iterations') != 140000:\n        raise ValueError('Renderer rejects non140k production revisions')\n    if any(row['step'] + 1 > 140000 for row in receipt['selectors'].values()):\n        raise ValueError('Renderer rejects over-budget weights')\n"
    assert new_adapter.count(load_before)==1;new_adapter=new_adapter.replace(load_before,load_after)
    assert new_adapter.replace(load_after,load_before).replace(after,before)==old_adapter
    adapter.write_text(new_adapter)
    recipe=json.loads((parent/'recipe.json').read_text());recipe['hard_max_iterations']=140000;write(dest/'recipe.json',recipe)
    for name in ['common_box.json','source_inventory.json','video_plan.json','pipeline.json']:
        shutil.copy2(parent/name,dest/name)
    changed=['LookCloser/scripts/train_marshawn_semantic_admission.py','LookCloser/scripts/marshawn_video_train.py']
    for name in old_seal['files_sha256']:
        if name not in changed and sha(dest/'code'/name)!=old_seal['files_sha256'][name]:raise ValueError(name)
    proof=dict(schema=1,parent_campaign=str(parent),parent_code_seal_sha256=sha(parent/'code_seal.json'),
        training_ast_identical_after_exact_enumerated_termination_reversal=True,unchanged_parent_files=len(old_seal['files_sha256'])-len(changed),
        modified_parent_files=changed,added_budget_module='LookCloser/scripts/marshawn_iteration_budget.py',
        limit=140000,first_step=0,last_step=139999,early_plateau_is_diagnostic_only=True,
        optimizer_schedule_unchanged=True,model_field_sampler_occupancy_math_unchanged=True,
        edits=[dict(before=a,after=b)for a,b in edits],created=time.time())
    write(dest/'factory_proof.json',proof)
    files={str(p.relative_to(dest/'code')):sha(p)for p in sorted((dest/'code').rglob('*'))if p.is_file()and'__pycache__'not in p.parts}
    seal=dict(schema=1,files_sha256=files,recipe_sha256=sha(dest/'recipe.json'),factory_proof_sha256=sha(dest/'factory_proof.json'))
    write(dest/'code_seal.json',seal)
    release=json.loads((parent/'release.json').read_text());release.update(production_enabled=False,
        code_seal_sha256=sha(dest/'code_seal.json'),factory_proof_sha256=sha(dest/'factory_proof.json'),recipe_sha256=sha(dest/'recipe.json'),
        hard_max_iterations=140000,completed_iterations_required=140000,source_parent_campaign=str(parent),
        source_parent_release_sha256=sha(parent/'release.json'),budget_qualification_pending=True)
    write(dest/'release.json',release)
    return dict(root=str(dest),files=len(files),code_seal_sha256=sha(dest/'code_seal.json'),pipeline_sha256=sha(dest/'pipeline.json'))

if __name__=='__main__':print(json.dumps(freeze(),indent=2))
