"""Build an isolated, audited video adaptation of the selected frozen winner.

No scientific function/config is taken from the dirty working tree. The only
adaptations are ALL66 input validation, per-frame admission verification,
release/provenance checks, and retention of explicitly named render selectors.
"""
from pathlib import Path
import argparse, ast, hashlib, json, shutil
from concurrent.futures import ThreadPoolExecutor
from marshawn_video_train import sha, atomic

BASE = Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_quarter_distortion_candidate_20261107')


def function(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)


def replace_function(source,name,replacement):
    n=function(source,name); lines=source.splitlines(keepends=True)
    return ''.join(lines[:n.lineno-1])+replacement.rstrip()+'\n'+''.join(lines[n.end_lineno:])


def freeze(root, base=BASE):
    root=Path(root).resolve(); base=Path(base).resolve(); output=root/'code'
    if (root/'code_seal.json').exists():
        from marshawn_video_train import verify_code
        return verify_code(root)
    inventory=json.loads((base/'source_manifest.json').read_text())['files_sha256']
    output.mkdir(parents=True,exist_ok=True)
    def copy_verified(item):
        relative,digest=item;source=base/'code'/relative;dest=output/relative
        if sha(source)!=digest:raise ValueError('Canonical winner source changed: '+relative)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        if sha(dest)!=digest:raise ValueError('Winner source changed while copying: '+relative)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(copy_verified,inventory.items()))
    scripts=output/'LookCloser/scripts'; original=(scripts/'train_marshawn_semantic_admission.py').read_text()
    adapted=original.replace("from run_marshawn_semantic_release import verify", "from marshawn_video_release import verify")
    adapted=adapted.replace("len(pipe.datamanager.train_dataset) != 63", "len(pipe.datamanager.train_dataset) != 66")
    adapted=adapted.replace("len(pipe.datamanager.eval_dataset) != 3", "len(pipe.datamanager.eval_dataset) != len(ready['validation'])")
    needle="    trainer.setup()\n    pipe, model = trainer.pipeline, trainer.pipeline.model"
    if adapted.count(needle)!=1:raise ValueError('Native setup audit insertion changed')
    adapted=adapted.replace(needle,"    trainer.setup()\n    from marshawn_video_train import cold_setup_audit\n    cold_setup_audit(trainer, output, resume)\n    pipe, model = trainer.pipeline, trainer.pipeline.model")
    validator='''def validate_data(data):
    import numpy as np
    from marshawn_video_metrics import actor_visibility
    from prepare_marshawn_tuning import verify_inventory
    ready = verify_inventory(data)
    if ready['train'] != 66 or not ready['validation'] or len(ready['records']) != 66:
        raise ValueError('Expected ALL66 training plus three in-sample monitors')
    if set(ready['validation']) != {'DC','HC','KA'}:
        review=ready.get('monitor_review',{})
        if review.get('reviewed') is not True or not review.get('evidence'):
            raise ValueError('Alternative TRAIN monitors require explicit reviewed evidence')
    transforms = json.loads((data / 'transforms.json').read_text())
    frames = {r['file_path']:r for r in transforms['frames']}
    train = transforms['train_filenames']
    monitors = transforms['val_filenames']
    if len(train)!=66 or len(set(train))!=66 or set(train)!=set(frames) or not set(monitors)<=set(train):
        raise ValueError('All physical cameras must train; monitors explicitly overlap')
    if set(frames[p]['physical_camera'] for p in monitors) != set(ready['validation']):
        raise ValueError('Wrong in-sample monitors')
    if sha(data / 'transforms.json') != ready['transforms_sha256']:raise ValueError('Transforms changed')
    for row in ready['records']:
        if sha(data / row['image']) != row['rgb_sha256']:raise ValueError('RGB changed')
        semantic = data / 'semantics' / (row['camera']+'.npz')
        if sha(semantic) != row['semantic_sha256']:raise ValueError('Semantic mask changed')
        with np.load(semantic,allow_pickle=False) as src:
            for actor in ('black','grey'):
                if src[actor].dtype != np.bool_:raise ValueError('Invalid actor mask dtype')
                actor_visibility(row,actor,src[actor])
    return ready
'''
    adapted=replace_function(adapted,'validate_data',validator)
    eval_visibility="""                    mask = source[actor]
                    from marshawn_video_metrics import actor_visibility
                    if actor_visibility(row, actor, mask) == 'absent':
                        continue
"""
    old_mask="                    mask = source[actor]\n"
    if adapted.count(old_mask)!=1:raise ValueError('Eval visibility insertion changed')
    adapted=adapted.replace(old_mask,eval_visibility)
    old_aggregate="        result = dict(step=step, **aggregate_actor_metrics(rows), opacity_diagnostics=holes)"
    new_aggregate="        from marshawn_video_metrics import aggregate_visible\n        result = dict(step=step, **aggregate_visible(rows, ready), opacity_diagnostics=holes)"
    if adapted.count(old_aggregate)!=1:raise ValueError('Eval aggregate insertion changed')
    adapted=adapted.replace(old_aggregate,new_aggregate)
    # Scientific training is unchanged; protect additional output-only winners.
    oldret='    protected = protected_checkpoints(evaluations, step)\n'
    newret=oldret+"    if evaluations:\n        from marshawn_video_train import selectors\n        protected.update(r['step'] for r in selectors(evaluations).values())\n"
    if adapted.count(oldret)!=1:raise ValueError('Retention insertion changed')
    adapted=adapted.replace(oldret,newret)
    (scripts/'train_marshawn_semantic_admission.py').write_text(adapted)
    semantic=(scripts/'marshawn_semantic_admission.py').read_text()
    delegated='''def load_support(manifest_path,expected_sha,data_dir,bounds,resolution,identity=False):
    from marshawn_video_support import load_support as frame_local_support
    return frame_local_support(manifest_path,expected_sha,data_dir,bounds,resolution,identity)
'''
    (scripts/'marshawn_semantic_admission.py').write_text(replace_function(semantic,'load_support',delegated))
    # Verify function AST equality, then reverse only enumerated operational edits
    # in train(): this also covers callback placement, RNG, losses and schedule.
    for name in ('build_config','choose_checkpoint','protected_checkpoints','finish','__getattr__','validate_spec'):
        if ast.dump(function(original,name),include_attributes=False)!=ast.dump(function(adapted,name),include_attributes=False):
            raise ValueError('Scientific source changed: '+name)
    restored=adapted.replace("from marshawn_video_release import verify","from run_marshawn_semantic_release import verify")
    restored=restored.replace("len(pipe.datamanager.train_dataset) != 66","len(pipe.datamanager.train_dataset) != 63")
    restored=restored.replace("len(pipe.datamanager.eval_dataset) != len(ready['validation'])","len(pipe.datamanager.eval_dataset) != 3")
    restored=restored.replace("    from marshawn_video_train import cold_setup_audit\n    cold_setup_audit(trainer, output, resume)\n",'')
    if ast.dump(function(restored,'train'),include_attributes=False)!=ast.dump(function(original,'train'),include_attributes=False):
        raise ValueError('Unexpected training-loop alteration')
    eval_restored=adapted.replace(eval_visibility,old_mask).replace(new_aggregate,old_aggregate)
    if ast.dump(function(eval_restored,'evaluate'),include_attributes=False)!=ast.dump(function(original,'evaluate'),include_attributes=False):
        raise ValueError('Unexpected present-actor metric math alteration')
    own=Path(__file__).resolve().parent
    files=[p.name for p in sorted(own.glob('marshawn_video_*.py'))]+['marshawn_tuning_controller.py']
    # These are input/render helpers introduced after the canonical model was
    # frozen. Existing scientific/model files remain the canonical copies.
    for name in ['audit_marshawn_support_rays_cpu.py','build_marshawn_semantic_resolution.py','marshawn_gpu_isolation.py']:
        if (own/name).exists() and ('LookCloser/scripts/'+name) not in inventory:files.append(name)
    for name in files:shutil.copy2(own/name,scripts/name)
    # Resolve the actual local import closure, including function-local imports.
    # Canonical dependencies are parsed from the copied canonical source; a
    # working-tree dependency is added only if the canonical snapshot lacks it.
    pending=list(files);visited=set();added=[]
    while pending:
        name=pending.pop()
        if name in visited:continue
        visited.add(name);text=(scripts/name).read_text()
        imported=set()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node,ast.Import):imported.update(a.name.split('.')[0]+'.py' for a in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module:imported.add(node.module.split('.')[0]+'.py')
        for dependency in imported:
            if not (own/dependency).is_file():continue
            if not (scripts/dependency).exists():
                shutil.copy2(own/dependency,scripts/dependency);added.append(dependency)
            pending.append(dependency)
    # The CPU ray audit compiles this source via Path(__file__).with_name(...).
    # Python import closure alone does not cover its required native asset.
    added_assets=[]
    if 'audit_marshawn_support_rays_cpu.py' in visited:
        asset='marshawn_semantic_ray_support.cpp'
        if not (own/asset).is_file():raise FileNotFoundError('Missing CPU audit asset: '+asset)
        if not (scripts/asset).exists():
            shutil.copy2(own/asset,scripts/asset);added_assets.append(asset)
    recipe=json.loads((base/'spec.json').read_text())
    for key in ('data','output','semantic_admission'):recipe.pop(key)
    atomic(root/'recipe.json',recipe)
    proof=dict(schema=1,canonical_source_manifest_sha256=sha(base/'source_manifest.json'),
        canonical_source=str(base),unchanged_scientific_functions=['build_config','choose_checkpoint',
        'protected_checkpoints','finish','__getattr__','validate_spec'],
        train_ast_equal_after_enumerated_operational_reversal=True,
        evaluate_ast_equal_after_explicit_known_absence_skip_and_aggregation_reversal=True,
        local_import_closure=sorted(visited),additional_local_dependencies=sorted(added),
        additional_local_assets=sorted(added_assets),
        changes=['63→66 training dataset guard','ALL66 in-sample monitor validation',
        'frame-local data-derived admission loader','campaign release verification',
        'read-only cold setup audit','retain raw actor LPIPS and PSNR07 output winners',
        'skip only explicitly reviewed known-absent monitor pairs; equal actors over visible views'],
        no_learning_math_changes=True)
    atomic(root/'factory_proof.json',proof)
    paths=[p for p in sorted(output.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    with ThreadPoolExecutor(max_workers=8) as pool:
        files_sha256=dict(pool.map(lambda p:(str(p.relative_to(output)),sha(p)),paths))
    seal=dict(schema=1,files_sha256=files_sha256,
              factory_proof_sha256=sha(root/'factory_proof.json'),recipe_sha256=sha(root/'recipe.json'))
    atomic(root/'code_seal.json',seal)
    return seal


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,default=BASE)
    a=p.parse_args();print(json.dumps({'sealed_files':len(freeze(a.root,a.base)['files_sha256'])}))
