"""Stage copied historical validation adapters; never edits live/native sources."""
import argparse,ast,difflib,hashlib,json,shutil
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replace_once(text,before,after):
    if text.count(before)!=1:raise ValueError('Unexpected frozen source structure: '+before[:80])
    return text.replace(before,after,1)


def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);code=out/'code';code.mkdir();sources=[]
    for name in ['marshawn_final_pipeline.py','marshawn_los_objective_frozen.py','marshawn_surface_band_objective.py','run_final_pair.py','staged_native_hash.py','marshawn_mvs_distribution_candidate.py']:
        original=a.native_code/name;text=original.read_text();updated=text
        if name=='marshawn_final_pipeline.py':
            updated=replace_once(updated,'from staged_native_hash import file_sha','from staged_native_hash import file_sha\nfrom marshawn_native_resolution_binding import validate_native_resolution_binding')
            updated=replace_once(updated,"        data=Path(cfg.datamanager.dataparser.data)","        data=Path(cfg.datamanager.dataparser.data)\n        geometry_transforms=Path(cfg.final_geometry_transforms)\n        resolution_binding=validate_native_resolution_binding(data/'transforms.json',geometry_transforms,\n            cfg.final_geometry_transforms_sha256,cfg.final_rgb_train_scale)")
            old="current_bounds" # Keep all original computation lines except constructor binding arguments.
            before="bounds,data/'transforms.json',\n            cfg.final_geometry_preparation"
            if updated.count(before)!=2:raise ValueError('Expected exactly SfM free and band geometry bindings')
            updated=updated.replace(before,"bounds,geometry_transforms,\n            cfg.final_geometry_preparation")
            updated=replace_once(updated,"        variant=cfg.final_geometry_variant","        object.__setattr__(self,'_native_resolution_binding',resolution_binding)\n        variant=cfg.final_geometry_variant")
        if name=='marshawn_mvs_distribution_candidate.py':
            updated=replace_once(updated,"        data=Path(pipeline.config.datamanager.dataparser.data)","        data=Path(pipeline.config.datamanager.dataparser.data)\n        geometry_transforms=Path(pipeline.config.final_geometry_transforms)")
            updated=replace_once(updated,"sha(data/'transforms.json')!=m['transforms_sha256']","sha(geometry_transforms)!=m['transforms_sha256']")
            updated=replace_once(updated,"tf=json.loads((data/'transforms.json').read_text())","tf=json.loads(geometry_transforms.read_text())")
        (code/name).write_text(updated)
        # Every training/sampling/grid method stays byte-equivalent at the AST level.
        def methods(src):
            tree=ast.parse(src);return {n.name+':'+m.name:ast.dump(m,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.ClassDef) for m in n.body if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef)) and m.name!='__init__'}
        assert methods(text)==methods(updated),name
        sources.append(dict(name=name,original_path=str(original),original_sha256=sha(original),staged_sha256=sha(code/name),nonconstructor_methods_AST_equal=True))
        if text!=updated:(out/(name+'.diff')).write_text(''.join(difflib.unified_diff(text.splitlines(True),updated.splitlines(True),fromfile=str(original),tofile=str(code/name))))
    helper=Path(__file__).with_name('marshawn_native_resolution_binding.py');shutil.copy2(helper,code/helper.name)
    result=dict(kind='ISOLATED_NATIVE_RESOLUTION_VALIDATION_ADAPTER',technical_GPU_gate_authorized=False,production_launch_authorized=False,source_files=sources,validator_sha256=sha(code/helper.name),stager_sha256=sha(__file__),native_source_directory_untouched=True,geometry_computations_unchanged=True,config_required={'final_geometry_transforms':'original immutable HD native transforms path','final_geometry_transforms_sha256':'exact original hash','final_rgb_train_scale':'1 identity or2 treatment','final_color_gate':'separate reviewed RGB gate binding the staged dataset','final_source_hashes':'hash every staged legacy dependency plus new validator'},limitations=['CPU AST equality excludes constructors; separate source/array/sampling tests and actual-native GPU parity are required.','No checkpoint, image cache, model or optimizer is constructed by staging.'])
    (out/'stage_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-code',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
