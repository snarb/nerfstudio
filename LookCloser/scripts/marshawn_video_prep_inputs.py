"""Frame-local video preparation commands; publishes only validated stage receipts.

The first-frame prototype reuses immutable INPUTS, never model/occupancy state.
Source paths remain relative inside data so staging to another host is possible.
"""
from pathlib import Path
import argparse,json,os,shutil
from prepare_marshawn_tuning import sha,write,refresh_inventory,verify_inventory

ROOT=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107')
BASE=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106/data')
RAW0=Path('/fsx/oregon/gta_marshawn_2_1/cedar/tiff16/by_local')
RAW1=Path('/fsx/oregon/gta_marshawn_2_1_tiff16_f1901-2020_20261105/by_local')

def raw_path(frame,camera):
    if not 1660<=frame<=2020:raise ValueError('No qualified RAW after2020')
    return (RAW0 if frame<=1900 else RAW1)/camera/f'{camera}.{frame-1252:06d}.tif'

def prototype(root):
    old=verify_inventory(BASE);out=root/'frames/001660/data';out.mkdir(parents=True,exist_ok=True)
    if (out/'prepared_ready.json').exists():
        hybrid_flags(out);admission_inputs(out);return out
    tf=json.loads((BASE/'transforms.json').read_text());names={f['physical_camera'] for f in tf['frames']}
    if len(names)!=66:raise ValueError('Expected all66 physical cameras')
    tf['train_filenames']=[f['file_path'] for f in tf['frames']]
    tf['val_filenames']=[f['file_path'] for f in tf['frames'] if f['physical_camera'] in ['DC','HC','KA']]
    tf['test_filenames']=list(tf['val_filenames']);tf['monitoring_policy']='DC/HC/KA are in-sample training monitors'
    records=[]
    for r in old['records']:
        for relative in [r['image'],f"semantics/{r['camera']}.npz",next(f['mask_path'] for f in tf['frames'] if f['physical_camera']==r['camera'])]:
            dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.symlink_to((BASE/relative).resolve())
        row=dict(r,semantic=str(out/'semantics'/f"{r['camera']}.npz"));records.append(row)
        stem=Path(r['image']).stem
        for suffix in ['.pt','.json']:
            src=BASE/'lookcloser_frequencies'/(stem+suffix);dest=out/'lookcloser_frequencies'/src.name;dest.parent.mkdir(exist_ok=True)
            if src.exists() and not dest.exists():dest.symlink_to(src.resolve())
    write(out/'transforms.json',tf);shutil.copy2(BASE/'box.json',out/'box.json')
    ready={k:v for k,v in old.items() if k not in ['inventory','box_sha256']}
    ready.update(train=66,records=records,validation=['DC','HC','KA'],monitoring='in-sample; former eval images now train',transforms_sha256=sha(out/'transforms.json'),frequency_policy='ALL66 independent source-bound SSIM.95 maps; three newly added views pending',input_reuse=dict(frame=1660,base_ready_sha256=sha(BASE/'ready.json'),model_or_learned_grid_reused=False))
    write(out/'ready.json',ready)
    missing=[Path(r['image']).stem for r in records if not (out/'lookcloser_frequencies'/(Path(r['image']).stem+'.pt')).exists()]
    write(out/'prepared_ready.json',dict(frame=1660,stage='INPUTS_QUALIFIED_FREQUENCY_PENDING',train=66,monitoring='in-sample DC/HC/KA',source_ready_sha256=sha(BASE/'ready.json'),transforms_sha256=sha(out/'transforms.json'),new_frequency_maps_required=missing,cold_model_required=True))
    hybrid_flags(out);admission_inputs(out)
    return out

def hybrid_flags(data):
    import numpy as np
    root=BASE.parent.parent/'two_actor_isolation'
    confidence={r['camera']:r for r in json.loads((root/'confidence_native_hair_v2/manifest.json').read_text())['rows']}
    visual=root/'automatic_aabb/input_v2'
    visual_rows={r['camera']:r for r in json.loads((visual/'manifest.json').read_text())['cameras']}
    ready=json.loads((data/'ready.json').read_text());changed=False
    for row in ready['records']:
        path=data/'semantics'/f"{row['camera']}.npz"
        with np.load(path) as z:
            if 'unknown' in z and 'source_ineligible' in z:continue
            arrays={k:z[k].copy() for k in z.files}
        oldsha=sha(path);cam=row['camera'];r=confidence[cam]
        if sha(r['archive'])!=r['archive_sha256']:raise ValueError('Canonical confidence changed '+cam)
        with np.load(r['archive']) as z:veto=z['source_ineligible'].copy()
        with np.load(visual/visual_rows[cam]['masks']) as z:unknown=z['unknown'].copy()
        arrays.update(unknown=unknown,source_ineligible=veto)
        temp=path.with_name(path.name+'.materializing')
        with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
        temp.replace(path)
        with np.load(path) as check:
            for key,value in arrays.items():
                if not np.array_equal(check[key],value):raise ValueError('Materialized semantic array changed '+cam+' '+key)
        row.update(semantic_sha256=sha(path),source_semantic_base_sha256=oldsha,semantic_flag_adaptation='ExplicitqualifiedUNKNOWN+vetoarrays appended; allbaselineW32/W64/actor/bodyarrays unchanged')
        changed=True
    for row in ready['records']:
        row['visibility']={'black':'present','grey':'present'}
    if changed:
        ready.pop('inventory',None);ready.pop('box_sha256',None)
    write(data/'ready.json',ready)
    prepared=json.loads((data/'prepared_ready.json').read_text());prepared['hybrid_semantics_explicit']=True;prepared['semantic_records_sha256']=__import__('hashlib').sha256(json.dumps([(r['camera'],r['semantic_sha256']) for r in ready['records']]).encode()).hexdigest();write(data/'prepared_ready.json',prepared)

def admission_inputs(data):
    import numpy as np
    source=BASE.parent/'native_semantic_grid_20000/occlusion_v2/W64/archives';out=data/'admission_inputs';out.mkdir(exist_ok=True)
    ready=json.loads((data/'ready.json').read_text());records=[]
    for row in ready['records']:
        name=row['camera'];dest=out/f'{name}.npz';old=source/f'{name}.npz'
        if old.is_file():
            if not dest.exists():dest.symlink_to(old.resolve())
            provenance='exactreviewedframe1660W64imageprojectionlabels; no learnedgrid'
        else:
            if not dest.exists():
                with np.load(data/'semantics'/f'{name}.npz') as z:support=z['W64']>0;valid=z['valid'].copy()
                np.savez_compressed(dest,support=support,valid=valid,unknown=np.ones_like(valid))
            provenance='formerEVALnowTRAIN; W64positiveswithclearvisibilityUNKNOWN'
        records.append(dict(camera=name,archive=f'admission_inputs/{name}.npz',archive_sha256=sha(dest),provenance=provenance))
    write(out/'manifest.json',dict(frame=1660,train_camera_count=66,native_pixel_support_width=64,records=records,no_model_or_grid_loaded=True))
    prepared=json.loads((data/'prepared_ready.json').read_text());prepared['admission_inputs_manifest_sha256']=sha(out/'manifest.json');write(data/'prepared_ready.json',prepared)
    return out

def source_inventory(root):
    tf=json.loads((BASE/'transforms.json').read_text());cameras=sorted(f['physical_camera'] for f in tf['frames']);rows=[]
    for frame in range(1660,2020,5):
        missing=[c for c in cameras if not raw_path(frame,c).is_file()]
        rows.append(dict(frame=frame,time_seconds=(frame-1660)/120,missing_cameras=missing,source_root=str(RAW0 if frame<=1900 else RAW1),local_frame=frame-1252))
    result=dict(source_fps=120,output_fps=24,frames=rows,train_count=72,cameras=cameras,raw_available_all=all(not r['missing_cameras'] for r in rows),continuation_GT_available=False,continuation_requirement='2020..2255 at24fps; qualified TIFFs stop2020',future60fps='Use global frames1660..2018 step2; no naming or recipe change required',no_model_or_learned_grid_loaded=True)
    write(root/'source_inventory.json',result);return result

def seal(data):
    ready=refresh_inventory(data)
    write(data/'frequency_ready.json',dict(frame=ready['frame'],train=ready['train'],ready_sha256=sha(data/'ready.json'),inventory_files=len(ready['inventory']['files'])))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prototype','inventory','seal']);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--data',type=Path);a=p.parse_args()
    if a.command=='prototype':print(prototype(a.root))
    elif a.command=='inventory':print(json.dumps(source_inventory(a.root)))
    else:seal(a.data)
