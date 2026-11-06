#!/usr/bin/env python3
"""CPU-only dual-source derivation/config adapter; never creates READY or trains."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def verify_frames(old,new):
    if set(old)!=set(new):raise ValueError('Descriptor keys differ')
    for k in set(old)-{'frames'}:
        if old[k]!=new[k]:raise ValueError('Split/global descriptor changed: '+k)
    a={r['physical_camera']:r for r in old['frames']};b={r['physical_camera']:r for r in new['frames']}
    if len(a)!=66 or len(b)!=66 or len(old['frames'])!=66 or len(new['frames'])!=66 or set(a)!=set(b):raise ValueError('Unique66 cameras required')
    train=set(old['train_filenames']);eval_names={r['physical_camera']for r in old['frames']if r['file_path']not in train}
    if len(train)!=63 or eval_names!={'DC','HC','KA'}:raise ValueError('Frozen63/3 split required')
    for split in ('train_filenames','val_filenames','test_filenames'):
        ids=lambda m:[r['physical_camera']for r in m['frames']if r['file_path']in m[split]]
        if ids(old)!=ids(new):raise ValueError('Within-split camera order changed')
    for camera,row in a.items():
        expected=dict(row);scale=2 if row['file_path']in train else 1
        for k in ('w','h','fl_x','fl_y','cx','cy'):expected[k]*=scale
        if expected!=b[camera]:raise ValueError('Pose/calibration/path derivation differs: '+camera)
    return dict(train=[r['physical_camera']for r in old['frames']if r['file_path']in train],validation=sorted(eval_names),combined_frame_interleaving_may_differ=True)

def validate_output(output,*protected_roots):
    target=Path(output).resolve()
    if target.name.lower()=='ready.json'or any(target.is_relative_to(Path(root).resolve())for root in protected_roots):
        raise ValueError('Inspection cannot write READY or modify either source dataset')

def verify_inventory(root,inventory):
    root=Path(root)
    for name,digest in inventory.items():
        if Path(name).is_absolute()or '..'in Path(name).parts:raise ValueError('Unsafe relative source path')
        if sha(root/name)!=digest:raise ValueError('Source hash changed: '+name)

def validate_map_audit(aggregate,cameras):
    if aggregate.get('status')!='EXPERIMENTAL_FREQUENCY_CANDIDATE_COMPLETE':raise ValueError('All63 maps must finish actual audit first')
    rows=aggregate.get('cameras',[])
    if len(rows)!=63 or {r['camera']for r in rows}!=set(cameras):raise ValueError('Map coverage differs fromTRAIN63')
    if any(set(r.get('output_hashes',{}))!={'map','metadata','censor'}for r in rows):raise ValueError('Missing source map/metadata/censor bindings')
    return True

def inspect(geometry_data,source_data,inventory_path,semantic_manifest):
    import numpy as np
    from marshawn_semantic_admission import load_support
    geometry_data=Path(geometry_data).resolve();source_data=Path(source_data).resolve();inventory_path=Path(inventory_path).resolve();semantic_manifest=Path(semantic_manifest).resolve()
    if geometry_data==source_data:raise ValueError('Separate HD geometry and4K source required')
    old=read(geometry_data/'transforms.json');new=read(source_data/'transforms.json');split=verify_frames(old,new)
    original=read(geometry_data/'ready.json');derived=read(source_data/'source_metadata.json');complete=read(source_data/'complete.json');inventory=read(inventory_path)
    verify_inventory(source_data,inventory)
    if sha(source_data/'source_metadata.json')!=complete['source_metadata_sha256']or sha(source_data/'transforms.json')!=derived['transforms_sha256']:raise ValueError('4K source descriptor changed')
    for key in ('aabb','world_to_box','source_confidence_sha256'):
        if original[key]!=derived[key]:raise ValueError('Geometry/native confidence changed: '+key)
    before={str(geometry_data/n):sha(geometry_data/n)for n in ('ready.json','transforms.json','box.json')}
    source_records={r['camera']:r for r in derived['records']};old_records={r['camera']:r for r in original['records']}
    for frame in old['frames']:
        camera=frame['physical_camera'];row=source_records[camera]
        if sha(source_data/frame['file_path'])!=row['rgb_sha256']or sha(source_data/'semantics'/f'{camera}.npz')!=row['semantic_sha256']:raise ValueError('Unbound RGB/confidence '+camera)
        if camera in split['validation']:
            for rel in (frame['file_path'],frame['mask_path'],'semantics/'+camera+'.npz'):
                if sha(geometry_data/rel)!=sha(source_data/rel):raise ValueError('Original EVAL9 bytes changed')
                before[str(geometry_data/rel)]=sha(geometry_data/rel)
            if row['crops']!=old_records[camera]['crops']:raise ValueError('EVAL crop changed')
        else:
            replay=read(source_data/'replay'/f'{camera}.json');receipt=read(source_data/'receipts'/f'{camera}.json')
            if not replay['rgb_byte_exact']or not replay['valid_exact']or not receipt['FHD_semantics_exact']or receipt['native_widths']!=[32,64]or receipt['physical_kernels']!=[33,65,129]:raise ValueError('Native mask/photometry replay not bound')
    mask,support=load_support(semantic_manifest,sha(semantic_manifest),geometry_data,np.array(original['aabb'],dtype=np.float32),(128,128,128))
    for p,h in before.items():
        if sha(p)!=h:raise ValueError('HD source changed during check')
    return dict(kind='CPU_SOURCE_DERIVATION_ONLY',adapter_sha256=sha(__file__),time=time.time(),geometry_data=str(geometry_data),training_source_data=str(source_data),source_inventory=str(inventory_path),source_inventory_sha256=sha(inventory_path),source_files_verified=len(inventory),geometry_source_hashes=before,semantic_manifest=str(semantic_manifest),semantic_manifest_sha256=sha(semantic_manifest),support=support,splits=split,eval9bytes_and_crops_identical=True,native_RAW_widths_preserved=True,projection_mode_changed=False,bundle='4K RGB/native confidence +4K4000maps vsHD1000; not pure-resolution causality',READY_created=False,GPU_used=False,production_authorized=False,ancestor_selection_pending=True)

def bind_config(config,receipt_path,expected_receipt_sha):
    """Construct a proposed config only. Does not instantiate model/optimizer."""
    if sha(receipt_path)!=expected_receipt_sha:raise ValueError('Derivation receipt changed')
    proof=read(receipt_path)
    if proof['kind']!='CPU_SOURCE_DERIVATION_ONLY'or not proof['eval9bytes_and_crops_identical']:raise ValueError('Unqualified derivation proof')
    for p,h in proof['geometry_source_hashes'].items():
        if sha(p)!=h:raise ValueError('Original geometry source changed')
    if sha(proof['source_inventory'])!=proof['source_inventory_sha256']:raise ValueError('Source inventory changed')
    verify_inventory(proof['training_source_data'],read(proof['source_inventory']))
    if sha(proof['semantic_manifest'])!=proof['semantic_manifest_sha256']:raise ValueError('Semantic source manifest changed')
    out=copy.deepcopy(config);out.pipeline.datamanager.data=Path(proof['training_source_data']);out.pipeline.datamanager.dataparser.data=Path(proof['training_source_data']);out.pipeline.model.semantic_data=proof['geometry_data']
    return out

def parser_check(frozen_worker,geometry_data,source_data):
    import torch
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    spec=importlib.util.spec_from_file_location('train_marshawn_semantic_admission',frozen_worker);worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    target=worker.__getattr__('TuningParser');result={}
    for split,count,factor in [('train',63,2),('val',3,1)]:
        values=[]
        for root in (geometry_data,source_data):
            cfg=NerfstudioDataParserConfig(data=Path(root),eval_mode='filename',orientation_method='none',center_method='none',auto_scale_poses=False,scale_factor=1.,scene_scale=1.,downscale_factor=1)
            values.append(target(cfg)._generate_dataparser_outputs(split))
        a,b=values;assert len(a.image_filenames)==len(b.image_filenames)==count
        if [p.name for p in a.image_filenames]!=[p.name for p in b.image_filenames]:raise ValueError('Actual parser camera order differs')
        if not torch.equal(a.cameras.camera_to_worlds,b.cameras.camera_to_worlds)or not torch.equal(a.scene_box.aabb,b.scene_box.aabb):raise ValueError('Actual parser geometry differs')
        for key in ('height','width','fx','fy','cx','cy'):
            if not torch.equal(getattr(a.cameras,key)*factor,getattr(b.cameras,key)):raise ValueError('Actual parser intrinsics differ '+key)
        result[split]=dict(count=count,scale=factor,poses_and_box_equal=True,image_order_equal=True,height=int(b.cameras.height[0]),width=int(b.cameras.width[0]))
    return result

if __name__=='__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=''  # CLI is a CPU-only inspection tool.
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('geometry-data','source-data','inventory','semantic-manifest','frozen-worker','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();validate_output(a.output,a.geometry_data,a.source_data);result=inspect(a.geometry_data,a.source_data,a.inventory,a.semantic_manifest);result['actual_CPU_parser']=parser_check(a.frozen_worker,a.geometry_data,a.source_data)
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('CPU inspection initialized CUDA')
    result['CUDA_initialized']=False
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ('geometry_source_hashes','splits')}))
