"""Freeze frame1660 inputs in the accepted floor frame; no RGB edits."""
from pathlib import Path
import argparse, hashlib, json, os
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def _base_inputs(data, ready):
    """Validate existing RGB/semantic bindings before adding any new inventory."""
    transforms=json.loads((data/'transforms.json').read_text())
    if sha(data/'transforms.json') != ready['transforms_sha256']:
        raise ValueError('Prepared transforms changed')
    box=json.loads((data/'box.json').read_text())
    for key in ('aabb','world_to_box'):
        if box[key] != ready[key]:raise ValueError('Prepared box differs: '+key)
    frames={f['file_path']:f for f in transforms['frames']}
    if len(frames)!=len(ready['records']):raise ValueError('Prepared frame/record count differs')
    files={'transforms.json','box.json'}
    for row in ready['records']:
        image=row['image'];semantic='semantics/'+row['camera']+'.npz'
        if sha(data/image)!=row['rgb_sha256']:raise ValueError('Prepared RGB changed: '+image)
        if sha(data/semantic)!=row['semantic_sha256']:raise ValueError('Prepared semantics changed: '+semantic)
        files.update((image,semantic,frames[image]['mask_path']))
    if not set(transforms['train_filenames']).issubset(frames):raise ValueError('Unknown training frame')
    return transforms,files


def _frequency_inventory(data, transforms):
    sources={};thresholds=set();files=set()
    for rel in transforms['train_filenames']:
        stem=Path(rel).stem;path=data/'lookcloser_frequencies'/(stem+'.pt')
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta.get('value_type')!='scalar_resolution' or meta.get('source_image_sha256')!=sha(data/rel):
            raise ValueError('Frequency source/value contract mismatch: '+stem)
        required=dict(train_steps_per_level=1000,patch_size=8,stride=8,n_levels=16,min_res=16,max_res=8192,eval_patch_batch_size=1024)
        for key,val in required.items():
            if meta.get(key)!=val:raise ValueError('Frequency recipe mismatch: '+stem+' '+key)
        threshold=float(meta['ssim_threshold'])
        if not .0<threshold<1.:raise ValueError('Invalid frequency threshold')
        thresholds.add(threshold)
        for f in (path,path.with_suffix('.json')):files.add(str(f.relative_to(data)))
        sources[stem]=dict(source_image_sha256=meta['source_image_sha256'],
            fit_script_sha256=meta.get('fit_script_sha256'),
            preprocess_source_sha256=meta.get('preprocess_source_sha256'),
            provenance_note='Missing legacy source hashes remain unknown, not inferred from current code')
    if len(thresholds)!=1:raise ValueError('Mixed or empty frequency-map recipe')
    recipe=dict(ssim_threshold=thresholds.pop(),train_steps_per_level=1000,patch_size=8,stride=8,
                ssim_window_size=7,n_levels=16,min_res=16,max_res=8192,eval_patch_batch_size=1024)
    return files,sources,recipe


def verify_inventory(data):
    """Read-only verification; missing evidence is not silently recreated."""
    data=Path(data);ready=json.loads((data/'ready.json').read_text())
    inventory=ready.get('inventory')
    if not inventory or inventory.get('schema_version')!=1:raise ValueError('Prepared inventory missing; run refresh_inventory before launch')
    transforms,files=_base_inputs(data,ready)
    map_files,sources,recipe=_frequency_inventory(data,transforms);files|=map_files
    if set(inventory['files'])!=files:raise ValueError('Prepared inventory file coverage differs')
    for relative,digest in inventory['files'].items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Inventory paths must be relative')
        if sha(data/path)!=digest:raise ValueError('Prepared inventory hash mismatch: '+relative)
    if inventory['frequency_sources']!=sources or inventory['frequency_recipe']!=recipe:
        raise ValueError('Frequency provenance changed')
    if ready.get('box_sha256')!=inventory['files']['box.json']:raise ValueError('Box receipt mismatch')
    return ready


def refresh_inventory(data):
    """Seal legacy prepared data once; otherwise verify instead of rebasing it.

    Does not alter images, masks, transforms, box or frequency maps. Call before
    launching workers, because adding this receipt changes ready.json identity.
    """
    data=Path(data);ready=json.loads((data/'ready.json').read_text())
    if 'inventory' in ready:return verify_inventory(data)
    transforms,files=_base_inputs(data,ready)
    map_files,sources,recipe=_frequency_inventory(data,transforms);files|=map_files
    recipe_path=Path(__file__).resolve().parents[1]/'experiments/static_leader_reproduction_recipe.md'
    recipe_source=ready.get('recipe_source')
    if recipe_source is None:
        if not recipe_path.is_file():raise ValueError('Leader recipe provenance missing')
        recipe_source=dict(path='LookCloser/experiments/static_leader_reproduction_recipe.md',sha256=sha(recipe_path))
    ready['recipe_source']=recipe_source
    ready['box_sha256']=sha(data/'box.json')
    ready['frequency_policy']=f"train-only progressive SSIM {recipe['ssim_threshold']}; source-bound scalar-resolution maps"
    ready['inventory']=dict(schema_version=1,files={rel:sha(data/rel) for rel in sorted(files)},
        frequency_recipe=recipe,frequency_sources=sources,recipe_source=recipe_source,
        preparation_source_sha256=sha(__file__),
        limitations=['Progressive-SSIM convergence proxy, not Fourier energy',
          'Maximum level conflates late convergence and unresolved patches in legacy maps',
          'Same seed does not certify bitwise GPU fit repeatability; exact produced map hashes are frozen',
          'Only complete8x8 map patches; residual raster rows are not stretched'])
    write(data/'ready.json',ready)
    return verify_inventory(data)

def prepare(output):
    import cv2
    from scipy.ndimage import distance_transform_edt,binary_fill_holes
    from marshawn_actor_coordinates import raw_mask_coverage_to_HD
    from PIL import Image,ImageDraw
    cv2.setNumThreads(2)
    root=Path('/fsx/tmp/marshawn_001660_hd');source=root/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png'
    confidence=root/'two_actor_isolation/confidence_native_hair_v2/manifest.json'
    m=json.loads(confidence.read_text());assert m['native_hair66_complete'] and len(m['rows'])==66
    boxpath=Path(__file__).resolve().parents[1]/'configs/aabb_review/marshawn_1660_1900_floor.json'
    box=json.loads(boxpath.read_text());transform=np.array(box['world_to_box'])
    tf=json.loads((source/'transforms.json').read_text());rows={r['camera']:r for r in m['rows']}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'ready.json').exists():
        verify_inventory(output);return
    for d in ['images','masks','semantics','lookcloser_frequencies','review']: (output/d).mkdir(exist_ok=True)
    records=[]
    for frame in tf['frames']:
        camera=frame['physical_camera'];r=rows[camera]
        for key in ['rgb','archive','raw_archive']:
            if sha(r[key])!=r[key+'_sha256']:raise ValueError('Source hash changed '+camera+' '+key)
        original=source/frame['file_path'];dest=output/frame['file_path']
        if not dest.exists():dest.symlink_to(original)
        valid=np.asarray(Image.open(source/frame['mask_path']).convert('L'))>0
        with np.load(r['archive']) as a: weights={k:a[k].copy() for k in ['W32','W64']};veto=a['source_ineligible'].copy();nominal=a['nominal_coverage']>=.5
        raw=np.load(r['raw_archive']);masks={}
        for actor in ['black','grey']:
            full=np.zeros((4320,8192),np.float32);x0,y0,x1,y1=raw['crop_xyxy'];full[y0:y1,x0:x1]=raw[actor]
            masks[actor]=(raw_mask_coverage_to_HD(full,r['camera_row'])>=.5)&~veto&valid
        # Assign add-only reviewed hair to its nearest actor, without overlap.
        extra=nominal&~(masks['black']|masks['grey'])&~veto&valid
        db=distance_transform_edt(~masks['black']);dg=distance_transform_edt(~masks['grey'])
        masks['black']|=extra&(db<=dg);masks['grey']|=extra&(dg<db)
        overlap=masks['black']&masks['grey'];masks['black'][overlap]=False;masks['grey'][overlap]=False
        core=binary_fill_holes(cv2.morphologyEx(nominal.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))))&valid&~veto
        outer=cv2.dilate(core.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33)))>0
        yy=np.where(core)[0]
        if len(yy):
            wide=cv2.dilate(core.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(65,65)))>0
            cut=int(yy.min()+.75*(yy.max()-yy.min()+1));outer[:cut]|=wide[:cut]
        band=outer&~core&valid&~veto
        for w in weights.values():w[~valid|veto]=0
        semantic=output/'semantics'/f'{camera}.npz';np.savez_compressed(semantic,**masks,**weights,core=core,band=band,valid=valid,nominal=nominal&valid&~veto)
        Image.fromarray(valid.astype('uint8')*255).save(output/frame['mask_path'])
        frame['transform_matrix']=(transform@np.array(frame['transform_matrix'])).tolist()
        if frame['file_path'] in tf['train_filenames']:
            freq=root/'frequency_radial_k4/maps'/(original.stem+'.pt');meta=json.loads(freq.with_suffix('.json').read_text())
            if meta.get('source_image_sha256')!=sha(original):raise ValueError('Frequency source mismatch '+camera)
            for p in [freq,freq.with_suffix('.json')]:
                target=output/'lookcloser_frequencies'/p.name
                if not target.exists():target.symlink_to(p)
        crops={}
        for actor,mask in masks.items():
            y,x=np.where(mask)
            if not len(x):raise ValueError('Empty actor '+camera+' '+actor)
            crops[actor]=[max(0,int(x.min())-32),max(0,int(y.min())-32),min(mask.shape[1],int(x.max())+33),min(mask.shape[0],int(y.max())+33)]
        records.append(dict(camera=camera,image=frame['file_path'],semantic=str(semantic),semantic_sha256=sha(semantic),rgb_sha256=sha(original),crops=crops))
        im=np.asarray(Image.open(original).convert('RGB')).copy();im[~(masks['black']|masks['grey'])]//=4
        preview=Image.fromarray(im);preview.thumbnail((640,360));ImageDraw.Draw(preview).text((5,5),camera,fill='red');preview.save(output/'review'/f'{camera}.jpg')
        print(camera,flush=True)
    tf['orientation_override']='none';tf.pop('applied_transform',None);tf.pop('applied_scale',None)
    write(output/'transforms.json',tf);write(output/'box.json',box)
    write(output/'ready.json',dict(frame=1660,train=63,validation=['DC','HC','KA'],aabb=box['aabb'],world_to_box=box['world_to_box'],source_confidence_sha256=sha(confidence),transforms_sha256=sha(output/'transforms.json'),records=records,frequency_policy='existing qualified SSIM.95 maps; train only',sparse_depth='not yet supplied; must explicitly qualify or document before launch'))
    refresh_inventory(output)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--refresh-inventory',action='store_true');a=p.parse_args();refresh_inventory(a.output) if a.refresh_inventory else prepare(a.output)
