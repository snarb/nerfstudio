"""Build auditable native semantic assets and TRAIN-only whole-cell admission."""
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, distance_transform_edt, maximum_filter
from common import read, atomic, sha


def assemble(source, output, policy='uniform16', all_train=False):
    src,out=Path(source),Path(output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'ready.json').exists():return read(out/'ready.json')
    base=read(src/'native_ready.json');tf=read(src/'transforms.json');box=read(src/'box.json')
    if all_train:tf['train_filenames']=[f['file_path'] for f in tf['frames']]
    names={Path(p).stem for p in tf['train_filenames']}
    for name in ('images','masks','linear','parts'):
        if not (out/name).exists():(out/name).symlink_to((src/name).resolve(),target_is_directory=True)
    for name in ('semantics','heads','cores','lookcloser_frequencies'):(out/name).mkdir(exist_ok=True)
    atomic(out/'transforms.json',tf);atomic(out/'box.json',box)
    records=[];heads={};cores={}
    cv2.setNumThreads(1)
    def one(row):
        c=row['camera'];f=row['camera_row']
        with np.load(src/'semantics'/(c+'.npz')) as z:a={k:z[k] for k in z.files}
        part=read(src/'parts'/(c+'.json'))
        if not part.get('qualified'):raise ValueError('Parts must pass seed qualification before asset assembly')
        with np.load(src/'parts'/(c+'.npz')) as z:crew=z['crew_exclusion'].copy()
        actor=(a['black']|a['grey'])&~crew;valid=a['valid']&~crew
        a['black']=actor;a['grey']=np.zeros_like(actor);a['valid']=valid
        for key in ('W32','W64'):a[key][crew]=0
        core=binary_fill_holes(cv2.morphologyEx(actor.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))))&valid
        outer=cv2.dilate(core.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33)))>0
        yy=np.where(core)[0]
        if len(yy):
            wide=cv2.dilate(core.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(65,65)))>0
            cut=int(yy.min()+.75*(yy.max()-yy.min()+1));outer[:cut]|=wide[:cut]
        a['core']=core;a['band']=outer&~core&valid
        sem=out/'semantics'/(c+'.npz');np.savez_compressed(sem,**a)
        r=dict(row,semantic_sha256=sha(sem));records.append(r)
        if c not in names:return
        part=read(src/'parts'/(c+'.json'))
        if part['archive_sha256']!=sha(src/'parts'/(c+'.npz')):raise ValueError('Part source changed')
        with np.load(src/'parts'/(c+'.npz')) as z:
            hp={'head'+str(p['id']):z['head'+str(p['id'])]&actor for p in part['records']}
        np.savez_compressed(out/'heads'/(c+'.npz'),**hp)
        heads[c]=dict(file=c+'.npz',sha256=sha(out/'heads'/(c+'.npz')),semantic_sha256=r['semantic_sha256'],actors=list(hp))
        opaque=distance_transform_edt(np.pad(actor&valid,1))[1:-1,1:-1]>32
        np.savez_compressed(out/'cores'/(c+'.npz'),core=opaque)
        cores[c]=dict(core_file=c+'.npz',core_sha256=sha(out/'cores'/(c+'.npz')),source_semantic_sha256=r['semantic_sha256'],core_pixels=int(opaque.sum()))
        dest=out/'lookcloser_frequencies'/(c+'.pt')
        if policy=='uniform16':torch.save(torch.full((f['h']//8,f['w']//8),16.,dtype=torch.float32),dest)
        else:
            incoming=src/'learned'/policy/(c+'.pt')
            if not incoming.exists():raise FileNotFoundError(incoming)
            shutil.copy2(incoming,dest)
        atomic(dest.with_suffix('.json'),dict(camera=c,value_type='scalar_resolution',patch_size=8,stride=8,n_levels=16,min_res=16,max_res=8192,image_shape=[f['h'],f['w']],source_image_sha256=row['rgb_sha256'],policy=policy,frequency_map_sha256=sha(dest)))
        print('ASSETS',c,flush=True)
    with ThreadPoolExecutor(6) as pool:list(pool.map(one,base['records']))
    records.sort(key=lambda r:r['camera'])
    atomic(out/'heads/manifest.json',dict(kind='SEQUENCE_NATIVE_SAM3_HEAD_FACE_IMPORTANCE_V1',train_cameras=sorted(names),records=heads))
    atomic(out/'cores/manifest.json',dict(kind='SEQUENCE_OPAQUE_CORE32_V1',train_cameras=sorted(names),records=cores,erosion_native_pixels=32))
    ready=dict(base,train=len(names),in_sample=all_train,frequency_source_policy=policy,records=records,transforms_sha256=sha(out/'transforms.json'))
    atomic(out/'ready.json',ready)
    return ready


def admission(data):
    from mask_visual_hull_aabb import MaskCamera,classify_cells
    from build_marshawn_semantic_resolution import cell_bounds
    from marshawn_semantic_cell_admission import WholeCellAdmission
    root=Path(data);ready=read(root/'ready.json');tf=read(root/'transforms.json');bounds=np.asarray(ready['aabb'])
    out=root/'admission';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():return read(out/'manifest.json')
    bg=np.zeros(128**3,np.uint8);records=[]
    for f in tf['frames']:
        if f['file_path'] not in tf['train_filenames']:continue
        c=f['physical_camera']
        with np.load(root/'semantics'/(c+'.npz')) as z:fg=z['black']|z['grey'];valid=z['valid']
        # Pixels beyond native crop/FOV remain UNKNOWN. Subject-adjacent64px remain supported.
        with np.load(root/'parts'/(c+'.npz')) as z:unknown=z['unknown_people']
        cam=MaskCamera(c,np.asarray(f['transform_matrix']),f['fl_x'],f['fl_y'],f['cx'],f['cy'],fg,unknown,valid).prepare(64)
        for offset in range(0,len(bg),32768):
            ids=np.arange(offset,min(offset+32768,len(bg)));ids=ids[bg[ids]<2]
            if not len(ids):continue
            lo,hi=cell_bounds(ids,128,bounds);_,_,clear,_=classify_cells([cam],lo,hi);bg[ids]+=clear.astype(np.uint8)
        records.append(dict(camera=c,semantic_sha256=sha(root/'semantics'/(c+'.npz'))))
        print('ADMISSION',c,int((bg<2).sum()),flush=True)
    base=(bg<2).reshape((128,)*3);pad=maximum_filter(base,size=3,mode='constant',cval=False)
    mapped=WholeCellAdmission(pad,bounds).map_target_grid((128,)*3,bounds)
    np.savez_compressed(out/'grid.npz',base=base,pad1=pad,mapped=mapped,aabb=bounds)
    m=dict(kind='SEQUENCE_TRAIN_ONLY_W64_WHOLE_CELL_V1',records=records,train_count=len(records),grid_sha256=sha(out/'grid.npz'),ready_sha256=sha(root/'ready.json'),transforms_sha256=sha(root/'transforms.json'),mapped_sha256=__import__('hashlib').sha256(mapped.tobytes()).hexdigest(),clear_evidence='two distinct TRAIN views classify whole8corner projected cell background;64nativepixel support; outsidecrop UNKNOWN; one pad1 then conservative wholecell mapping; learned R0')
    atomic(out/'manifest.json',m);return m


def exr_variant(source,output):
    """Same crop geometry/masks/maps, scene-linear32-bit EXR targets."""
    from nerfstudio.data.utils.data_utils import write_exr_image
    src,out=Path(source),Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'ready.json').exists():return read(out/'ready.json')
    (out/'images').mkdir(exist_ok=True)
    for name in ('masks','linear','parts','semantics','heads','cores','lookcloser_frequencies'):
        if not (out/name).exists():(out/name).symlink_to((src/name).resolve(),target_is_directory=True)
    ready=read(src/'ready.json');tf=read(src/'transforms.json');records=[]
    for row in ready['records']:
        c=row['camera'];path=out/'images'/(c+'.exr')
        if not path.exists():write_exr_image(path,np.load(src/'linear'/(c+'.npy')))
        records.append(dict(row,image='images/'+c+'.exr',rgb_sha256=sha(path),evaluation_png_reference=str((src/row['image']).resolve()),evaluation_png_sha256=row['rgb_sha256']))
    for f in tf['frames']:f['file_path']=str(Path(f['file_path']).with_suffix('.exr'))
    for key in ('train_filenames','val_filenames','test_filenames'):tf[key]=[str(Path(p).with_suffix('.exr')) for p in tf[key]]
    atomic(out/'transforms.json',tf);atomic(out/'box.json',read(src/'box.json'))
    ready=dict(ready,records=records,transforms_sha256=sha(out/'transforms.json'),image_format='scene_linear_rec709_exr32')
    atomic(out/'ready.json',ready)
    (out/'admission').mkdir(exist_ok=True);shutil.copy2(src/'admission/grid.npz',out/'admission/grid.npz')
    m=read(src/'admission/manifest.json');m.update(ready_sha256=sha(out/'ready.json'),transforms_sha256=sha(out/'transforms.json'),geometry_identity_parent_sha256=sha(src/'admission/manifest.json'))
    atomic(out/'admission/manifest.json',m);return ready


def map_variant(template,native,output,policy):
    """Only frequency-map files differ; share exact immutable semantic bytes."""
    src,native,out=Path(template),Path(native),Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'ready.json').exists():return read(out/'ready.json')
    ready=read(src/'ready.json');tf=read(src/'transforms.json')
    for name in ('images','masks','linear','parts','semantics','heads','cores'):
        if not (out/name).exists():(out/name).symlink_to((src/name).resolve(),target_is_directory=True)
    atomic(out/'transforms.json',tf);atomic(out/'box.json',read(src/'box.json'))
    (out/'lookcloser_frequencies').mkdir(exist_ok=True)
    for filename in tf['train_filenames']:
        c=Path(filename).stem;source=native/'learned'/policy/(c+'.pt');dest=out/'lookcloser_frequencies'/(c+'.pt')
        shutil.copy2(source,dest);meta=read(src/'lookcloser_frequencies'/(c+'.json'))
        meta.update(policy=policy,frequency_map_sha256=sha(dest),fit_receipt_sha256=sha(source.with_suffix('.json')))
        atomic(dest.with_suffix('.json'),meta)
    ready.update(frequency_source_policy=policy);atomic(out/'ready.json',ready)
    (out/'admission').mkdir(exist_ok=True);shutil.copy2(src/'admission/grid.npz',out/'admission/grid.npz')
    m=read(src/'admission/manifest.json');m.update(ready_sha256=sha(out/'ready.json'),geometry_identity_parent_sha256=sha(src/'admission/manifest.json'))
    atomic(out/'admission/manifest.json',m);return ready


def scale_variant(source,output,scale):
    """Change model units isotropically; accepted physical AABB membership is invariant."""
    src,out=Path(source),Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'ready.json').exists():return read(out/'ready.json')
    if not 0<scale<=10:raise ValueError('Positive global model scale required')
    ready=read(src/'ready.json');tf=read(src/'transforms.json');original=read(src/'box.json')
    box=dict(original);box['canonical_rig_box']=original
    box['aabb']=(np.asarray(original['aabb'])*scale).tolist()
    S=np.diag([scale,scale,scale,1.]);box['world_to_box']=(S@np.asarray(original['world_to_box'])).tolist()
    box['box_to_world']=np.linalg.inv(np.asarray(box['world_to_box'])).tolist()
    box['coordinate_system']='Model units: isotropic global scale of accepted floor box; physical box unchanged'
    box['global_model_units_scale']=scale
    for f in tf['frames']:
        pose=np.asarray(f['transform_matrix']);pose[:3,3]*=scale
        # Cameras keep orthonormal rotation axes. Only centers change units.
        f['transform_matrix']=pose.tolist()
    by_name={f['physical_camera']:f for f in tf['frames']}
    ready.update(aabb=box['aabb'],world_to_box=box['world_to_box'],model_units_scale=scale,records=[dict(r,camera_row=by_name[r['camera']]) for r in ready['records']])
    for name in ('images','masks','linear','parts','semantics','heads','cores','lookcloser_frequencies'):
        if not (out/name).exists():(out/name).symlink_to((src/name).resolve(),target_is_directory=True)
    atomic(out/'box.json',box);atomic(out/'transforms.json',tf);ready['transforms_sha256']=sha(out/'transforms.json');atomic(out/'ready.json',ready)
    (out/'admission').mkdir(exist_ok=True)
    with np.load(src/'admission/grid.npz') as z:arrays={k:z[k] for k in z.files}
    arrays['aabb']*=scale;np.savez_compressed(out/'admission/grid.npz',**arrays)
    m=read(src/'admission/manifest.json');m.update(ready_sha256=sha(out/'ready.json'),transforms_sha256=sha(out/'transforms.json'),grid_sha256=sha(out/'admission/grid.npz'),units_parent_sha256=sha(src/'admission/manifest.json'))
    atomic(out/'admission/manifest.json',m)
    rng=np.random.default_rng(7281);bounds=np.asarray(original['aabb']);p=rng.uniform(bounds[0]-.5,bounds[1]+.5,(10000,3))
    np.testing.assert_array_equal(((p>=bounds[0])&(p<=bounds[1])).all(1),((p*scale>=np.asarray(box['aabb'])[0])&(p*scale<=np.asarray(box['aabb'])[1])).all(1))
    atomic(out/'units_parity.json',dict(passed=True,points=10000,global_scale=scale,physical_aabb_unchanged=True,camera_rotations_unchanged=True,focal_and_native_pixel_rays_unchanged=True))
    return ready


def resolution_variant(source,output,max_res):
    """Uniform16 remains the exact minimum scalar under a new consumer ladder."""
    src,out=Path(source),Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'ready.json').exists():return read(out/'ready.json')
    ready=read(src/'ready.json')
    if ready['frequency_source_policy']!='uniform16':raise ValueError('Resolution ablation requires uniform16; learned fitter schedules are separate')
    for name in ('images','masks','linear','parts','semantics','heads','cores'):
        (out/name).symlink_to((src/name).resolve(),target_is_directory=True)
    for name in ('transforms.json','box.json'):shutil.copy2(src/name,out/name)
    (out/'lookcloser_frequencies').mkdir(exist_ok=True)
    for path in (src/'lookcloser_frequencies').glob('*.pt'):
        values=torch.load(path,map_location='cpu',weights_only=True)
        if not torch.all(values==16):raise ValueError('Uniform16 numeric contract differs')
        dest=out/'lookcloser_frequencies'/path.name;shutil.copy2(path,dest)
        metadata=read(path.with_suffix('.json'));metadata.update(max_res=max_res,consumer_grid_schedule_override=True,source_uniform_scalar_unchanged=16)
        atomic(dest.with_suffix('.json'),metadata)
    atomic(out/'ready.json',dict(ready,consumer_max_res=max_res))
    (out/'admission').mkdir(exist_ok=True);shutil.copy2(src/'admission/grid.npz',out/'admission/grid.npz')
    m=read(src/'admission/manifest.json');m.update(ready_sha256=sha(out/'ready.json'))
    atomic(out/'admission/manifest.json',m);return read(out/'ready.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--policy',default='uniform16');p.add_argument('--all-train',action='store_true');a=p.parse_args();assemble(a.source,a.output,a.policy,a.all_train);admission(a.output)
