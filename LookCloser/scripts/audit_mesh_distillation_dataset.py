"""Independent data/loader audit and native review panels; no training."""
import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

from joint_temporal_texture import read, sha, atomic_json, cameras, SOURCE
from prepare_mesh_distillation_dataset import OUTPUT, FRAME


def package(out):
    """Unique image stems are required by LookCloser's frequency-map cache."""
    root=out/'synthetic';request=read(out/'request.json')
    receipt=out/'packaging.json'
    if receipt.exists():
        for name,digest in read(out/'dataset_hashes.json').items():
            assert sha(out/name)==digest,name
        return
    old=read(root/'transforms.json');new=deepcopy(old)
    atomic_json(out/'config/synthetic_before_flat_package.json',old)
    mapping={}
    for row in new['frames']:
        name=row['teacher_view_id']
        fields={'file_path':('images','.png'), 'mask_path':('masks','.png'),
                'depth_file_path':('depth','.npy.gz'), 'full_depth_file_path':('full_depth_z','.npy.gz'),
                'confidence_file_path':('confidence','.png')}
        for key,(folder,ext) in fields.items():
            source=root/row[key];dest=root/folder/(name+ext);dest.parent.mkdir(exist_ok=True)
            if not dest.exists():
                try:os.link(source,dest)
                except OSError:shutil.copyfile(source,dest)
            assert sha(source)==sha(dest)
            before=row[key];row[key]=str(dest.relative_to(root))
            if key=='file_path':mapping[before]=row[key]
    for split in ['train_filenames','val_filenames','test_filenames']:
        new[split]=[mapping[x] for x in old[split]]
    assert len({Path(x['file_path']).stem for x in new['frames']})==324
    atomic_json(root/'transforms.json',new)
    atomic_json(out/'dataset_hashes.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
                if p.is_file() and p.parts[len(out.parts)] in ['mesh','config','synthetic','real']})
    atomic_json(receipt,dict(status='pass',unique_stems=324,uses_verified_hardlinks_or_copies=True,
                request_sha256=sha(out/'request.json'),script_sha256=sha(__file__),
                transforms_sha256=sha(root/'transforms.json'),
                reason='LookCloser frequency-map cache keys by image stem; each must be unique'))


def parser_smoke(out):
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    from nerfstudio.data.utils.data_utils import get_depth_image_from_path
    request=read(out/'request.json');rows=[]
    for name in ['train_0033','train_0062','val_0000']:
        row=deepcopy(next(p['camera'] for p in request['plan'] if p['id']==name))
        root=out/'synthetic/views'/name
        row.update(file_path=str(root/'rgb.png'),mask_path=str(root/'mask.png'),
                   depth_file_path=str(root/'depth_supervision.npy.gz'))
        rows.append(row)
    path=out/'checks/parser_smoke.json'
    atomic_json(path,dict(orientation_override='none',frames=rows,
                train_filenames=[r['file_path'] for r in rows[:2]],
                val_filenames=[rows[2]['file_path']],test_filenames=[rows[2]['file_path']]))
    config=NerfstudioDataParserConfig(data=path,orientation_method='none',center_method='none',
        auto_scale_poses=False,scale_factor=1.,depth_unit_scale_factor=1.,downscale_factor=1,
        eval_mode='filename',load_3D_points=False)
    parser=config.setup();counts={}
    for split,n in [('train',2),('val',1)]:
        data=parser.get_dataparser_outputs(split=split);assert len(data.image_filenames)==n
        counts[split]=n
        assert data.dataparser_scale==1 and data.metadata['depth_unit_scale_factor']==1
        for path in data.metadata['depth_filenames']:
            loaded=get_depth_image_from_path(path,height=1080,width=1920,scale_factor=1.)
            with gzip.open(path,'rb') as f:raw=np.load(f)
            np.testing.assert_array_equal(loaded.numpy().squeeze(),raw)
    atomic_json(out/'checks/parser_smoke_result.json',dict(status='pass',counts=counts,depth_loading_exact=True))


def panels(out, names):
    review=out/'review';review.mkdir(exist_ok=True)
    for name in names:
        dest=out/'synthetic/views'/name
        rgb=Image.open(dest/'rgb.png')
        hit=np.array(Image.open(dest/'mesh_hit.png'))>0
        mask=np.array(Image.open(dest/'mask.png'))>0
        weight=np.array(Image.open(dest/'confidence.png'))
        panel=Image.new('RGB',(1500,570),'#303030');draw=ImageDraw.Draw(panel)
        panel.paste(rgb.crop((650,400,1150,950)),(0,20))
        panel.paste(rgb.crop((687,540,987,800)).resize((500,433)),(500,20))
        diagnostic=np.zeros((*hit.shape,3),np.uint8)
        diagnostic[hit]=[150,0,0];diagnostic[mask]=np.stack([weight[mask]]*3,axis=1)
        panel.paste(Image.fromarray(diagnostic).crop((650,400,1150,950)),(1000,20))
        draw.text((5,3),name+' native face / lips enlarged / weight (red=unsupported)',fill='white')
        panel.save(review/(name+'_native.png'))


def contacts(out):
    plan=read(out/'request.json')['plan'];review=out/'review';review.mkdir(exist_ok=True)
    for start in range(0,len(plan),24):
        group=plan[start:start+24]
        if not all((out/'synthetic/views'/p['id']/'complete.json').exists() for p in group):continue
        sheet=Image.new('RGB',(6*240,4*455),'#303030');draw=ImageDraw.Draw(sheet)
        for j,p in enumerate(group):
            im=Image.open(out/'synthetic/views'/p['id']/'rgb.png').transpose(Image.Transpose.ROTATE_90)
            im.thumbnail((240,427));x=(j%6)*240;y=(j//6)*455
            sheet.paste(im,(x,y+24));draw.text((x+4,y+5),p['id'],fill='white')
        sheet.save(review/f'contact_{start//24:02d}.jpg',quality=93)


def pose_plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=read(out/'request.json')['plan'];fig,ax=plt.subplots(figsize=(11,4))
    for kind,color,label in [('real_train_pose','black','62 real train positions'),
                             ('train','royalblue','238 synthetic train positions'),
                             ('val','darkorange','24 synthetic validation positions')]:
        selected=[p for p in plan if (p['kind']==kind if kind=='real_train_pose' else p['kind']=='local_interpolation' and p['split']==kind)]
        xy=[]
        for p in selected:
            n=p['camera']['physical_camera']
            xy.append(p.get('rig_xy',[ord(n[0])-65,ord(n.split('_')[1][0])-65]))
        xy=np.asarray(xy);ax.scatter(xy[:,0],xy[:,1],s=20 if kind=='real_train_pose' else 9,label=label,color=color,alpha=.8)
    ax.set_xticks(range(14),list('ABCDEFGHIJKLMN'));ax.set_yticks(range(5),list('ABCDE'))
    ax.set_xlabel('Rig column');ax.set_ylabel('Rig row');ax.invert_yaxis();ax.legend(loc='upper left',bbox_to_anchor=(1.02,1))
    fig.tight_layout();fig.savefig(out/'review/camera_sampling.png',dpi=150);plt.close(fig)


def train_pairs(out,names):
    request=read(out/'request.json');real=read(out/'real/transforms.json')
    gt_by_name={r['physical_camera']:r for r in real['frames']}
    for name in names:
        p=next(p for p in request['plan'] if p['id']==name)
        if p['kind']!='real_train_pose':raise ValueError('Train pose required')
        gt=Image.open(out/'real'/gt_by_name[p['camera']['physical_camera']]['file_path'])
        pred=Image.open(out/'synthetic/views'/name/'rgb.png')
        panel=Image.new('RGB',(1700,980),'#303030');draw=ImageDraw.Draw(panel)
        for i,(im,label) in enumerate([(gt,'real train RGB'),(pred,'mesh teacher')]):
            crop=im.transpose(Image.Transpose.ROTATE_90).crop((150,550,1000,1500))
            panel.paste(crop,(i*850,30));draw.text((i*850+6,6),name+' '+label,fill='white')
        panel.save(out/'review'/(name+'_real_pair.png'))


def parity(out, name):
    """Compare cached producer to unmodified historical renderer byte-for-byte."""
    import torch
    import render_smooth_temporal_mesh_video as renderer
    import importlib
    from run_view_consistent_dynamic_video import install
    request=read(out/'request.json');p=next(p for p in request['plan'] if p['id']==name)
    importlib.reload(renderer)  # Each parity control starts from unpatched source.
    install();torch.set_num_threads(2)
    dest=out/'parity'/name;(dest/'frames').mkdir(parents=True,exist_ok=True)
    atomic_json(dest/'request.json',dict(recipe=request['recipe']))
    rec=deepcopy(request['record']);rec['camera']=p['camera']
    manifest=dict(source_images=[dict(physical_camera=k,sha256=v) for k,v in request['source_hashes'].items()])
    renderer.render_one(dest,rec,manifest)
    checks={}
    for old,new in [('prediction_native.png','rgb.png'),('source_ids.png','source_ids.png')]:
        a=np.array(Image.open(dest/'frames'/FRAME/old));b=np.array(Image.open(out/'synthetic/views'/name/new))
        checks[new]=dict(exact_equal=bool(np.array_equal(a,b)), changed_pixels=int(np.any(a!=b,axis=2).sum()) if a.ndim==3 else int((a!=b).sum()))
    atomic_json(out/'parity'/f'{name}.json',checks)
    assert all(x['exact_equal'] for x in checks.values()),checks
    print('parity',name,'RGB/source IDs byte-identical')


def heldout_reference(out):
    import torch
    import render_smooth_temporal_mesh_video as renderer
    from run_view_consistent_dynamic_video import install
    request=read(out/'request.json');real=read(out/'real/transforms.json')
    target=next(f for f in real['frames'] if f['physical_camera']=='F004_B005_1210O9')
    target=deepcopy(target);target.pop('file_path')
    install();torch.set_num_threads(2)
    dest=out/'heldout_teacher';(dest/'frames').mkdir(parents=True,exist_ok=True)
    atomic_json(dest/'request.json',dict(recipe=request['recipe'],uses_target_rgb=False,
                dataset_request_sha256=sha(out/'request.json'),purpose='evaluation only, not pretraining input'))
    rec=deepcopy(request['record']);rec['camera']=target
    manifest=dict(source_images=[dict(physical_camera=k,sha256=v) for k,v in request['source_hashes'].items()])
    renderer.render_one(dest,rec,manifest)
    gt=Image.open(out/'real'/real['val_filenames'][0])
    pred=Image.open(dest/'frames'/FRAME/'prediction_native.png')
    panel=Image.new('RGB',(1000,580),'#303030');draw=ImageDraw.Draw(panel)
    for i,(im,label) in enumerate([(gt,'held-out real, fixed exposure'),(pred,'teacher, train RGB only')]):
        panel.paste(im.crop((650,400,1150,950)),(i*500,30));draw.text((i*500+5,5),label,fill='white')
    (out/'review').mkdir(exist_ok=True);panel.save(out/'review/heldout_face_pair.png')
    old=Path('/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/config/face_polygons/000973.json')
    # The legacy polygon visibly included background/hair; do NOT reuse it as
    # face-only supervision. These points were manually selected on this GT.
    local=[(140,5),(215,18),(290,30),(370,55),(450,88),(484,145),
           (488,215),(468,250),(430,278),(393,293),(350,300),
           (314,301),(275,302),(239,299),(202,289),(180,268),
           (149,255),(115,241),(83,225),(65,207),(65,166),
           (83,123),(98,78),(120,32)]
    roi=dict(schema_version=1,frame_id=FRAME,selection_method='manual_polygon_on_heldout_gt_only',
        prediction_used_for_selection=False,
        ground_truth_sha256=sha(out/'real'/real['val_filenames'][0]),
        include_polygons=[[[x+650,y+400] for x,y in local]],exclude_polygons=[],
        included_anatomy=['visible frontal face skin','eyes','nose','mouth'],
        excluded_anatomy_or_objects=['hair','neck','room','hand','lipstick tube'],
        rejected_legacy_roi_path=str(old),rejected_legacy_roi_sha256=sha(old),
        notes='New GT-only inset face polygon. Legacy ROI included visible hair/background. Do not compare numbers with that protocol.',
        roi_visual_status='pending')
    atomic_json(out/'real/face_roi.json',roi)
    overlay=gt.copy();draw=ImageDraw.Draw(overlay)
    for poly in roi['include_polygons']:draw.line([tuple(p) for p in poly]+[tuple(poly[0])],fill='lime',width=2)
    for poly in roi['exclude_polygons']:draw.line([tuple(p) for p in poly]+[tuple(poly[0])],fill='red',width=2)
    overlay.crop((650,400,1150,950)).save(out/'review/heldout_gt_roi.png')


def audit(out):
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    from nerfstudio.data.utils.data_utils import get_depth_image_from_path
    request=read(out/'request.json')
    assert {p.name for p in (out/'synthetic/views').iterdir() if p.is_dir()}=={p['id'] for p in request['plan']}
    for p in request['plan']:
        root=out/'synthetic/views'/p['id']
        mask=np.array(Image.open(root/'mask.png'))>0
        hit=np.array(Image.open(root/'mesh_hit.png'))>0
        count=np.array(Image.open(root/'visibility_count.png'))
        source=np.array(Image.open(root/'source_ids.png'))
        depthmask=np.array(Image.open(root/'depth_mask.png'))>0
        inferred=np.array(Image.open(root/'inferred_geometry.png'))>0
        confidence=np.array(Image.open(root/'confidence.png'))
        assert np.array_equal(mask,hit&(count>0))
        assert np.array_equal(mask,source<62)
        assert not (depthmask&(~mask|inferred)).any()
        assert not (confidence[~mask]>0).any()
        with gzip.open(root/'depth_supervision.npy.gz','rb') as f:depth=np.load(f)
        assert np.array_equal(depth>0,depthmask)
    rows,_,_=cameras(FRAME)
    assert sha(SOURCE/FRAME/'transforms.json')==request['source_transforms_sha256']
    for row in rows:
        assert sha(row['file_path'])==request['source_hashes'][row['physical_camera']]
    protocol=read(out/'real/protocol.json')
    assert sha(protocol['heldout_source'])==protocol['heldout_source_sha256']
    for name,digest in read(out/'dataset_hashes.json').items():
        assert sha(out/name)==digest,name
    reports={}
    for dataset,expected in [('synthetic',(300,24)),('real',(62,1))]:
        config=NerfstudioDataParserConfig(data=out/dataset,orientation_method='none',
            center_method='none',auto_scale_poses=False,scale_factor=1.,depth_unit_scale_factor=1.,
            downscale_factor=1,eval_mode='filename',load_3D_points=False,scene_scale=.15)
        parser=config.setup();parsed={}
        meta=read(out/dataset/'transforms.json');byfile={f['file_path']:f for f in meta['frames']}
        for split,count in zip(['train','val'],expected):
            data=parser.get_dataparser_outputs(split=split)
            assert len(data.image_filenames)==count
            assert data.dataparser_scale==1.
            np.testing.assert_allclose(data.dataparser_transform.numpy(),np.eye(4)[:3],atol=1e-7)
            for i,path in enumerate(data.image_filenames):
                row=byfile[str(path.relative_to(out/dataset))]
                np.testing.assert_allclose(data.cameras.camera_to_worlds[i].numpy(),np.array(row['transform_matrix'])[:3],atol=1e-7)
            parsed[split]=len(data.image_filenames)
            if dataset=='synthetic':
                depthpath=data.metadata['depth_filenames'][0]
                actual=get_depth_image_from_path(depthpath,height=1080,width=1920,scale_factor=1.)
                with gzip.open(depthpath,'rb') as f:saved=np.load(f)
                np.testing.assert_allclose(actual.numpy().squeeze(),saved,atol=0)
                assert len(data.mask_filenames)==count
        reports[dataset]=parsed
    syn=read(out/'synthetic/transforms.json');real=read(out/'real/transforms.json')
    synthetic_by_name={f['physical_camera']:f for f in syn['frames'] if not f['physical_camera'].startswith('synthetic_')}
    assert len(synthetic_by_name)==62
    for row in real['frames']:
        if row['physical_camera'] not in synthetic_by_name:continue
        teacher=synthetic_by_name[row['physical_camera']]
        for key in ['transform_matrix','fl_x','fl_y','cx','cy','w','h','k1','k2','p1','p2']:
            np.testing.assert_array_equal(row[key],teacher[key])
    assert set(synthetic_by_name).isdisjoint({'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'})
    assert read(out/'real/face_roi.json')['ground_truth_sha256']==sha(out/'real'/real['val_filenames'][0])
    # Independent camera-z convention check at sampled mesh hits, including
    # off-axis rays: z converts to Euclidean depth by the pinhole ray norm.
    import open3d as o3d
    from bake_joint_temporal_mesh import camera_depth
    from diffusion_mesh_repair import scene_for
    mesh=o3d.io.read_triangle_mesh(str(out/'mesh/teacher.ply'))
    v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles,np.uint32)
    scene=scene_for(v,t);errors=[]
    for name in ['train_0033','train_0062','val_0000']:
        row=next(p['camera'] for p in request['plan'] if p['id']==name)
        d,ids,b=camera_depth(scene,row);ys,xs=np.where(np.isfinite(d));ys=ys[::1009];xs=xs[::1009]
        weights=np.column_stack([1-b[ys,xs].sum(1),b[ys,xs]])
        point=(v[t[ids[ys,xs]]]*weights[:,:,None]).sum(1)
        pose=np.array(row['transform_matrix']);q=(point-pose[:3,3])@pose[:3,:3]
        z=-q[:,2];norm=np.sqrt(1+((xs+.5-row['cx'])/row['fl_x'])**2+((ys+.5-row['cy'])/row['fl_y'])**2)
        errors.append(float(np.max(np.abs(z-d[ys,xs]))))
        np.testing.assert_allclose(z,d[ys,xs],atol=2e-6)
        np.testing.assert_allclose(np.linalg.norm(point-pose[:3,3],axis=1),d[ys,xs]*norm,atol=2e-6)
    result=dict(status='pass',dataset_hashes_sha256=sha(out/'dataset_hashes.json'),
                parser_splits=reports,pose_depth_same_units=True,camera_z_max_errors=errors,
                all_source_rgb_and_transforms_rehashed=True,real_and_synthetic_train_calibration_exact=True,
                no_training_started=True,confidence_is_not_independent_mvs_evidence=True,
                audit_script_sha256=sha(__file__),python_executable=sys.executable,
                loader_hashes={name:sha(Path(__file__).resolve().parents[2]/name) for name in
                    ['nerfstudio/data/dataparsers/nerfstudio_dataparser.py','nerfstudio/data/utils/data_utils.py']})
    atomic_json(out/'independent_audit.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['panels','parity','audit','heldout','package','smoke','contacts','poses','train-pairs'])
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--views',nargs='+',default=['train_0033','train_0062','val_0000'])
    a=p.parse_args()
    if a.action=='panels':panels(a.output,a.views)
    elif a.action=='parity':
        for name in a.views:parity(a.output,name)
    elif a.action=='heldout':heldout_reference(a.output)
    elif a.action=='package':package(a.output)
    elif a.action=='smoke':parser_smoke(a.output)
    elif a.action=='contacts':contacts(a.output)
    elif a.action=='poses':pose_plot(a.output)
    elif a.action=='train-pairs':train_pairs(a.output,a.views)
    else:audit(a.output)
