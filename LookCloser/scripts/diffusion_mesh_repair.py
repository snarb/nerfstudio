"""Isolated, provenance-first synthetic-view mesh repair experiment.

Generated RGB is an explicit shape prior, never a real camera measurement. See
experiments/dec5_diffusion_mesh_repair.md for hypotheses and acceptance receipts.
"""
from __future__ import annotations
import argparse
import subprocess
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation,Slerp
from joint_temporal_texture import cameras,read,sha,atomic_json,display,exr,CALIBRATION,HELD_CAMERAS,ROOT
from bake_joint_temporal_mesh import camera_depth
from render_patchmatch_camera_path import normalize_frame

BASE=Path('/mnt/data/lookcloser_dec5_5a3_hard_texture_v2/frames/000973')
OUTPUT=Path('/mnt/data/lookcloser_dec5_5a3_diffusion_mesh_repair_000973')


def scene_for(vertices,triangles):
    scene=o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices,np.float32)),o3d.core.Tensor(np.asarray(triangles,np.uint32)))
    return scene


def render_atlas(scene,atlas,texture,row):
    depth,ids,b=camera_depth(scene,row);valid=ids!=np.iinfo(np.uint32).max
    weights=np.column_stack((1-b[valid].sum(1),b[valid]))
    uv=(atlas['uv'][atlas['indices'][ids[valid]]]*weights[...,None]).sum(1)
    hh,ww=texture.shape[:2];u=(uv[:,0]*ww-.5).astype(np.float32);v=((1-uv[:,1])*hh-.5).astype(np.float32)
    colors=[cv2.remap(texture,u[s:s+16000][None],v[s:s+16000][None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]
            for s in range(0,len(u),16000)]
    rgb=np.zeros((row['h'],row['w'],3),np.uint8)
    if colors:rgb[valid]=np.concatenate(colors)
    return rgb,depth,ids,b


def interpolate_camera(left,right,t):
    row=deepcopy(left);a,b=np.asarray(left['transform_matrix']),np.asarray(right['transform_matrix'])
    pose=np.eye(4);pose[:3,:3]=Slerp([0,1],Rotation.from_matrix([a[:3,:3],b[:3,:3]]))(t).as_matrix()
    pose[:3,3]=(1-t)*a[:3,3]+t*b[:3,3];row['transform_matrix']=pose.tolist()
    for key in ['fl_x','fl_y','cx','cy']:row[key]=(1-t)*left[key]+t*right[key]
    row.pop('file_path',None);row['physical_camera']=f'synthetic_{t:.3f}'
    row['synthetic']=True;row['anchors']=[left['physical_camera'],right['physical_camera']];row['blend']=t
    return row


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    rows,mesh,meta=cameras('000973');by_name={r['physical_camera']:r for r in rows}
    atlas=dict(np.load(BASE/'atlas_geometry.npz'));texture=np.asarray(Image.open(BASE/'texture_joint.png').convert('RGB'))
    scene=scene_for(atlas['vertices'],atlas['triangles'])
    cal=read(CALIBRATION);j=normalize_frame(next(r for r in cal['frames'] if r['physical_camera']=='J004_D005_1210TA'),cal,read(meta))
    _,ids,b=camera_depth(scene,j);f=int(ids[310,875]);uv=b[310,875]
    tube=(atlas['vertices'][atlas['triangles'][f]]*np.array([1-uv.sum(),*uv])[:,None]).sum(0)
    query=[interpolate_camera(by_name['I004_D005_1210Q7'],by_name['K004_D005_121016'],t) for t in [.25,.5,.75]]
    request={'frame':'000973','base_mesh':str(mesh),'base_mesh_sha256':sha(mesh),'base_metadata':str(meta),
             'base_metadata_sha256':sha(meta),'base_atlas':str(BASE),'base_texture_sha256':sha(BASE/'texture_joint.png'),
             'calibration_sha256':sha(CALIBRATION),'query_cameras':query,'tube_seed':tube.tolist(),
             'uses_heldout_rgb_for_repair':False,'synthetic_views_are_measurements':False}
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Experiment request mismatch')
    atomic_json(output/'request.json',request)
    from joint_temporal_texture import project
    views=[]
    for i,row in enumerate(query):
        target=output/'views'/f'{i:02d}';target.mkdir(parents=True,exist_ok=True)
        rgb,depth,ids,b=render_atlas(scene,atlas,texture,row)
        Image.fromarray(rgb).save(target/'render_native.png')
        uv,z=project(tube[None],[row]);x,y=np.rint(uv[0,0]).astype(int)
        x0=int(np.clip(x-190,0,1920-512));y0=int(np.clip(y-180,0,1080-512));box=[x0,y0,x0+512,y0+512]
        crop=Image.fromarray(rgb).crop(box).transpose(Image.Transpose.ROTATE_90).resize((1024,1024),Image.Resampling.LANCZOS)
        crop.save(target/'edit_target.png')
        np.savez_compressed(target/'geometry.npz',depth=depth,ids=ids,barycentric=b)
        record={'camera':row,'native_crop_xyxy':box,'crop_rotation_ccw_degrees':90,'crop_resize':2,
                'native_sha256':sha(target/'render_native.png'),'edit_target_sha256':sha(target/'edit_target.png'),
                'image_role':'synthetic_render_edit_target','source_frame':'000973'}
        atomic_json(target/'request.json',record);views.append(record)
        print(f'prepared synthetic_view={i} crop={box}',flush=True)
    # Only original TRAIN images are provided as shape/material references.
    reference_names=['I004_D005_1210Q7','K004_D005_121016','J004_C005_1210I4','I004_C005_1210BA']
    gain=read(ROOT/'exposure.json')['fixed_exposure_gain'];panel=Image.new('RGB',(1024,1024));draw=ImageDraw.Draw(panel)
    references=[]
    for i,name in enumerate(reference_names):
        row=by_name[name];rgb=np.rint(display(exr(row['file_path']),gain)*255).clip(0,255).astype(np.uint8)
        uv,_=project(tube[None],[row]);x,y=np.rint(uv[0,0]).astype(int)
        x0=int(np.clip(x-130,0,1920-256));y0=int(np.clip(y-100,0,1080-256));box=[x0,y0,x0+256,y0+256]
        crop=Image.fromarray(rgb).crop(box).transpose(Image.Transpose.ROTATE_90).resize((512,488))
        panel.paste(crop,((i%2)*512,(i//2)*512+24));draw.text(((i%2)*512+4,(i//2)*512+4),name,fill='white')
        references.append({'camera':name,'source_sha256':sha(row['file_path']),'native_crop_xyxy':box})
    panel.save(output/'train_references.png')
    atomic_json(output/'prepared.json',{'status':'prepared_for_imagegen','views':views,'references':references,
                'train_references_sha256':sha(output/'train_references.png'),'script_sha256':sha(__file__)})


def edit_masks(output):
    polygons=[
        {'tube':[[291,612],[368,602],[418,635],[415,796],[373,842],[301,835]],
         'chin':[[478,838],[558,895],[695,923],[864,888],[875,942],[707,962],[547,941],[478,905]]},
        {'tube':[[291,611],[367,603],[419,643],[416,796],[369,840],[301,835]],
         'chin':[[526,846],[588,889],[729,925],[925,889],[938,941],[745,965],[575,940],[527,905]]},
        {'tube':[[291,610],[368,603],[427,638],[428,795],[374,844],[300,835]],
         'chin':[[579,847],[663,897],[803,925],[1008,874],[1023,931],[825,965],[657,943],[580,907]]}]
    for i,regions in enumerate(polygons):
        target=output/'views'/f'{i:02d}';mask=Image.new('L',(1024,1024));draw=ImageDraw.Draw(mask)
        for points in regions.values():draw.polygon([tuple(p) for p in points],fill=255)
        mask.save(target/'edit_mask.png')
        atomic_json(target/'edit_regions.json',{'regions':regions,'role':'manual_mask_on_render_not_heldout_rgb',
                     'mask_sha256':sha(target/'edit_mask.png'),'target_sha256':sha(target/'edit_target.png')})


def register_generated(base,generated,mask):
    """Fit only one global similarity outside the edit area; no optical-flow cheat."""
    generated=cv2.resize(generated,(base.shape[1],base.shape[0]),interpolation=cv2.INTER_AREA)
    stable=(cv2.dilate(mask,np.ones((41,41),np.uint8))==0).astype(np.uint8)*255
    stable[np.max(base,axis=2)<20]=0
    sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.015)
    ka,da=sift.detectAndCompute(cv2.cvtColor(generated,cv2.COLOR_RGB2GRAY),stable)
    kb,db=sift.detectAndCompute(cv2.cvtColor(base,cv2.COLOR_RGB2GRAY),stable)
    if da is None or db is None:raise ValueError('No stable registration features')
    pairs=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance<.72*n.distance]
    if len(pairs)<20:raise ValueError('Insufficient non-edit registration matches')
    a=np.float32([ka[m.queryIdx].pt for m in pairs]);b=np.float32([kb[m.trainIdx].pt for m in pairs])
    affine,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=5000)
    if affine is None or int(inliers.sum())<20:raise ValueError('Generated image registration failed')
    error=np.linalg.norm(a@affine[:,:2].T+affine[:,2]-b,axis=1)[inliers[:,0]>0]
    scale=np.linalg.norm(affine[:,0]);translation=np.linalg.norm(affine[:,2])
    if abs(scale-1)>.03 or translation>25 or np.median(error)>1.5:raise ValueError('Generated camera drift too large')
    aligned=cv2.warpAffine(generated,affine,(base.shape[1],base.shape[0]),flags=cv2.INTER_LINEAR)
    return aligned,{'similarity_generated_to_target':affine.tolist(),'matches':len(pairs),
                    'inliers':int(inliers.sum()),'inlier_error_median_crop_pixels':float(np.median(error)),
                    'scale':float(scale),'translation_crop_pixels':float(translation),
                    'outside_mask_raw_mean_absolute_rgb_difference':float(np.abs(generated.astype(float)-base)[mask==0].mean()),
                    'camera_poses_changed':False,'local_nonrigid_alignment':False}


def import_generated(output):
    records=[]
    for i in range(3):
        target=output/'views'/f'{i:02d}';request=read(target/'request.json')
        base=np.asarray(Image.open(target/'edit_target.png').convert('RGB'))
        raw=np.asarray(Image.open(target/'generated_raw.png').convert('RGB'))
        mask=np.asarray(Image.open(target/'edit_mask.png').convert('L'))
        aligned,record=register_generated(base,raw,mask)
        Image.fromarray(aligned).save(target/'generated_aligned.png')
        native=np.asarray(Image.open(target/'render_native.png').convert('RGB')).copy()
        x0,y0,x1,y1=request['native_crop_xyxy']
        small=np.rot90(cv2.resize(aligned,(512,512),interpolation=cv2.INTER_AREA),-1)
        localmask=np.rot90(cv2.resize(mask,(512,512),interpolation=cv2.INTER_NEAREST),-1)>0
        alpha=np.minimum(cv2.distanceTransform(localmask.astype(np.uint8),cv2.DIST_L2,5)/2.,1.)
        crop=native[y0:y1,x0:x1].copy()
        edited=np.rint(crop*(1-alpha[...,None])+small*alpha[...,None]).astype(np.uint8)
        if not np.array_equal(edited[~localmask],crop[~localmask]):raise ValueError('Outside-mask edit leakage')
        native[y0:y1,x0:x1]=edited
        Image.fromarray(native).save(target/'synthetic_masked_native.png')
        Image.fromarray((localmask*255).astype(np.uint8)).save(target/'mask_native_crop.png')
        preview=Image.fromarray(edited).transpose(Image.Transpose.ROTATE_90).resize((1024,1024))
        preview.save(target/'synthetic_masked_preview.png')
        record.update({'view':i,'generated_raw_sha256':sha(target/'generated_raw.png'),
                       'mask_sha256':sha(target/'edit_mask.png'),'synthetic_masked_sha256':sha(target/'synthetic_masked_native.png'),
                       'outside_mask_native_max_difference':0,'status':'synthetic_prior_not_real_observation'})
        atomic_json(target/'generation_receipt.json',record);records.append(record)
        print(f'generated view={i} registration_median={record["inlier_error_median_crop_pixels"]:.3f} crop_pixels',flush=True)
    atomic_json(output/'generation_manifest.json',{'tool':'built-in imagegen','views':records,
                'prompt_files':[f'views/{i:02d}/prompt.txt' for i in range(3)],'all_generated_rgb_is_synthetic':True})


def prepare_mvs(output,include_synthetic=True):
    from joint_temporal_texture import project
    request=read(output/'request.json');tube=np.array(request['tube_seed']);rows,_,_=cameras('000973')
    by_name={r['physical_camera']:r for r in rows};gains=read(ROOT/'camera_profiles.json')
    response=dict(zip(gains['physical_cameras'],np.array(gains['rgb_gain'])))
    gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    data=output/'mvs'/'data';data.mkdir(parents=True,exist_ok=True);frames=[];provenance=[]
    real_names=['I004_D005_1210Q7','K004_D005_121016','J004_C005_1210I4','I004_C005_1210BA',
                'K004_C005_1210BC','J004_B005_1210GR','H004_D005_1210SF','L004_D005_1210T4']
    for i,name in enumerate(real_names):
        row=deepcopy(by_name[name]);uv,_=project(tube[None],[row]);x,y=np.rint(uv[0,0]).astype(int)
        x0=int(np.clip(x-190,0,1920-512));y0=int(np.clip(y-180,0,1080-512))
        rgb=np.rint(display(exr(row['file_path'])*response[name],gain)*255).clip(0,255).astype(np.uint8)
        file=f'frame_train_real_{i:02d}.jpg';Image.fromarray(rgb[y0:y0+512,x0:x0+512]).save(data/file,quality=98,subsampling=0)
        provenance.append({'file':file,'kind':'real_train_rgb','physical_camera':name,'source_sha256':sha(row['file_path']),
                           'native_crop_xyxy':[x0,y0,x0+512,y0+512],'image_sha256':sha(data/file)})
        row.update(file_path=file,w=512,h=512,cx=row['cx']-x0,cy=row['cy']-y0);row.pop('colmap_im_id',None);frames.append(row)
    for i in range(3 if include_synthetic else 0):
        target=output/'views'/f'{i:02d}';r=read(target/'request.json');row=deepcopy(r['camera']);x0,y0,x1,y1=r['native_crop_xyxy']
        rgb=Image.open(target/'synthetic_masked_native.png');file=f'frame_train_synthetic_{i:02d}.jpg'
        rgb.crop((x0,y0,x1,y1)).save(data/file,quality=98,subsampling=0)
        row.update(file_path=file,w=512,h=512,cx=row['cx']-x0,cy=row['cy']-y0);row.pop('colmap_im_id',None);frames.append(row)
        provenance.append({'file':file,'kind':'masked_diffusion_prior_on_mesh_render','source_sha256':sha(target/'synthetic_masked_native.png'),
                           'native_crop_xyxy':[x0,y0,x1,y1],'image_sha256':sha(data/file),'independent_measurement':False})
    payload={'camera_model':'OPENCV','frames':frames,'train_filenames':[r['file_path'] for r in frames],
             'val_filenames':[],'test_filenames':[],'coordinate_system':'original_mesh_normalized','synthetic_prior_count':3 if include_synthetic else 0}
    atomic_json(data/'transforms.json',payload)
    atomic_json(output/'mvs'/'input_manifest.json',{'real_train_views':8,'synthetic_prior_views':3 if include_synthetic else 0,'sources':provenance,
                'heldout_rgb_used':False,'poses_optimized':False,'hypothesis':'localized_generated_views_plus_real_train_fixed_camera_stereo'})


def run_mvs(output):
    from export_nerfstudio_colmap_model import export_model
    mvs=output/'mvs';data=mvs/'data';model=mvs/'model';dense=mvs/'dense';logs=mvs/'logs';logs.mkdir(exist_ok=True)
    colmap='/home/brans/lookcloser_temp/colmap_5509fffe_dev3_bundle/colmap_pinned'
    version=subprocess.check_output([colmap,'-h'],text=True)
    if not all(s in version for s in ['COLMAP 3.13.0.dev0','5509fffe','with CUDA']):raise ValueError('Unverified COLMAP build')
    (mvs/'colmap_version.txt').write_text(version)
    if not model.exists():export_model(data,model,split='train',camera_model='PINHOLE')
    def run(name,args):
        print(f'mvs stage={name} started',flush=True);start=time.monotonic()
        with (logs/f'{name}.log').open('w') as f:subprocess.run([colmap,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
        atomic_json(logs/f'{name}.json',{'seconds':time.monotonic()-start,'command':[colmap,*args],'log_sha256':sha(logs/f'{name}.log')})
        print(f'mvs stage={name} complete seconds={time.monotonic()-start:.1f}',flush=True)
    if not dense.exists():run('undistort',['image_undistorter','--image_path',str(data),'--input_path',str(model),
          '--output_path',str(dense),'--output_type','COLMAP','--max_image_size','512','--copy_policy','copy'])
    names=read(data/'transforms.json')['train_filenames']
    (dense/'stereo'/'patch-match.cfg').write_text('\n'.join(line for name in names for line in [name,', '.join(n for n in names if n!=name)])+'\n')
    scale=read(read(output/'request.json')['base_metadata'])['dataparser_scale']
    common=['patch_match_stereo','--workspace_path',str(dense),'--workspace_format','COLMAP',
            '--PatchMatchStereo.max_image_size','512','--PatchMatchStereo.gpu_index','0',
            '--PatchMatchStereo.depth_min',str(4.5*scale),'--PatchMatchStereo.depth_max',str(20.*scale),
            '--PatchMatchStereo.num_iterations','3','--PatchMatchStereo.write_consistency_graph','1']
    run('photometric',common+['--PatchMatchStereo.geom_consistency','0','--PatchMatchStereo.filter','0'])
    run('geometric',common+['--PatchMatchStereo.geom_consistency','1','--PatchMatchStereo.geom_consistency_max_cost','6',
            '--PatchMatchStereo.filter','1','--PatchMatchStereo.filter_min_ncc','.1',
            '--PatchMatchStereo.filter_min_triangulation_angle','1','--PatchMatchStereo.filter_min_num_consistent','2',
            '--PatchMatchStereo.filter_geom_consistency_max_cost','2'])
    summarize_mvs(output)


def summarize_mvs(output):
    from import_colmap_mvs_depth_dataset import read_colmap_dense_array
    mvs=output/'mvs';dense=mvs/'dense';names=read(mvs/'data'/'transforms.json')['train_filenames']
    stats=[]
    for name in names:
        path=dense/'stereo'/'depth_maps'/f'{name}.geometric.bin';depth=read_colmap_dense_array(path)[...,0]
        if depth.shape!=(512,512) or not np.isfinite(depth).all():raise ValueError('Invalid local MVS depth')
        record={'name':name,'coverage':float((depth>0).mean()),'depth_sha256':sha(path)}
        if 'synthetic' in name:
            i=int(Path(name).stem[-2:]);target=output/'views'/f'{i:02d}'
            mask=np.asarray(Image.open(target/'mask_native_crop.png'))>0
            record['edit_region_coverage']=float((depth[mask]>0).mean())
            boxes=read(target/'edit_regions.json')['regions'];parts={}
            for label,points in boxes.items():
                part=Image.new('L',(1024,1024));ImageDraw.Draw(part).polygon([tuple(p) for p in points],fill=255)
                part=np.rot90(cv2.resize(np.asarray(part),(512,512),interpolation=cv2.INTER_NEAREST),-1)>0
                parts[label]={'positive_depth_fraction':float((depth[part]>0).mean()),'pixels':int(part.sum())}
            record['regions']=parts
            preview=cv2.applyColorMap(np.rint(np.clip((depth-.55)/.55,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)[...,::-1]
            preview[depth<=0]=0;Image.fromarray(np.rot90(preview)).resize((1024,1024)).save(target/'mvs_depth_preview.png')
        stats.append(record)
    atomic_json(mvs/'result.json',{'status':'depths_completed_not_accepted','depths':stats,'real_train_views':8,
                'synthetic_prior_views':sum('synthetic' in name for name in names)})


def fuse_mvs(output):
    from import_colmap_mvs_depth_dataset import read_colmap_dense_array,load_binary_pinhole_calibration
    mvs=output/'mvs';dense=mvs/'dense';rows=read(mvs/'data'/'transforms.json')['frames']
    calibration=load_binary_pinhole_calibration(dense/'sparse')
    # Camera J/B is farther from the subject (>1.1 normalized depth). The first
    # exploratory crop range was too narrow; keep the original recipe's range
    # converted into this already-normalized coordinate system for every camera.
    device=o3d.core.Device('CUDA:0');voxel=.00025;trunc=.0015
    depth_max=20.*read(read(output/'request.json')['base_metadata'])['dataparser_scale']
    grid=o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf','weight'),attr_dtypes=(o3d.core.float32,o3d.core.float32),
              attr_channels=((1,),(1,)),voxel_size=voxel,block_resolution=16,block_count=50000,device=device)
    observations=[];blocks=[]
    for row in rows:
        print(f'discover_blocks={row["file_path"]}',flush=True)
        for key,value in calibration[row['file_path']].items():
            if not np.isclose(row[key],value,atol=1e-7):raise ValueError('Cropped calibration changed during undistortion')
        depth=read_colmap_dense_array(dense/'stereo'/'depth_maps'/f'{row["file_path"]}.geometric.bin')[...,0]
        image=o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(depth),device=device))
        intrinsic=o3d.core.Tensor(np.array([[row['fl_x'],0,row['cx']],[0,row['fl_y'],row['cy']],[0,0,1]],np.float64))
        extrinsic=o3d.core.Tensor(np.linalg.inv(np.array(row['transform_matrix'])@np.diag([1.,-1.,-1.,1.])))
        coords=grid.compute_unique_block_coordinates(image,intrinsic,extrinsic,depth_scale=1.,depth_max=depth_max,trunc_voxel_multiplier=trunc/voxel)
        blocks.append(coords.cpu().numpy());observations.append((image,intrinsic,extrinsic))
    union=np.unique(np.concatenate(blocks),axis=0);coords=o3d.core.Tensor(union,device=device)
    for i,(image,intrinsic,extrinsic) in enumerate(observations):
        grid.integrate(coords,image,intrinsic,extrinsic,depth_scale=1.,depth_max=depth_max,trunc_voxel_multiplier=trunc/voxel)
        print(f'synthetic_prior_fusion view={i+1}/{len(rows)}',flush=True)
    mesh=grid.extract_triangle_mesh(weight_threshold=2.).to_legacy();mesh.compute_vertex_normals()
    target=mvs/'local_mesh.ply';o3d.io.write_triangle_mesh(str(target),mesh)
    atomic_json(mvs/'fusion.json',{'mesh':str(target),'mesh_sha256':sha(target),'vertices':len(mesh.vertices),
                 'triangles':len(mesh.triangles),'voxel':voxel,'truncation':trunc,'extraction_weight':2,
                 'integrate_full_block_union':True,'allocated_blocks':len(union),'synthetic_prior_views':sum('synthetic' in r['file_path'] for r in rows),
                 'real_train_views':8,'device':str(device),'raw_tsdf_volume_saved':False,'status':'local_mesh_requires_visual_gate'})


def real_control(output):
    control=output/'real_only_control';control.mkdir(exist_ok=True)
    atomic_json(control/'request.json',read(output/'request.json'))
    prepare_mvs(control,include_synthetic=False);run_mvs(control);fuse_mvs(control)


def compare_mvs(output):
    """Matched eight-real-view control separates crop/subset effects from editing."""
    candidates={'original_62':read(output/'request.json')['base_mesh'],
                'real_8':str(output/'real_only_control/mvs/local_mesh.ply'),
                'real_8_synthetic_3':str(output/'mvs/local_mesh.ply')}
    loaded={};records=[]
    for label,path in candidates.items():
        mesh=o3d.io.read_triangle_mesh(path);mesh.compute_triangle_normals()
        loaded[label]=(scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles)),np.asarray(mesh.triangle_normals))
    for i in range(3):
        target=output/'views'/f'{i:02d}';vr=read(target/'request.json');box=vr['native_crop_xyxy']
        panel=Image.new('RGB',(768*3,768+26));draw=ImageDraw.Draw(panel)
        regions=read(target/'edit_regions.json')['regions']
        for j,(label,(scene,normals)) in enumerate(loaded.items()):
            d,ids,_=camera_depth(scene,vr['camera']);valid=np.isfinite(d)
            rgb=np.zeros((1080,1920,3),np.uint8);rgb[valid]=np.rint((normals[ids[valid]]*.5+.5)*255).clip(0,255).astype(np.uint8)
            crop=Image.fromarray(rgb).crop(box).transpose(Image.Transpose.ROTATE_90).resize((768,768))
            panel.paste(crop,(j*768,26));draw.text((j*768+4,5),label,fill='white')
            coverage=np.asarray(Image.fromarray(valid).crop(box).transpose(Image.Transpose.ROTATE_90).resize((1024,1024),Image.Resampling.NEAREST))
            stats={}
            for name,polygon in regions.items():
                mask=Image.new('L',(1024,1024));ImageDraw.Draw(mask).polygon([tuple(p) for p in polygon],fill=255)
                stats[name]=float(coverage[np.asarray(mask)>0].mean())
            records.append({'view':i,'variant':label,'render_hit_fraction_in_edit_region':stats})
        panel.save(target/'matched_mvs_control.png')
    atomic_json(output/'matched_mvs_control.json',{'mesh_sha256':{k:sha(v) for k,v in candidates.items()},
        'records':records,'hit_is_not_surface_correctness':True,'comparison_real8_vs_real8_synthetic3_has_matched_real_inputs':True})


def inspect_geometry(output):
    from local_mesh_repair import boundary_loops,mask_votes,project_crop
    atlas=dict(np.load(BASE/'atlas_geometry.npz'));vertices=atlas['vertices'];triangles=atlas['triangles'];centers=vertices[triangles].mean(1)
    views=[]
    for i in range(3):
        target=output/'views'/f'{i:02d}';r=read(target/'request.json')
        views.append((r['camera'],r['native_crop_xyxy'],read(target/'edit_regions.json')['regions']))
    tube_votes=mask_votes(centers,views,'tube');chin_votes=mask_votes(centers,views,'chin')
    # Freeze real-depth evidence from the earlier independently audited 62-view
    # native-footprint pass. It refers to exactly the same original face inventory.
    evidence_root=Path('/mnt/data/lookcloser_dec5_5a3_surface_repair/supported_shell_control/mesh')
    receipt=read(evidence_root/'carving_request.json')
    if receipt['mesh_sha256']!=read(output/'request.json')['base_mesh_sha256']:raise ValueError('Real-depth evidence mesh mismatch')
    evidence=dict(np.load(evidence_root/'triangle_evidence.npz'))
    print('evidence_arrays='+str({k:v.shape for k,v in evidence.items()}),flush=True)
    near=evidence['near_counts'];free=evidence['free_counts']
    tube=tube_votes>=2;chin=chin_votes>=2
    np.savez_compressed(output/'region_evidence.npz',tube_faces=tube,chin_faces=chin,near_counts=near,free_counts=free)
    loops,rejected=boundary_loops(triangles);records=[]
    for i,loop in enumerate(loops):
        points=vertices[loop];fraction=float((mask_votes(points,views,'chin')>=2).mean())
        if fraction<=0:continue
        records.append({'loop':i,'vertices':loop.tolist(),'edge_count':len(loop),'chin_mask_fraction':fraction,
                        'diameter':float(np.linalg.norm(np.ptp(points,axis=0))),'center':points.mean(0).tolist()})
    def counts(mask):return {'triangles':int(mask.sum()),'strict_real_support_lt2':int((mask&(near<2)).sum()),
                        'unsupported_and_contradicted':int((mask&(near<2)&(free>=3)).sum()),'support_median':float(np.median(near[mask]))}
    atomic_json(output/'geometry_diagnosis.json',{'tube':counts(tube),'chin':counts(chin),'chin_boundary_loops':records,
                'all_boundary_loops':len(loops),'non_simple_boundary_components':len(rejected),
                'real_depth_evidence_path':str(evidence_root),'real_depth_evidence_sha256':sha(evidence_root/'triangle_evidence.npz'),
                'uses_generated_depth_as_real_evidence':False})
    for i,(row,box,_) in enumerate(views):
        overlay=Image.open(output/'views'/f'{i:02d}'/'edit_target.png').convert('RGB');draw=ImageDraw.Draw(overlay)
        for record in records:
            loop=np.array(record['vertices']);uv,_=project_crop(vertices[loop],row,box)
            valid=mask_votes(vertices[loop],views,'chin')>=2
            for k,(x,y) in enumerate(uv):
                color='red' if valid[k] else 'lime'
                draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
                if k%12==0:draw.text((x+3,y+3),str(k),fill='yellow')
        overlay.save(output/'views'/f'{i:02d}'/'chin_boundary_overlay.png')
    base_scene=scene_for(vertices,triangles);local=o3d.io.read_triangle_mesh(str(output/'mvs'/'local_mesh.ply'))
    local.compute_triangle_normals();base_normals=np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],vertices[triangles[:,2]]-vertices[triangles[:,0]])
    base_normals/=np.linalg.norm(base_normals,axis=1)[:,None].clip(1e-12)
    local_scene=scene_for(np.asarray(local.vertices),np.asarray(local.triangles));local_normals=np.asarray(local.triangle_normals)
    for i,(row,box,_) in enumerate(views):
        target=output/'views'/f'{i:02d}';panel=Image.new('RGB',(1024*2,1024+25));draw=ImageDraw.Draw(panel)
        for j,(label,scene,normals) in enumerate([('original TSDF surface',base_scene,base_normals),('real + generated stereo surface',local_scene,local_normals)]):
            d,ids,b=camera_depth(scene,row);valid=np.isfinite(d)
            rgb=np.zeros((1080,1920,3),np.uint8)
            rgb[valid]=np.rint((normals[ids[valid]]*.5+.5)*255).clip(0,255).astype(np.uint8)
            crop=Image.fromarray(rgb).crop(box).transpose(Image.Transpose.ROTATE_90).resize((1024,1024))
            panel.paste(crop,(j*1024,25));draw.text((j*1024+5,5),label,fill='white')
        panel.save(target/'geometry_comparison.png')


def local_repair(output):
    from local_mesh_repair import close_selected_holes
    atlas=dict(np.load(BASE/'atlas_geometry.npz'));vertices=atlas['vertices'];triangles=atlas['triangles']
    diagnosis=read(output/'geometry_diagnosis.json');evidence=np.load(output/'region_evidence.npz')
    # Explicit reviewed loop IDs are tied to the immutable mesh and retained
    # overlays, not a hidden generic face/neck rule. No other hole is filled.
    loops=[np.array(r['vertices']) for r in diagnosis['chin_boundary_loops'] if r['loop'] in [103,105]]
    if list(map(len,loops))!=[228,3]:raise ValueError('Reviewed chin boundary inventory changed')
    v,t,fill=close_selected_holes(vertices,triangles,loops)
    remove=evidence['tube_faces']&(evidence['near_counts']<2)&(evidence['free_counts']>=3)
    keep=np.r_[~remove,np.ones(len(t)-len(triangles),bool)];t=t[keep]
    target=output/'local_repair';target.mkdir(exist_ok=True)
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t));mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(target/'mesh.ply'),mesh)
    np.save(target/'base_face_ids.npy',np.r_[np.flatnonzero(~remove),np.full(fill['added_triangles'],-1)])
    atomic_json(target/'operations.json',{'base_mesh_sha256':read(output/'request.json')['base_mesh_sha256'],
                 'output_mesh_sha256':sha(target/'mesh.ply'),'coordinate_system_unchanged':True,
                 'chin_hole_fill':fill,'removed_tube_triangles':int(remove.sum()),
                 'removal_rule':'within reviewed tube masks in at least 2 views AND fewer than 2 real depth supports AND at least 3 real free-space contradictions',
                 'synthetic_rgb_used_as_real_depth':False,'status':'candidate_requires_texture_and_visual_gate'})
    # Immediate geometry-only review before baking expensive full-resolution UV.
    scene=scene_for(v,t);mesh.compute_triangle_normals();normals=np.asarray(mesh.triangle_normals)
    for i in range(3):
        vr=read(output/'views'/f'{i:02d}'/'request.json');d,ids,b=camera_depth(scene,vr['camera']);valid=np.isfinite(d)
        rgb=np.zeros((1080,1920,3),np.uint8);rgb[valid]=np.rint((normals[ids[valid]]*.5+.5)*255).clip(0,255).astype(np.uint8)
        Image.fromarray(rgb).crop(vr['native_crop_xyxy']).transpose(Image.Transpose.ROTATE_90).resize((1024,1024)).save(target/f'geometry_{i:02d}.png')


def cylinder_repair(output):
    from local_mesh_repair import project_crop,fit_cylinder
    atlas=dict(np.load(BASE/'atlas_geometry.npz'));centers=atlas['vertices'][atlas['triangles']].mean(1)
    evidence=np.load(output/'region_evidence.npz');views=[];observations=[];middle_votes=np.zeros(len(centers),np.uint8)
    metal_votes=np.zeros(len(centers),np.uint8);visible_votes=np.zeros(len(centers),np.uint8)
    for i in range(3):
        target=output/'views'/f'{i:02d}';r=read(target/'request.json');regions=read(target/'edit_regions.json')['regions']
        views.append((r['camera'],r['native_crop_xyxy'],regions))
        image=np.asarray(Image.open(target/'generated_aligned.png').convert('RGB')).astype(np.float32)
        roi=Image.new('L',(1024,1024));ImageDraw.Draw(roi).polygon([tuple(p) for p in regions['tube']],fill=255)
        maximum=image.max(2);minimum=image.min(2)
        gray=((maximum-minimum)<maximum*.23)&(maximum>65)&(np.asarray(roi)>0)
        gray=cv2.morphologyEx(gray.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        n,labels,stats,_=cv2.connectedComponentsWithStats(gray,8)
        if n<2:raise ValueError('No generated silver casing region')
        gray=labels==(1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        yy,xx=np.where(gray);samples=[]
        for y in range(650,751):
            xs=np.flatnonzero(gray[y])
            if len(xs)>=10:samples.append([y,(xs.min()+xs.max())/2,(xs.max()-xs.min())/2])
        samples=np.array(samples)
        if len(samples)<60:raise ValueError('Insufficient cylinder silhouette samples')
        observation={'centerline_x_of_y':np.polyfit(samples[:,0],samples[:,1],1).tolist(),
                     'median_half_width':float(np.median(samples[:,2])),'top_y':float(np.quantile(yy,.002)),
                     'source':'generated image silhouette, shape prior only'}
        observations.append(observation);Image.fromarray((gray*255).astype(np.uint8)).save(target/'generated_tube_mask.png')
        uv,z=project_crop(centers,r['camera'],r['native_crop_xyxy'])
        from joint_temporal_texture import project
        native_uv,native_z=project(centers,[r['camera']]);native_uv=native_uv[0]
        native_depth=np.load(target/'geometry.npz')['depth'];native_depth=np.where(np.isfinite(native_depth),native_depth,0)
        for start in range(0,len(centers),16000):
            sample=cv2.remap(native_depth,native_uv[start:start+16000,0][None],native_uv[start:start+16000,1][None],cv2.INTER_NEAREST)[0]
            visible_votes[start:start+16000]+=(sample>0)&(np.abs(sample-native_z[0,start:start+16000])<.0006)
        middle_votes+=(uv[:,1]>=650)&(uv[:,1]<=750)&(uv[:,0]>=290)&(uv[:,0]<=430)&(z>0)
        for start in range(0,len(uv),16000):
            sample=cv2.remap(gray.astype(np.uint8),uv[start:start+16000,0][None],uv[start:start+16000,1][None],cv2.INTER_NEAREST)[0]
            metal_votes[start:start+16000]+=sample
    seed=np.array(read(output/'request.json')['tube_seed'])
    selected=evidence['tube_faces']&(evidence['near_counts']>=3)&(middle_votes>=2)&(visible_votes>=2)&(np.linalg.norm(centers-seed,axis=1)<.008)
    if selected.sum()<30:raise ValueError('Insufficient real supported shaft points')
    fit=fit_cylinder(centers[selected],views,observations)
    atomic_json(output/'cylinder_fit_diagnostic.json',{'fit':fit,'point_bbox':[centers[selected].min(0).tolist(),centers[selected].max(0).tolist()],
                'observations':observations,'selected_points':int(selected.sum())})
    if not fit['passes_real_radial_gate']:raise ValueError('Cylinder does not agree with visible observed metal surface')
    center=np.array(fit['center']);axis=np.array(fit['axis']);radius=fit['radius']
    delta=centers-center;along=delta@axis;radial=np.linalg.norm(delta-along[:,None]*axis,axis=1)
    region=evidence['tube_faces']&(along>=fit['lower']-radius*.3)&(along<=fit['upper']+radius*.5)&(radial<radius*5)
    remove=region&((evidence['near_counts']<2)|(metal_votes>=2))
    # First candidate already includes the explicitly selected chin fill.
    prior=output/'local_repair';mesh=o3d.io.read_triangle_mesh(str(prior/'mesh.ply'))
    v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles);base_ids=np.load(prior/'base_face_ids.npy')
    keep=(base_ids<0)|~remove[np.maximum(base_ids,0)];t=t[keep];base_ids=base_ids[keep]
    import trimesh
    cylinder=trimesh.creation.cylinder(radius=radius,height=fit['upper']-fit['lower'],sections=96)
    e1=np.cross(axis,[0.,0.,1.]);e1/=np.linalg.norm(e1);e2=np.cross(axis,e1)
    cv=cylinder.vertices@np.column_stack((e1,e2,axis)).T+center+axis*(fit['upper']+fit['lower'])/2
    ct=cylinder.faces+len(v);v=np.concatenate((v,cv));t=np.concatenate((t,ct));base_ids=np.r_[base_ids,np.full(len(ct),-1)]
    target=output/'cylinder_repair';target.mkdir(exist_ok=True)
    result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t));result.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(target/'mesh.ply'),result);np.save(target/'base_face_ids.npy',base_ids)
    atomic_json(target/'operations.json',{'base_mesh_sha256':read(output/'request.json')['base_mesh_sha256'],
                 'output_mesh_sha256':sha(target/'mesh.ply'),'coordinate_system_unchanged':True,
                 'cylinder_fit':fit,'synthetic_silhouette_observations':observations,'removed_original_triangles':int(remove.sum()),
                 'cylinder_triangles':len(ct),'previous_operations_sha256':sha(prior/'operations.json'),
                 'all_original_vertex_coordinates_unchanged':True,'status':'candidate_requires_real_view_validation'})


def refine_cylinder(output):
    """Reviewed endpoint correction, not a change to the fitted shaft radius/axis.

    Real I/D, K/D and H/C crops show the first cylinder stops above the finger
    occlusion and leaves the old jagged cap. Extend into that occluded contact;
    replace only the small cap neighborhood. Preserve the rejected first mesh.
    """
    import trimesh
    prior=output/'cylinder_repair';ops=read(prior/'operations.json');fit=deepcopy(ops['cylinder_fit'])
    center=np.array(fit['center']);axis=np.array(fit['axis']);radius=fit['radius']
    mesh=o3d.io.read_triangle_mesh(str(prior/'mesh.ply'));v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
    base_ids=np.load(prior/'base_face_ids.npy');t=t[:-ops['cylinder_triangles']];base_ids=base_ids[:-ops['cylinder_triangles']]
    original=dict(np.load(BASE/'atlas_geometry.npz'));centers=original['vertices'][original['triangles']].mean(1)
    delta=centers-center;along=delta@axis;radial=np.linalg.norm(delta-along[:,None]*axis,axis=1)
    cap=(np.abs(along-fit['upper'])<.0009)&(radial<1.8*radius)
    keep=(base_ids<0)|~cap[np.maximum(base_ids,0)];removed=int((~keep).sum());t=t[keep];base_ids=base_ids[keep]
    fit['lower']-=.0015;fit['upper']+=.0003
    cylinder=trimesh.creation.cylinder(radius=radius,height=fit['upper']-fit['lower'],sections=96)
    e1=np.cross(axis,[0.,0.,1.]);e1/=np.linalg.norm(e1);e2=np.cross(axis,e1)
    cv=cylinder.vertices@np.column_stack((e1,e2,axis)).T+center+axis*(fit['upper']+fit['lower'])/2
    ct=cylinder.faces+len(v);v=np.concatenate((v,cv));t=np.concatenate((t,ct));base_ids=np.r_[base_ids,np.full(len(ct),-1)]
    target=output/'cylinder_refined';target.mkdir(exist_ok=True)
    result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t));result.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(target/'mesh.ply'),result);np.save(target/'base_face_ids.npy',base_ids)
    atomic_json(target/'operations.json',dict(ops,output_mesh_sha256=sha(target/'mesh.ply'),cylinder_fit=fit,
        previous_operations_sha256=sha(prior/'operations.json'),additional_old_cap_triangles_removed=removed,
        manual_endpoint_prior={'upper_extension_normalized':.0003,'occluded_lower_extension_normalized':.0015,
            'reviewed_train_cameras':['I004_D005_1210Q7','K004_D005_121016','H004_C005_1210SZ'],
            'observed_backside_claimed':False},status='candidate_requires_real_view_validation'))


def object_completion(output):
    """Remove residual wall only with independent free-space evidence.

    The first geometry-only object-volume cut touched a supported finger. The
    corrected version protects EVERY face with two near observations, even if
    it falls inside the cylinder's projected edit region.
    """
    prior=output/'cylinder_refined';ops=read(prior/'operations.json');fit=ops['cylinder_fit']
    mesh=o3d.io.read_triangle_mesh(str(prior/'mesh.ply'));v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
    ids=np.load(prior/'base_face_ids.npy');center=np.array(fit['center']);axis=np.array(fit['axis'])
    cc=v[t].mean(1)-center;along=cc@axis;radial=np.linalg.norm(cc-along[:,None]*axis,axis=1)
    remove=(ids>=0)&(along>-.0025)&(along<fit['upper']+.0009)&(radial<3*fit['radius'])
    from local_mesh_repair import conservative_depth_removal
    evidence=np.load(output/'region_evidence.npz');proposed=int(remove.sum())
    remove=conservative_depth_removal(remove,ids,evidence['near_counts'],evidence['free_counts'])
    target=output/'object_completion_supported';target.mkdir(exist_ok=True)
    mesh.triangles=o3d.utility.Vector3iVector(t[~remove]);mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(target/'mesh.ply'),mesh);np.save(target/'base_face_ids.npy',ids[~remove])
    atomic_json(target/'operations.json',dict(ops,output_mesh_sha256=sha(target/'mesh.ply'),
        previous_operations_sha256=sha(prior/'operations.json'),object_prior_wall_removal={
            'removed_original_triangles':int(remove.sum()),'along_minimum':-.0025,
            'along_maximum':fit['upper']+.0009,'radial_maximum':3*fit['radius'],
            'independent_depth_evidence_required':True,'minimum_free_views':3,'maximum_near_views':1,
            'protected_proposed_triangles':proposed-int(remove.sum()),'finger_contact_interval_protected':True},
        status='object_prior_candidate_requires_real_view_validation'))


def main():
    actions={'prepare':prepare,'edit-masks':edit_masks,'import-generated':import_generated,'prepare-mvs':prepare_mvs,
             'run-mvs':run_mvs,'summarize-mvs':summarize_mvs,'fuse-mvs':fuse_mvs}
    actions['inspect-geometry']=inspect_geometry
    actions['real-control']=real_control
    actions['compare-mvs']=compare_mvs
    actions['local-repair']=local_repair
    actions['cylinder-repair']=cylinder_repair
    actions['refine-cylinder']=refine_cylinder
    actions['object-completion']=object_completion
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=actions)
    p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    actions[a.action](a.output)


if __name__=='__main__':main()
