"""One-camera semantic lip texture control; mesh/other pixels stay unchanged.

Train-only landmarks define the surface region. Among three central train
cameras with >=99% centroid visibility, choose the smallest calibration angle
to the target. Every changed output pixel must still pass native visibility.
"""
from pathlib import Path
from copy import deepcopy
import numpy as np
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json, cameras, project, display, ROOT as COLOR
from infer_train_lip_regions import ROOT as MASKS, CAMERAS
from render_cinematic_6k_output import scale_camera
from temporal_texture_view_prior import angle_weights

ROOT=Path('/mnt/data/dec5_coherent_lip_texture')
BASE=Path('/mnt/data/dec5_cinematic_wide_spiral_6k_output_v1')
FRAME='001083'


def depth_samples(depth,uv):
    uv=uv.copy();near=np.rint(uv);uv=np.where(abs(uv-near)<=.001,near,uv)
    xy=np.floor(uv).astype(np.int64);fraction=uv-xy;h,w=depth.shape
    samples=[];weights=[]
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        samples.append(depth[np.clip(xy[:,1]+dy,0,h-1),np.clip(xy[:,0]+dx,0,w-1)])
        weights.append((fraction[:,0] if dx else 1-fraction[:,0])*(fraction[:,1] if dy else 1-fraction[:,1]))
    taps=np.array(samples);weight=np.array(weights)
    return uv,(taps*weight).sum(0),taps,weight


def main(mask_margin=0):
    import open3d as o3d
    import render_cinematic_6k_output as native
    from admit_mhr_local_patch_depth import Scene2
    from bake_joint_temporal_mesh import camera_depth
    ROOT.mkdir(exist_ok=False);folder=ROOT/'frames'/FRAME;folder.mkdir(parents=True)
    baseline=BASE/'frames'/FRAME;complete=read(baseline/'complete.json')
    assert complete['request_sha256']==sha(BASE/'request.json')
    for name,h in complete['hashes'].items():assert sha(baseline/name)==h
    parent=Path(read(BASE/'request.json')['parent']);q=read(parent/'request.json')
    from joint_temporal_texture import CALIBRATION
    assert sha(COLOR/'parameters.npz')==q['profiles_sha256']
    assert sha(COLOR/'exposure.json')==q['exposure_sha256']
    assert sha(CALIBRATION)==q['calibration_sha256']
    row=next(r for r in q['inventory'] if r['frame_id']==FRAME)
    rows,_,_=cameras(FRAME);indices=[next(i for i,r in enumerate(rows) if r['physical_camera']==name) for name in CAMERAS]
    assert sha(row['mesh'])==row['mesh_sha256']
    mesh=o3d.io.read_triangle_mesh(row['mesh']);v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles)
    tv=v[t];centroids=tv.mean(1);scene=Scene2(v,t)
    mq=read(MASKS/'request.json');mr=read(MASKS/'result.json');assert mr['request_sha256']==sha(MASKS/'request.json')
    votes=np.zeros(len(t),int);visibility=[];input_hashes={}
    maskspec=row['source_masks'];maskroot=Path(maskspec['root'])
    for name,key in [('masks.npz','masks_sha256'),('cameras.json','cameras_sha256'),('complete.json','complete_sha256')]:assert sha(maskroot/name)==maskspec[key]
    foreground=dict(zip(read(maskroot/'cameras.json'),np.load(maskroot/'masks.npz')['masks']))
    for ci in indices:
        camera=rows[ci];name=camera['physical_camera'];maskfile=MASKS/FRAME/(name+'_mask.png')
        receipt=next(r for r in mr['records'] if r['frame']==FRAME and r['camera']==name)
        assert sha(maskfile)==receipt['hashes']['_mask.png'];input_hashes[str(maskfile)]=sha(maskfile)
        mask=np.asarray(Image.open(maskfile))>0
        if mask_margin:
            from scipy.ndimage import binary_dilation
            yy,xx=np.mgrid[-mask_margin:mask_margin+1,-mask_margin:mask_margin+1]
            mask=binary_dilation(mask,structure=xx*xx+yy*yy<=mask_margin*mask_margin)
        uv,z=project(centroids,[camera]);uv=uv[0];z=z[0]
        xx=np.rint(uv[:,1]-mq['native_crop_portrait'][0]).astype(int)
        yy=np.rint(camera['w']-1-uv[:,0]-mq['native_crop_portrait'][1]).astype(int)
        inside=(z>0)&(xx>=0)&(xx<mask.shape[1])&(yy>=0)&(yy<mask.shape[0])
        semantic=np.zeros(len(t),bool);semantic[inside]=mask[yy[inside],xx[inside]]
        d,_,_=camera_depth(scene,camera);d=np.where(np.isfinite(d)&(foreground[name]>0),d,0)
        _,sampled,_,_=depth_samples(d,uv)
        visible=(z>0)&(sampled>0)&(abs(sampled-z)<.0015*z)
        visible&=(uv[:,0]>2)&(uv[:,0]<1917)&(uv[:,1]>2)&(uv[:,1]<1077)
        votes+=semantic&visible;visibility.append(visible)
    region=votes>=2;assert region.sum()>10
    visibility=np.array(visibility);coverage=visibility[:,region].mean(1)
    _,angles=angle_weights([rows[i] for i in indices],row['camera'],4.)
    eligible=coverage>=.99;assert eligible.any(),coverage
    chosen_local=int(np.argmin(np.where(eligible,angles,np.inf)));chosen=indices[chosen_local]
    target=read(baseline/'result.json')['camera'];surface_uv,_=project(v[np.unique(t[region])],[target]);surface_uv=surface_uv[0]
    px=surface_uv[:,1];py=target['w']-1-surface_uv[:,0]
    box=(max(0,int(np.floor(px.min()))-5),max(0,int(np.floor(py.min()))-5),
        min(target['h'],int(np.ceil(px.max()))+6),min(target['w'],int(np.ceil(py.max()))+6))
    x0,y0,x1,y1=box;crop_camera=deepcopy(target)
    crop_camera['cx']-=target['w']-y1;crop_camera['cy']-=x0;crop_camera.update(w=y1-y0,h=x1-x0)
    d,face,bary=camera_depth(scene,crop_camera);d=np.rot90(d);face=np.rot90(face);bary=np.rot90(bary)
    hit=np.isfinite(d);selected=np.zeros(hit.shape,bool);selected[hit]=region[face[hit]]
    pixel=np.flatnonzero(selected);f=face.ravel()[pixel];b=bary.reshape(-1,2)[pixel]
    weights=np.column_stack((1-b.sum(1),b));points=(tv[f]*weights[:,:,None]).sum(1)
    source=scale_camera(rows[chosen],5461,3072);uv,z=project(points,[source]);uv=uv[0];z=z[0]
    sd,_,_=camera_depth(scene,source)
    sx=5461/1920;sy=3072/1080;xx=np.minimum(((np.arange(5461)+.5)/sx).astype(int),1919);yy=np.minimum(((np.arange(3072)+.5)/sy).astype(int),1079)
    fg=foreground[source['physical_camera']][yy[:,None],xx[None]]>0
    sd=np.where(np.isfinite(sd)&fg,sd,0);uv,center,taps,tap_weights=depth_samples(sd,uv)
    valid=(z>0)&(center>0)&(abs(center-z)<.0015*z)
    valid&=(((taps>0)&(abs(taps-z)<.003*z))|(tap_weights==0)).all(0)
    hd=(uv+.5)/[sx,sy]-.5;valid&=(hd[:,0]>2)&(hd[:,0]<1917)&(hd[:,1]>2)&(hd[:,1]<1077)
    request=dict(frame=FRAME,script_sha256=sha(__file__),script_hashes={n:sha(Path(__file__).with_name(n)) for n in
        ['render_cinematic_6k_output.py','joint_temporal_texture.py','infer_train_lip_regions.py','bake_joint_temporal_mesh.py']},
        decoder_sha256=sha(Path(__file__).with_name('convert_dec5_5a3_pq16_to_exr.py')),
        baseline_complete_sha256=sha(baseline/'complete.json'),lip_request_sha256=sha(MASKS/'request.json'),
        input_hashes=input_hashes,mesh_sha256=sha(row['mesh']),region_rule='two of three train lip+visibility votes at centroid',
        candidate_cameras=CAMERAS,minimum_centroid_visibility=.99,coverage=coverage.tolist(),angles=angles.tolist(),
        mask_margin_source_pixels=mask_margin,profiles_sha256=q['profiles_sha256'],
        exposure_sha256=q['exposure_sha256'],calibration_sha256=q['calibration_sha256'],
        chosen_source=source['physical_camera'],source_rule='minimum calibration angle among eligible cameras',
        target_used_for_rendering_and_angular_preference_only=True,heldout_used=False,geometry_changed=False,
        rgb_averaging=False,outside_region_rgb_preserved=True,invalid_footprint_retains_baseline=True,
        native_worker_sha256=sha(native.__file__))
    # Worker identity is separately bound; this local controller is not its code.
    worker_request=dict(request,controller_sha256=request['script_sha256'],script_sha256=sha(native.__file__))
    atomic_json(ROOT/'request.json',worker_request)
    native.OUT=ROOT;native.REMOTE_ROOT='/fsx/tmp/lookcloser_coherent_lip_texture_v1'
    native.call(['ssh',native.REMOTE,'mkdir','-p',native.REMOTE_ROOT])
    native.call(['scp','-q',native.__file__,Path(__file__).with_name('convert_dec5_5a3_pq16_to_exr.py'),f'{native.REMOTE}:{native.REMOTE_ROOT}/'])
    samples=native.native_samples(FRAME,rows,{chosen:uv[valid]},folder)[chosen]
    gains=np.load(COLOR/'parameters.npz')['log_gain'];gain=np.exp(gains[chosen]-gains.mean(0));exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']
    before=np.asarray(Image.open(baseline/'frame.png').crop(box)).copy();after=before.copy()
    after.reshape(-1,3)[pixel[valid]]=np.rint(display(samples*gain,exposure)*255).clip(0,255).astype(np.uint8)
    changed=np.any(before!=after,2);allowed=np.zeros(selected.shape,bool);allowed.ravel()[pixel[valid]]=True
    assert not (changed&~allowed).any()
    Image.fromarray(after).save(folder/'lip_crop.png');panel=Image.new('RGB',(before.shape[1]*2,before.shape[0]+24));draw=ImageDraw.Draw(panel)
    for i,(title,im) in enumerate([('Frozen graph sources',before),(f'Coherent lip: {source["physical_camera"]}',after)]):
        panel.paste(Image.fromarray(im),(i*before.shape[1],24));draw.text((i*before.shape[1]+4,4),title,fill='white')
    panel.save(folder/'comparison.png')
    np.savez_compressed(folder/'evidence.npz',region_faces=np.flatnonzero(region),selected_pixels=pixel,valid=valid,source_uv=uv)
    atomic_json(folder/'result.json',dict(request_sha256=sha(ROOT/'request.json'),region_faces=int(region.sum()),
        crop_box=box,region_pixels=len(pixel),valid_pixels=int(valid.sum()),changed_rgb_pixels=int(changed.sum()),
        maximum_code_difference=int(abs(after.astype(int)-before.astype(int)).max()),production_accepted=False,
        visual_status='pending',hashes={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='result.json'}))
    print(source['physical_camera'],coverage,'pixels',len(pixel),'valid',int(valid.sum()),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT)
    parser.add_argument('--mask-margin',type=int,default=0)
    args=parser.parse_args()
    if not 0<=args.mask_margin<=8:parser.error('Diagnostic margin must be 0..8 source pixels')
    ROOT=args.output
    main(args.mask_margin)
