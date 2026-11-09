#!/usr/bin/env python3
"""Calibrated, train-only learned-stereo screen against existing hair witnesses.

Uses official FoundationStereo weights without fitting. Exact stereo/Q roundtrip
precedes inference; no relative-depth scale alignment or validation/test RGB.
"""
from pathlib import Path
import argparse,json,sys,time
import cv2,numpy as np,torch
from dec5_hair import cv_pose,project,resize_pinhole
from import_colmap_mvs_depth_dataset import load_binary_pinhole_calibration,read_colmap_dense_array
from run_dec5_hair_campaign import atomic_json,sha


def camera_matrix(f,rotate):
    if rotate:return np.array([[f['fl_y'],0,f['cy']-.5],[0,f['fl_x'],f['w']-f['cx']-.5],[0,0,1.]])
    return np.array([[f['fl_x'],0,f['cx']-.5],[0,f['fl_y'],f['cy']-.5],[0,0,1.]])


def prepare_pair(left,right,images,masks,points):
    for rotate in [True,False]:
        A=np.array([[0,1,0],[-1,0,0],[0,0,1.]]) if rotate else np.eye(3)
        for swap in [False,True]:
            fl,fr=(right,left) if swap else (left,right)
            cl,cr=cv_pose(fl),cv_pose(fr);Rl=cl[:3,:3]@A.T;Rr=cr[:3,:3]@A.T
            relative=Rr.T@Rl;translation=Rr.T@(cl[:3,3]-cr[:3,3])
            size=(fl['h'],fl['w']) if rotate else (fl['w'],fl['h'])
            Kl,Kr=camera_matrix(fl,rotate),camera_matrix(fr,rotate)
            r1,r2,p1,p2,Q,_,_=cv2.stereoRectify(Kl,np.zeros(5),Kr,np.zeros(5),size,relative,translation,flags=cv2.CALIB_ZERO_DISPARITY,alpha=1)
            tx=float((r2@translation)[0])
            if abs(p2[1,3])>1e-8 or tx>=0:continue
            # Convergent, long-focal cameras have >1000px absolute disparity.
            # Center their train-only hair silhouettes in a calibrated virtual
            # stereo pair, keeping a 128px nominal disparity. Different cx is
            # retained explicitly in Q; no depth labels set this image shift.
            centers=[]
            for frame,K,rectification in [(fl,Kl,r1),(fr,Kr,r2)]:
                mask=masks[frame['physical_camera']]
                if rotate:mask=np.rot90(mask)
                yy,xx=np.nonzero(mask>.5)
                if not len(xx):raise ValueError('Missing train hair silhouette')
                ray=rectification@np.linalg.solve(K,np.array([xx.mean(),yy.mean(),1.]))
                centers.append(ray[:2]/ray[2])
            focal=float(np.mean([Kl[0,0],Kl[1,1],Kr[0,0],Kr[1,1]]))
            cx1=size[0]/2+64-focal*centers[0][0];cx2=size[0]/2-64-focal*centers[1][0]
            cy=size[1]/2-focal*(centers[0][1]+centers[1][1])/2
            p1=np.array([[focal,0,cx1,0],[0,focal,cy,0],[0,0,1,0.]])
            p2=np.array([[focal,0,cx2,focal*tx],[0,focal,cy,0],[0,0,1,0.]])
            Q=np.array([[1,0,0,-cx1],[0,1,0,-cy],[0,0,0,focal],[0,0,-1/tx,(cx1-cx2)/tx]])
            maps=[cv2.initUndistortRectifyMap(K,np.zeros(5),r,p,size,cv2.CV_32FC1) for K,r,p in [(Kl,r1,p1),(Kr,r2,p2)]]
            rect=[];rect_masks=[]
            for frame,m in zip([fl,fr],maps):
                image=images[frame['physical_camera']];mask=masks[frame['physical_camera']]
                if rotate:image=np.rot90(image);mask=np.rot90(mask)
                rect.append(cv2.remap(image,*m,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT))
                rect_masks.append(cv2.remap(mask,*m,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT))
            yy,xx=np.nonzero((rect_masks[0]>.5)|(rect_masks[1]>.5))
            if not len(xx):raise ValueError('Rectification lost hair ROI')
            box=[max(int(xx.min())-32,0),max(int(yy.min())-32,0),min(int(xx.max())+33,size[0]),min(int(yy.max())+33,size[1])]
            x0,y0,x1,y1=box
            if (x1-x0)*(y1-y0)>900000:raise ValueError('Head union exceeds conservative inference area cap')
            xr=(points-cl[:3,3])@Rl@r1.T
            uv=xr[:,:2]/xr[:,2,None];uv=uv*p1[[0,1],[0,1]]+p1[:2,2]
            disparity=-p2[0,3]/xr[:,2]+p1[0,2]-p2[0,2]
            xright=(points-cr[:3,3])@Rr@r2.T
            vertical=focal*xright[:,1]/xright[:,2]+cy
            if np.max(np.abs(vertical-uv[:,1]))>1e-7:raise ValueError('Epipolar vertical alignment failed')
            h=np.c_[uv,disparity,np.ones(len(uv))]@Q.T;recovered=h[:,:3]/h[:,3,None]
            world=recovered@r1@Rl.T+cl[:3,3]
            error=float(np.linalg.norm(world-points,axis=-1).max())
            if error>1e-7:raise ValueError('Stereo/Q/world roundtrip failed')
            return dict(left=fl,right=fr,left_rect=rect[0][y0:y1,x0:x1],right_rect=rect[1][y0:y1,x0:x1],
                box=box,Q=Q,R1=r1,R_left=Rl,center=cl[:3,3],uv=uv,reference_disparity=disparity,
                rotate_ccw=rotate,swapped=swap,roundtrip_error=error,principal_point_delta=float(cx1-cx2))
    raise ValueError('No positive horizontal-disparity rectification')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','external-root','weights','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--input-scale',type=float,default=1.)
    p.add_argument('--gpu-cap-gib',type=float,default=14)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if not .25<=a.input_scale<=1:raise ValueError('Input scale must be .25..1')
    torch.set_num_threads(2)
    if a.gpu_cap_gib<=0:raise ValueError('Positive inference memory cap required')
    free,total=torch.cuda.mem_get_info()
    if free<(a.gpu_cap_gib+16)*(1<<30):raise ValueError('Inference cap plus 16 GiB service reserve is unavailable')
    torch.cuda.set_per_process_memory_fraction(a.gpu_cap_gib*(1<<30)/total)
    torch.manual_seed(42);np.random.seed(42);torch.set_grad_enabled(False)
    data=a.root/'dataset';payload=json.loads((data/'transforms.json').read_text());frames={f['physical_camera']:f for f in payload['frames'] if f['file_path'] in payload['train_filenames']}
    names=['E004_C005_1210YM','F004_C005_121059','G004_C005_121037'];mvs=a.root/'pm1920_native_v2/dense';cal=load_binary_pinhole_calibration(mvs/'sparse')
    images={};masks={};by_name={}
    for name in names:
        f=frames[name];image=cv2.imread(str(mvs/'images'/f['file_path']))
        if image is None:raise ValueError('Missing undistorted train image')
        images[name]=image[...,::-1].copy();f=resize_pinhole(dict(f,**cal[f['file_path']]),image.shape[1],image.shape[0]);by_name[name]=f
        mask=np.load(a.root/'semantics'/(name+'.npz'))['probability'][...,0].astype(np.float32)/65535
        masks[name]=cv2.resize(mask,(f['w'],f['h']),interpolation=cv2.INTER_LINEAR)
    points=np.load(a.root/'roma_triplet/triangulated.npz')['points']
    prepared=[]
    for i,j in [(0,1),(1,2),(0,2)]:prepared.append(prepare_pair(by_name[names[i]],by_name[names[j]],images,masks,points))
    atomic_json(a.output/'preflight.json',dict(pairs=[dict(left=x['left']['physical_camera'],right=x['right']['physical_camera'],box=x['box'],rotate_ccw=x['rotate_ccw'],swapped=x['swapped'],roundtrip_error=x['roundtrip_error'],principal_point_delta=x['principal_point_delta'],reference_disparity_quantiles=np.quantile(x['reference_disparity'],[0,.5,1]).tolist()) for x in prepared],rectification_center='train-only hair silhouette centroids; nominal disparity 128; cx difference retained in Q',uses_validation_rgb=False,uses_test_rgb=False))
    sys.path.insert(0,str(a.external_root))
    import timm
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder
    cfg=OmegaConf.load(a.weights.parent/'cfg.yaml');cfg.vit_size=cfg.get('vit_size','vitl')
    # Strict checkpoint loading covers every parameter. Avoid downloading a
    # redundant ImageNet initialization that would immediately be overwritten.
    create=timm.create_model
    def no_initial_download(*args,**kwargs):kwargs['pretrained']=False;return create(*args,**kwargs)
    timm.create_model=no_initial_download
    try:model=FoundationStereo(cfg)
    finally:timm.create_model=create
    checkpoint=torch.load(a.weights,map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
    model.cuda().eval();rows=[]
    for index,pair in enumerate(prepared):
        if pair['reference_disparity'].max()*a.input_scale>=cfg.max_disp*.95:
            rows.append(dict(pair=index,status='reference exceeds pretrained disparity range'));continue
        orig_h,orig_w=pair['left_rect'].shape[:2]
        resized=[cv2.resize(pair[k],None,fx=a.input_scale,fy=a.input_scale,interpolation=cv2.INTER_AREA) for k in ['left_rect','right_rect']]
        left,right=[torch.as_tensor(np.ascontiguousarray(image),device='cuda').permute(2,0,1)[None].float() for image in resized]
        h,w=left.shape[-2:];pad=InputPadder(left.shape,divis_by=32);l,r=pad.pad(left,right)
        start=time.time();torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type='cuda',dtype=torch.float16):disparity=model(l,r,iters=32,test_mode=True)
        disparity=pad.unpad(disparity.float()).reshape(h,w).cpu().numpy()
        disparity=cv2.resize(disparity,(orig_w,orig_h),interpolation=cv2.INTER_LINEAR)*(orig_w/w)
        x0,y0,x1,y1=pair['box'];uv=pair['uv'];sample=cv2.remap(disparity,(uv[:,0]-x0).astype(np.float32)[:,None],(uv[:,1]-y0).astype(np.float32)[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[:,0]
        homogeneous=np.c_[uv,sample,np.ones(len(uv))]@pair['Q'].T
        xyz=homogeneous[:,:3]/np.maximum(homogeneous[:,3,None],1e-12)
        world=xyz@pair['R1']@pair['R_left'].T+pair['center'];_,depth=project(world,pair['left']);_,reference=project(points,pair['left'])
        valid=(sample>0)&np.isfinite(depth)&(depth>0)&(uv[:,0]>=x0)&(uv[:,0]<x1-1)&(uv[:,1]>=y0)&(uv[:,1]<y1-1)
        relative=np.where(valid,np.abs(depth-reference)/reference,1.)
        f=pair['left'];raw=read_colmap_dense_array(mvs/'stereo/depth_maps'/(f['file_path']+'.geometric.bin'))[...,0]
        old_uv,_=project(points,f);raw_values=cv2.remap(raw,(old_uv[:,0]-.5).astype(np.float32)[:,None],(old_uv[:,1]-.5).astype(np.float32)[:,None],cv2.INTER_LINEAR)[:,0]
        raw_error=np.where(raw_values>0,np.abs(raw_values-reference)/reference,1.)
        np.savez_compressed(a.output/('pair_%d.npz'%index),disparity=disparity,witness_prediction=depth,witness_reference=reference,valid=valid,Q=pair['Q'],crop=pair['box'])
        vis=cv2.applyColorMap(np.rint(np.clip(disparity/cfg.max_disp,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
        cv2.imwrite(str(a.output/('pair_%d.jpg'%index)),np.concatenate((pair['left_rect'][...,::-1],pair['right_rect'][...,::-1],vis),1))
        rows.append(dict(pair=index,left=f['physical_camera'],right=pair['right']['physical_camera'],valid_fraction=float(valid.mean()),median_relative=float(np.median(relative)),p90_relative=float(np.quantile(relative,.9)),
            patchmatch_median_relative=float(np.median(raw_error)),patchmatch_p90_relative=float(np.quantile(raw_error,.9)),seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            actual_input_shape=[h,w],effective_xy_scale=[w/orig_w,h/orig_h],upsampled_disparity=a.input_scale<1))
        atomic_json(a.output/'progress.json',dict(completed=index+1,total=3,rows=rows))
    atomic_json(a.output/'metrics.json',dict(rows=rows,checkpoint_sha256=sha(a.weights),checkpoint_source='official NVlabs FoundationStereo linked public weights, ViT-large 23-51-11',
        official_repository='https://github.com/NVlabs/FoundationStereo',license='NVIDIA non-commercial research license; separate experiment only',
        strict_all_parameters_loaded=True,redundant_timm_pretrained_initialization_download_disabled=True,uses_validation_rgb=False,uses_test_rgb=False,
        input_scale=a.input_scale,
        gpu_cap_gib=a.gpu_cap_gib,
        depth_units='original calibration, no relative-scale fitting',proxy='133 reconstructed train-image hair witnesses, not independent 3D GT',implementation_sha256=sha(__file__)))

if __name__=='__main__':main()
