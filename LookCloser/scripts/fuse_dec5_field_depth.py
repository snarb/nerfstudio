#!/usr/bin/env python3
"""Build one shared 3D receiver surface from frozen train-only LookCloser depth.

This uses no PatchMatch depth or Gaussian geometry. TSDF excludes transparent
matte boundaries; the renderer must retain an explicit boundary/miss fallback.
"""
from pathlib import Path
import argparse,json
import numpy as np
from dec5_hair import cv_pose,integer_center_pinhole,VALIDATION,TEST
from run_dec5_hair_campaign import atomic_json,sha


def fuse(data,depths,training,checkpoint,targets,output,voxel,depth_key):
    import open3d as o3d
    if voxel<=0:raise ValueError('Positive voxel size required')
    request=json.loads((training/'request.json').read_text());source=json.loads((depths/'complete.json').read_text())
    mattes=json.loads((targets/'complete.json').read_text());dataset_sha=sha(data/'transforms.json')
    if (source['dataset_sha256']!=dataset_sha or request['data_sha256']!=dataset_sha or mattes['dataset_sha256']!=dataset_sha
            or source['checkpoint_sha256']!=sha(checkpoint) or not source.get('frozen_state_unchanged')
            or not source.get('source_visibility_only') or source.get('uses_validation_rgb') is not False
            or source.get('uses_test_rgb') is not False or mattes.get('uses_validation_rgb') is not False
            or mattes.get('uses_test_rgb') is not False):raise ValueError('Shared surface requires one frozen train-only calibrated field and train mattes')
    payload=json.loads((data/'transforms.json').read_text())
    frames=[f for f in payload['frames'] if f['file_path'] in payload['train_filenames']]
    names={f['physical_camera'] for f in frames};by_depth={r['camera']:r for r in source['records']};by_matte={r['camera']:r for r in mattes['records']}
    if len(frames)!=59 or names!=set(by_depth) or names!=set(by_matte) or names.intersection(VALIDATION+TEST):raise ValueError('Train59 field/matte inventory mismatch')
    output.mkdir(parents=True,exist_ok=False)
    volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel,sdf_trunc=8*voxel,color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    rows=[]
    for i,frame in enumerate(frames):
        name=frame['physical_camera'];dp=depths/(name+'.npz');mp=targets/(name+'.npz')
        if sha(dp)!=by_depth[name]['sha256'] or sha(mp)!=by_matte[name]['sha256']:raise ValueError('Changed field depth or foreground target')
        if any(abs(float(frame.get(k,0)))>1e-12 for k in ['k1','k2','k3','k4','p1','p2']):raise ValueError('Undistorted pinhole arrays are required')
        field=np.load(dp);matte=np.load(mp);depth=field[depth_key].astype(np.float32)
        h,w=frame['h'],frame['w']
        if depth.shape!=(h,w):raise ValueError('Field depth/camera raster mismatch')
        x0,y0,x1,y1=matte['portrait_crop'].astype(int);upright=np.zeros((w,h),np.float32)
        upright[y0:y1,x0:x1]=matte['alpha'].astype(np.float32)/65535
        alpha=np.ascontiguousarray(np.rot90(upright,-1))
        valid=(alpha>=.5)&field['roi']&(field['opacity']>=.5)&np.isfinite(depth)&(depth>0)
        depth=np.where(valid,depth,0).astype(np.float32)
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.zeros((h,w,3),np.uint8)),o3d.geometry.Image(depth),depth_scale=1.,depth_trunc=100.,convert_rgb_to_intensity=False)
        intrinsic=o3d.camera.PinholeCameraIntrinsic(w,h,*integer_center_pinhole(frame))
        volume.integrate(rgbd,intrinsic,np.linalg.inv(cv_pose(frame)))
        rows.append(dict(camera=name,depth_sha256=sha(dp),matte_sha256=sha(mp),valid_pixels=int(valid.sum())))
        atomic_json(output/'progress.json',dict(status='integrating',completed=i+1,total=59))
    mesh=volume.extract_triangle_mesh()
    if not len(mesh.triangles):raise RuntimeError('Field TSDF produced an empty surface')
    o3d.io.write_triangle_mesh(str(output/'mesh.ply'),mesh)
    atomic_json(output/'complete.json',dict(records=rows,triangles=len(mesh.triangles),vertices=len(mesh.vertices),
        voxel_length=voxel,sdf_trunc=8*voxel,depth_key=depth_key,source='LookCloser native ray depths only',
        depth_convention='OpenCV camera-z',units='original calibration units',
        pixel_centers='continuous centers x+.5,y+.5; Open3D integer-center cx-.5,cy-.5',
        input_mask='train person matte alpha>=.5, native opacity>=.5, known head ROI',
        transparent_boundary_policy='excluded from mesh; explicit receiver fallback required',component_pruning=False,
        checkpoint_sha256=sha(checkpoint),dataset_sha256=dataset_sha,source_depth_receipt_sha256=sha(depths/'complete.json'),
        mesh_sha256=sha(output/'mesh.ply'),implementation_sha256=sha(__file__),uses_validation_rgb=False,uses_test_rgb=False))
    atomic_json(output/'progress.json',dict(status='complete',completed=59,total=59,triangles=len(mesh.triangles)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','depths','training','checkpoint','targets','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--voxel-length',type=float,default=.001);p.add_argument('--depth-key',choices=['expected','median'],default='median')
    a=p.parse_args();fuse(a.data,a.depths,a.training,a.checkpoint,a.targets,a.output,a.voxel_length,a.depth_key)


if __name__=='__main__':main()
