#!/usr/bin/env python3
"""Measure existing PatchMatch/shared meshes on the same RoMa hair witnesses."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from dec5_hair import cv_pose,project,unproject
from import_colmap_mvs_depth_dataset import read_colmap_dense_array
from run_dec5_hair_campaign import atomic_json,sha


def score(root,mvs,correspondences,receipt_path,meshes,output):
    import open3d as o3d
    if output.exists():raise FileExistsError(output)
    witnesses=np.load(correspondences);receipt=json.loads(receipt_path.read_text())
    payload=json.loads((root/'dataset/transforms.json').read_text());frames={f['physical_camera']:f for f in payload['frames']}
    scenes={};mesh_sha={}
    for p in meshes:
        mesh=o3d.io.read_triangle_mesh(str(p));scene=o3d.t.geometry.RaycastingScene(nthreads=2)
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));scenes[str(p)]=scene;mesh_sha[str(p)]=sha(p)
    pooled={};coverage={};rows=[]
    for uv_key,row in zip(['uv_a','uv_b','uv_c'],receipt['rows']):
        frame=frames[row['frame']['physical_camera']]
        np.testing.assert_allclose(frame['transform_matrix'],row['frame']['transform_matrix'],rtol=0,atol=1e-10)
        uv=witnesses[uv_key];_,reference=project(witnesses['points'],frame)
        array=read_colmap_dense_array(mvs/'dense/stereo/depth_maps'/(frame['file_path']+'.geometric.bin'))[...,0]
        values={}
        for label,mode in [('patchmatch_nearest',cv2.INTER_NEAREST),('patchmatch_bilinear',cv2.INTER_LINEAR)]:
            values[label]=cv2.remap(array,(uv[:,0]-.5).astype(np.float32)[:,None],(uv[:,1]-.5).astype(np.float32)[:,None],mode)[:,0]
        origin=cv_pose(frame)[:3,3];directions=unproject(uv,np.ones(len(uv)),frame)-origin
        rays=np.concatenate([np.broadcast_to(origin,directions.shape),directions],-1).astype(np.float32)
        for label,scene in scenes.items():values[label]=scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        metrics={}
        for label,value in values.items():
            valid=np.isfinite(value)&(value>0);relative=np.where(valid,np.abs(value-reference)/reference,1.)
            pooled.setdefault(label,[]).append(relative);coverage.setdefault(label,[]).append(valid)
            metrics[label]=dict(median_absolute_relative=float(np.median(relative)),p90_absolute_relative=float(np.quantile(relative,.9)),valid_fraction=float(valid.mean()))
        rows.append(dict(camera=frame['physical_camera'],witnesses=len(uv),metrics=metrics))
    aggregate={k:dict(median_absolute_relative=float(np.median(np.concatenate(v))),p90_absolute_relative=float(np.quantile(np.concatenate(v),.9)),valid_fraction=float(np.concatenate(coverage[k]).mean())) for k,v in pooled.items()}
    atomic_json(output,dict(aggregate=aggregate,records=rows,proxy='same 133 independent-detector triangulated train-RGB points; not independent 3D GT',
        depth_convention='camera-z in calibration units',invalid_depth_relative_penalty=1.,
        meshes_sha256=mesh_sha,witness_sha256=sha(correspondences),matching_receipt_sha256=sha(receipt_path),
        uses_validation_rgb=False,uses_test_rgb=False,implementation_sha256=sha(__file__)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','mvs','correspondences','matching-receipt','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mesh',type=Path,action='append',default=[])
    a=p.parse_args();score(a.root,a.mvs,a.correspondences,a.matching_receipt,a.mesh,a.output)


if __name__=='__main__':main()
