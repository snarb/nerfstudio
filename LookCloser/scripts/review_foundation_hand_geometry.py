"""Metric-scale learned stereo surfaces and cross-pair checks; no mesh rollout."""
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.ndimage import map_coordinates
from joint_temporal_texture import read,sha,atomic_json,cameras
from calibrated_stereo_rectification import disparity_to_world
from review_hand_silhouette_volume import shaded
from review_jaw_repair_transfer import panel

ROOT=Path('/mnt/data/dec5_foundation_hand_geometry')
SOURCES=[Path('/mnt/data/dec5_foundation_hand_stereo/001037'),Path('/mnt/data/dec5_foundation_wrist_stereo/001037')]
MOVIE=Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150')


def grid_triangles(points,valid,maximum_edge=.002):
    ids=np.full(valid.shape,-1,np.int32);ids[valid]=np.arange(valid.sum())
    a,b,c,d=ids[:-1,:-1],ids[:-1,1:],ids[1:,:-1],ids[1:,1:]
    triangles=[]
    for x,y,z in [(a,c,b),(b,c,d)]:
        keep=(x>=0)&(y>=0)&(z>=0);triangles.append(np.column_stack([x[keep],y[keep],z[keep]]))
    triangles=np.concatenate(triangles);v=points[valid]
    edges=np.linalg.norm(v[triangles]-v[triangles[:,[1,2,0]]],axis=2)
    return v,triangles[(edges<=maximum_edge).all(1)]


def run():
    ROOT.mkdir(exist_ok=False);records=[];maps=[];dependencies={}
    for source in SOURCES:
        request=read(source/'request.json');done=read(source/'inference/complete.json')
        assert done['request_sha256']==sha(source/'inference/request.json')
        assert read(source/'inference/request.json')['staged_request_sha256']==sha(source/'request.json')
        dependencies[str(source/'request.json')]=sha(source/'request.json')
        dependencies[str(source/'inference/complete.json')]=sha(source/'inference/complete.json')
        for r in request['pairs']:
            folder=Path(r['directory']);name=folder.name;dest=ROOT/name;dest.mkdir()
            cal=np.load(folder/'calibration.npz');prediction=source/'inference'/name/'prediction.npz'
            assert sha(folder/'calibration.npz')==r['hashes']['calibration.npz']
            result=read(prediction.parent/'complete.json');assert sha(prediction)==result['prediction_sha256']
            data=np.load(prediction);dl=data['left_disparity'];y,x=np.indices(dl.shape)
            xyz=disparity_to_world(x,y,dl,cal['cropped_intrinsic'],cal['rectified_extrinsic'],float(cal['baseline']),float(cal['disparity_offset']))
            valid=data['valid_source_domain']&cal['left_mask'].astype(bool)
            right_skin=map_coordinates(cal['right_mask'].astype(float),[y,x-dl],order=1,mode='constant',cval=0)>.999
            for variant,keep in [('raw',valid&right_skin),('lr2',valid&right_skin&data['consistent'])]:
                v,t=grid_triangles(xyz,keep);mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t))
                mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals();path=dest/(variant+'.ply')
                o3d.io.write_triangle_mesh(str(path),mesh)
                records.append(dict(pair=name,variant=variant,vertices=len(mesh.vertices),triangles=len(mesh.triangles),mesh_sha256=sha(path)))
            mask=valid&right_skin&data['consistent']
            maps.append(dict(name=name,xyz=xyz,valid=mask,depth=data['rectified_depth'],K=cal['cropped_intrinsic'],E=cal['rectified_extrinsic']))
            dependencies[str(prediction)]=sha(prediction);dependencies[str(folder/'calibration.npz')]=sha(folder/'calibration.npz')
            print('meshed',name,records[-2:],flush=True)
    # Compare distinct stereo-pair predictions in a common metric frame.
    comparisons=[]
    for a in maps:
        points=a['xyz'][a['valid']][::4]
        for b in maps:
            if a is b:continue
            p=points@b['E'][:3,:3].T+b['E'][:3,3];z=p[:,2];uv=(p@b['K'].T);uv=uv[:,:2]/uv[:,2:]
            known=(z>0)&(map_coordinates(b['valid'].astype(float),uv.T[::-1],order=1,mode='constant',cval=0)>.999)
            depth=map_coordinates(b['depth'],uv.T[::-1],order=1,mode='constant',cval=0)
            delta=np.abs(depth[known]-z[known])
            comparisons.append(dict(source=a['name'],other=b['name'],query_points=len(points),overlap=int(known.sum()),
                absolute_depth_difference_median=float(np.median(delta)) if len(delta) else None,
                absolute_depth_difference_p90=float(np.percentile(delta,90)) if len(delta) else None,
                within_001_fraction=float((delta<=.001).mean()) if len(delta) else None,
                within_002_fraction=float((delta<=.002).mean()) if len(delta) else None))
    np.savez_compressed(ROOT/'point_fields.npz',**{m['name']+'_'+k:m[k] for m in maps for k in ['xyz','valid','depth','K','E']})
    atomic_json(ROOT/'result.json',dict(meshes=records,cross_pair=comparisons,dependencies=dependencies,
        script_sha256=sha(__file__),maximum_grid_edge=.002,source_rgb_for_depth=True,heldout_used=False,
        cross_pair_agreement_not_independent_ground_truth=True,production_updated=False,visual_status='pending'))
    views0,_,_=cameras('001037');views={r['physical_camera']:r for r in views0 if r['physical_camera'] in ['H004_A005_1210M6','E004_C005_1210YM']}
    entry=next(r for r in read(MOVIE/'request.json')['inventory'] if r['frame_id']=='001037');views['moving']=entry['camera']
    meshes={'production':o3d.io.read_triangle_mesh(entry['mesh'])}
    meshes.update({r['pair']:o3d.io.read_triangle_mesh(str(ROOT/r['pair']/'lr2.ply')) for r in records if r['variant']=='lr2'})
    panels=[]
    for name,row in views.items():
        images=[];labels=[]
        for n,m in meshes.items():im,_=shaded(m,row);images.append(im);labels.append(n)
        box=(0,1400,500,1920) if name!='moving' else (100,1360,720,1920)
        path=ROOT/'review'/(name+'.png');panel(path,images,labels,box);panels.append(dict(path=str(path),sha256=sha(path)))
    atomic_json(ROOT/'review/result.json',dict(panels=panels,visual_status='pending',geometry_shading_not_rgb_prediction=True))
    print('cross pair',comparisons,flush=True)


if __name__=='__main__':run()
