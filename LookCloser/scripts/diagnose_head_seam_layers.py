"""Separate RGB-source seams, missing geometry, and earlier head-completion faces."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from review_jaw_repair_transfer import verified_image,panel

ROOT=Path('/mnt/data/dec5_head_seam_layers')
BASE=Path('/mnt/data/dec5_central_train_pose_transfer')
CASES=[('001193','native_G004_C005_121037'),('001123','native_K004_C005_1210BC'),('001193','moving')]


def run():
    ROOT.mkdir(exist_ok=False);records=[]
    for frame,view in CASES:
        location=BASE/frame/view;image,result=verified_image(location,frame)
        request=read(location/'request.json');entry=request['inventory'][0]
        meshpath=Path(entry['mesh']);assert sha(meshpath)==entry['mesh_sha256']
        receipt=read(meshpath.parent/'complete.json');oldpath=receipt['request']['source_mesh']
        assert sha(oldpath)==receipt['request']['source_sha256']
        m=o3d.io.read_triangle_mesh(str(meshpath));old=o3d.io.read_triangle_mesh(oldpath)
        v=np.asarray(m.vertices,np.float32);t=np.asarray(m.triangles,np.uint32)
        ov=np.asarray(old.vertices,np.float32);ot=np.asarray(old.triangles,np.uint32)
        np.testing.assert_array_equal(v[t[:len(ot)]],ov[ot])
        d,ids,b=camera_depth(scene_for(v,t),entry['camera']);hit=np.isfinite(d)
        saved=np.load(location/'frames'/frame/'target_depth.npz')['depth']
        np.testing.assert_allclose(np.where(hit,d,0),saved,rtol=1e-6,atol=1e-8)
        source=np.array(Image.open(location/'frames'/frame/'source_ids.png'))
        normal=np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]])
        normal/=np.linalg.norm(normal,axis=1)[:,None].clip(1e-12)
        center=np.array(entry['camera']['transform_matrix'])[:3,3]
        facing=np.zeros(d.shape,np.float32)
        point=(v[t[ids[hit]]]*np.column_stack((1-b[hit].sum(1),b[hit]))[:,:,None]).sum(1)
        direction=center-point;direction/=np.linalg.norm(direction,axis=1)[:,None]
        facing[hit]=np.abs((normal[ids[hit]]*direction).sum(1))*.7+.3
        clay=np.repeat(np.rint(facing[...,None]*255).astype(np.uint8),3,2)
        added=hit&(ids>=len(ot));added_map=np.zeros((*d.shape,3),np.uint8);added_map[hit]=[120,120,120];added_map[added]=[0,200,255]
        boundary=np.zeros(d.shape,bool)
        boundary[:,1:]|=(source[:,1:]!=source[:,:-1])&(source[:,1:]!=255)&(source[:,:-1]!=255)
        boundary[1:]|=(source[1:]!=source[:-1])&(source[1:]!=255)&(source[:-1]!=255)
        overlay=np.array(Image.open(location/'frames'/frame/'prediction_native.png'));overlay[boundary]=[255,0,0]
        rng=np.random.default_rng(17);palette=rng.integers(40,240,(256,3),dtype=np.uint8);palette[255]=0
        support=np.zeros((*d.shape,3),np.uint8);support[hit]=[200,200,200];support[hit&(source==255)]=[255,0,255]
        out=ROOT/(frame+'_'+view);out.mkdir()
        maps={'rgb':image,'clay':np.rot90(clay),'source_boundary':np.rot90(overlay),
              'source_id':np.rot90(palette[source]),'added_faces':np.rot90(added_map),'geometry_support':np.rot90(support)}
        paths=[]
        for name,array in maps.items():
            p=out/(name+'.png');panel(p,[array],[name],(160,450,1000,1250));paths.append(p)
        np.savez_compressed(out/'layers.npz',depth=d,face_ids=ids,source_ids=source,added=added,source_boundary=boundary)
        records.append(dict(frame=frame,view=view,mesh_sha256=sha(meshpath),old_mesh_sha256=sha(oldpath),
            original_triangle_prefix_verified=True,depth_replay_verified=True,source_request_sha256=sha(location/'request.json'),
            old_triangles=len(ot),added_triangles=len(t)-len(ot),
            files={p.name:sha(p) for p in [*paths,out/'layers.npz']},visual_status='pending'))
    atomic_json(ROOT/'result.json',dict(records=records,script_sha256=sha(__file__),
        no_image_quality_metrics=True,diagnostic_only=True,geometry_changed=False,video_changed=False))
    print('Prepared three geometry/source-layer cases',flush=True)


if __name__=='__main__':run()
