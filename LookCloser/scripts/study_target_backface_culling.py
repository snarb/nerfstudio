"""Opt-in target-only backface culling; train visibility and mesh stay unchanged.

The triangle subset is used solely for target ray intersection. Original face
IDs and barycentrics are restored before texture lookup, so face labels retain
their identity. No inferred RGB is generated for hidden back surfaces.
"""
import argparse
from copy import deepcopy
import multiprocessing
from pathlib import Path
import numpy as np
import open3d as o3d
from study_multiview_face_prior import read,save,sha
from study_train_gap_subfaces import ROOT as PARENT,FRAME
from review_measured_free_surface import VIEWS

ROOT=Path('/mnt/data/dec5_target_backface_culling')


def front_faces(vertices,triangles,center):
    v,t,c=np.asarray(vertices),np.asarray(triangles),np.asarray(center)
    tv=v[t];normal=np.cross(tv[:,1]-tv[:,0],tv[:,2]-tv[:,0])
    return np.einsum('ij,ij->i',normal,c-tv[:,0])>0


def remap_ids(ids,retained):
    ids=np.asarray(ids);out=np.full(ids.shape,np.iinfo(np.uint32).max,np.uint32)
    valid=ids!=np.iinfo(np.uint32).max
    out[valid]=np.asarray(retained,np.uint32)[ids[valid]]
    return out


def run(view):
    from diffusion_mesh_repair import scene_for
    from run_view_consistent_dynamic_video import install
    import render_smooth_temporal_mesh_video as renderer
    old=PARENT/'carved'/FRAME/'rgb'/view;parent=renderer.verify_request(old)
    row=parent['inventory'][0];camera=row['camera']
    mesh=o3d.io.read_triangle_mesh(row['mesh']);assert sha(row['mesh'])==row['mesh_sha256']
    v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles,np.uint32)
    retain=np.flatnonzero(front_faces(v,t,np.array(camera['transform_matrix'])[:3,3]))
    target_scene=scene_for(v,t[retain]);original_depth=renderer.camera_depth
    calls=[]
    def culled(scene,query):
        if query['physical_camera']!=camera['physical_camera']:return original_depth(scene,query)
        assert query==camera
        d,ids,b=original_depth(target_scene,query);ids=remap_ids(ids,retain)
        calls.append(dict(retained_faces=len(retain),total_faces=len(t),hits=int(np.isfinite(d).sum())))
        return d,ids,b
    # Install before the source-mask wrapper captures camera_depth. Its source
    # camera calls use the original scene; only this distinct target ID changes.
    renderer.camera_depth=culled
    implementation=install();assert implementation==parent['source_quality_implementation_sha256']
    renderer.torch.set_num_threads(2)
    request=deepcopy(parent);request.update(target_backface_culling=True,source_backface_culling=False,
        mesh_changed=False,baseline_request_sha256=sha(old/'request.json'),
        target_culling_script_sha256=sha(__file__),production_changed=False,
        original_face_ids_preserved=True,source_visibility_and_graph_unchanged=True)
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    out=ROOT/FRAME/view;out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir()
    save(out/'request.json',request)
    renderer.render(out,[FRAME]);assert len(calls)==1
    np.save(out/'target_retained_faces.npy',retain)
    save(out/'culling_audit.json',dict(calls=calls,request_sha256=sha(out/'request.json'),
        frame_complete_sha256=sha(out/'frames'/FRAME/'complete.json'),
        retained_faces_sha256=sha(out/'target_retained_faces.npy'),source_visibility_changed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--view',choices=VIEWS)
    a=p.parse_args()
    if a.view:run(a.view)
    else:
        with multiprocessing.get_context('spawn').Pool(3,maxtasksperchild=1) as pool:pool.map(run,VIEWS)
