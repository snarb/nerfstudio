"""Persist normal/graph evidence; does not edit the mesh or render inputs."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from study_multiview_face_prior import read,save,sha
from joint_temporal_texture import cameras
from temporal_texture_view_prior import angle_weights
from study_temporal_source_retention import BASE


def main():
    root=Path('/mnt/data/dec5_face_angular_visibility');frame='001123';q=read(BASE/'request.json')
    record=next(r for r in q['inventory'] if r['frame_id']==frame);assert sha(record['mesh'])==record['mesh_sha256']
    r=read(BASE/'frames'/frame/'result.json');path=Path('/mnt/data/dec5_nose_source_visibility_001123/evidence.npz')
    e=np.load(path);m=o3d.io.read_triangle_mesh(record['mesh']);m.compute_triangle_normals();m.compute_vertex_normals()
    v=np.asarray(m.vertices);t=np.asarray(m.triangles);f=e['face_ids'];pts=e['points'];rows,_,_=cameras(frame)
    centers=np.array([x['transform_matrix'] for x in rows])[:,:3,3];dire=centers[:,None]-pts;dist=np.linalg.norm(dire,axis=2);dire/=dist[...,None]
    angle,_=angle_weights(rows,r['camera'],q['recipe']['target_angle_sigma_degrees'])
    normal=np.asarray(m.triangle_normals)[f];vertex=np.asarray(m.vertex_normals)[t[f]].mean(1);vertex/=np.linalg.norm(vertex,axis=1)[:,None]
    old=e['source_ids'];j=np.arange(len(old));values={}
    for label,n in [('face',normal),('mean_vertex',vertex)]:
        incidence=abs((dire*n).sum(-1));weight=incidence**2/dist**2*angle[:,None]
        values[label]=dict(h_incidence_quantiles=np.quantile(incidence[33],[0,.25,.5,.75,1]).tolist(),
            i_incidence_quantiles=np.quantile(incidence[38],[0,.25,.5,.75,1]).tolist(),
            h_beats_old=int((weight[33]>weight[old,j]).sum()))
    selected={}
    x,y=e['native_xy'].T
    for mode in ['raster','consensus']:
        dest=root/frame/mode;out=np.array(Image.open(dest/'source_ids.png'))
        selected[mode]=dict(changed_diagnostic_sources=int((out[y,x]!=old).sum()),
            h_diagnostic_sources=int((out[y,x]==33).sum()),source_ids_sha256=sha(dest/'source_ids.png'))
    save(root/'normal_diagnosis.json',dict(normals=values,source_controls=selected,
        face_to_mean_vertex_normal_degrees=np.quantile(np.degrees(np.arccos(np.clip((normal*vertex).sum(1),-1,1))),[0,.25,.5,.75,1]).tolist(),
        first_edge_length_quantiles=np.quantile(np.linalg.norm(v[t[f]][:,1]-v[t[f]][:,0],axis=1),[0,.5,1]).tolist(),
        input_hashes={str(path):sha(path),record['mesh']:sha(record['mesh']),str(Path(__file__).resolve()):sha(__file__)},
        interpretation='Normals are locally coherent but grazing. Simple vertex-normal smoothing does not favor H/C more often. This is not a proof of true mesh geometry.'))
    print(values,selected,flush=True)


if __name__=='__main__':main()
