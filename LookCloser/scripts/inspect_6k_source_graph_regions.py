"""Check whether apparent screen-space source slivers are mesh-label islands."""
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from joint_temporal_texture import read, sha, atomic_json
from hard_surface_texture import face_adjacency
from diagnose_6k_source_seams import ROOT, NEW


def main():
    import open3d as o3d
    frame='001083';out=ROOT/'graph_regions';out.mkdir(exist_ok=False)
    request=read(NEW/'request.json');parent=Path(request['parent'])
    source=next(r for r in read(parent/'request.json')['inventory'] if r['frame_id']==frame)
    assert sha(source['mesh'])==source['mesh_sha256']
    mesh=o3d.io.read_triangle_mesh(source['mesh']);triangles=np.asarray(mesh.triangles)
    labels=np.load(NEW/'frames'/frame/'face_source_labels.npy')
    edges=face_adjacency(triangles);same=edges[labels[edges[:,0]]==labels[edges[:,1]]]
    graph=coo_matrix((np.ones(len(same)),(same[:,0],same[:,1])),shape=(len(triangles),len(triangles))).tocsr()
    count,component=connected_components(graph,directed=False);sizes=np.bincount(component)
    degree=np.bincount(edges.ravel(),minlength=len(triangles));records=[]
    prior=ROOT/'mesh_label_attribution/result.json'
    for row in read(prior)['records']:
        assert sha(row['evidence'])==row['evidence_sha256']
        a=np.load(row['evidence']);face=a['face_ids'][a['valid']];sources=[]
        for camera in np.unique(labels[face]):
            selected=np.unique(face[labels[face]==camera]);cc=np.unique(component[selected])
            sizes_in_roi=[dict(component=int(c),mesh_faces=int(sizes[c]),
                roi_faces=int((component[selected]==c).sum())) for c in cc]
            sources.append(dict(source=int(camera),roi_faces=len(selected),
                faces_with_fewer_than_three_neighbors=int((degree[selected]<3).sum()),regions=sizes_in_roi))
        records.append(dict(region=row['region'],sources=sources))
    atomic_json(out/'result.json',dict(records=records,mesh_label_components=count,
        adjacency_edges=len(edges),script_sha256=sha(__file__),
        input_hashes={str(prior):sha(prior),source['mesh']:sha(source['mesh']),
            str(NEW/'frames'/frame/'face_source_labels.npy'):sha(NEW/'frames'/frame/'face_source_labels.npy')},
        screen_components_are_not_mesh_components=True,production_modified=False))
    print(records,flush=True)


if __name__=='__main__':main()
