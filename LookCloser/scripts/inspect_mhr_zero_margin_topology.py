"""Posthoc localization of topology regressions; never constructs a patch."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import study_mhr_zero_margin as study
from study_multiview_face_prior import read,save,sha
from triangulate_face_prior import quantiles


def main():
    import open3d as o3d
    study.configure();fit=study.fit
    a=np.load(study.FINAL/'fit.npz');v,t,n=a['vertices'],a['triangles'],a['neutral']
    before=np.load(study.CONTROL/'review_v2/silhouette_topology.npz');after=np.load(study.FINAL/'review_v2/silhouette_topology.npz')
    old=set(map(tuple,before['strict_pairs']))
    new=np.array([p for p in after['strict_pairs'] if tuple(p) not in old],int).reshape(-1,2)
    reversed_new=np.setdiff1d(after['reversed_triangles'],before['reversed_triangles'])
    primary=np.load(study.FINAL/'locality/silhouette.npz')
    residual=np.load(study.ROOT/'comparison/residual.npz')
    groups={'new_crossing_vs_margin2':np.unique(new),'new_normal_change_vs_margin2':reversed_new}
    stats={}
    for name,ids in groups.items():
        center=n[t[ids]].mean(1);scene=fit.Scene2(v,t[ids]);distances={}
        for label,points in [('primary_rim',primary['rim_points']),('primary_hit',primary['points']),('residual_hit',residual['margin0_points'])]:
            nearest=scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))['points'].numpy()
            distances[label]=quantiles(np.linalg.norm(nearest-points,axis=1))
            distances[label]['minimum']=float(np.linalg.norm(nearest-points,axis=1).min())
        stats[name]=dict(triangles=len(ids),neutral_y=quantiles(center[:,1]),neutral_z=quantiles(center[:,2]),
            front_centroids=int((center[:,2]>=0).sum()),back_centroids=int((center[:,2]<0).sum()),
            all_vertices_lower_band=int(((n[t[ids],1]>=135)&(n[t[ids],1]<=153)).all(1).sum()),
            distance_to_queries=distances,
            residual_first_hit_facets=int(np.isin(residual['margin0_faces'],ids).sum()))
    dest=study.ROOT/'topology_localization';dest.mkdir(exist_ok=False)
    np.savez_compressed(dest/'evidence.npz',new_pairs=new,new_crossing_facets=groups['new_crossing_vs_margin2'],new_normal_changes=reversed_new)
    _,rows,_,_,_,_=fit.prepare()
    inference=read('/mnt/data/dec5_multiview_face_prior/inference.json')
    predictions={r['camera']:r for r in inference['records'] if r['frame']=='001193' and r['detected']==1}
    files=[]
    for prefix in ['C004_E','G004_B','M004_B']:
        row=next(r for r in rows if r['physical_camera'].startswith(prefix));name=row['physical_camera']
        x0,y0,x1,y1=predictions[name]['native_review_box'];y1=min(1500,y1+180)
        source=study.FINAL/'review_v2'/(name+'_clay.png');im=Image.open(source).convert('RGB')
        im=im.crop((im.width*3//4,24,im.width,im.height));draw=ImageDraw.Draw(im)
        # Wire overlay is anatomical localization, not an occlusion-aware render.
        for label,color in [('new_crossing_vs_margin2',(255,50,50)),('new_normal_change_vs_margin2',(50,220,255))]:
            uv,z,_=fit.project_jacobian(v,row);coords=np.c_[uv[:,1],1919-uv[:,0]]-np.array([x0,y0])
            for face in t[groups[label]]:
                if (z[face]>0).all():draw.line([tuple(p) for p in coords[np.r_[face,face[0]]]],fill=color,width=1)
        path=dest/(name+'_unsafe_wire.png');im.save(path);files.append(dict(path=str(path),sha256=sha(path)))
    inputs=[study.FINAL/'fit.npz',study.CONTROL/'review_v2/silhouette_topology.npz',study.FINAL/'review_v2/silhouette_topology.npz',study.FINAL/'locality/silhouette.npz',study.ROOT/'comparison/residual.npz']
    save(dest/'result.json',dict(new_strict_pairs_vs_margin2=len(new),localization=stats,
        input_hashes={str(p):sha(p) for p in inputs},files=files,script_sha256=sha(__file__),
        wire_overlay_not_visibility_filtered=True,no_patch_constructed=True))
    print(stats,flush=True)


if __name__=='__main__':main()
