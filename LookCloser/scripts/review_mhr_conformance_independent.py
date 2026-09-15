"""Read-only independent conformance review: leakage, orientation and intersections."""
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
from scipy import sparse
from scipy.sparse.linalg import lsmr
from study_multiview_face_prior import read,save,sha,portrait_to_native,CROP
from triangulate_face_prior import projection_matrices,project,quantiles
from conform_mhr_measured_surface import uniform_laplacian,barycentric_matrix

ROOT=Path('/mnt/data/dec5_mhr_conformance_independent_review')
SOURCE=Path('/mnt/data/dec5_mhr_measured_conformance')
PARENT=Path('/mnt/data/dec5_mhr_local_head_prior')
RGB=Path('/mnt/data/dec5_multiview_face_prior')
ARMS=['smooth025','smooth100','smooth400']

def train_only_replay(initial,base,obs,recipe):
    """Remove every validation record before any correspondence or solve."""
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    select=~obs['validation'][obs['camera']];points=obs['points'][select];normals=obs['normals'][select];neck=obs['neck'][select]
    tri=initial['triangles'];n=len(base);active=np.flatnonzero(initial['neutral'][:,1]>135);headtri=tri[(initial['neutral'][tri,1]>140).all(1)]
    lap=uniform_laplacian(tri,n)[active][:,active];mag=sparse.eye(len(active),format='csr')/(.006*np.sqrt(len(active)));current=base.copy()
    for outer in range(4):
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(current),o3d.utility.Vector3iVector(headtri));mesh.compute_triangle_normals();scene=scene_for(current,headtri)
        hit=scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)));ids=hit['primitive_ids'].numpy();uv=hit['primitive_uvs'].numpy();bary=np.c_[1-uv.sum(1),uv];distance=np.linalg.norm(hit['points'].numpy()-points,axis=1);dot=np.sum(np.asarray(mesh.triangle_normals)[ids]*normals,axis=1);use=(distance<=.006)&(dot>=.25)
        assoc=barycentric_matrix(headtri[ids[use]],bary[use],n);robust=1/np.sqrt(1+(distance[use]/.001)**2);group=np.where(neck[use],int((use&neck).sum()),int((use&~neck).sum()));weight=np.sqrt(robust/(2*group))/.001
        data=sparse.diags(weight)@assoc[:,active];target=(points[use]-assoc@base)*weight[:,None];smooth=lap*2/(.0005*np.sqrt(len(active)));system=sparse.vstack((data,smooth,mag),format='csr');rhs=np.vstack((target,np.zeros((2*len(active),3))))
        displacement=np.column_stack([lsmr(system,rhs[:,axis],atol=1e-8,btol=1e-8,maxiter=600)[0] for axis in range(3)]);current=base.copy();current[active]+=displacement
    return current

def main():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from study_confidence_depth_prior import unproject
    ROOT.mkdir(exist_ok=False);started=time.monotonic();protocol=read(SOURCE/'protocol.json');check=[]
    for p,digest in protocol['input_hashes'].items():assert sha(p)==digest,p;check.append(p)
    assert sha(Path(__file__).with_name('conform_mhr_measured_surface.py'))==protocol['script_sha256']
    assert sha(protocol['original_mesh'])==protocol['original_mesh_sha256'];initial=np.load(SOURCE/'initial.npz');obs=np.load(SOURCE/'anchors.npz');base=np.load(PARENT/'head20_neck6/fit.npz')['vertices'];tri=initial['triangles'];neutral=initial['neutral'];centers=neutral[tri].mean(1)
    replay=train_only_replay(initial,base,obs,protocol['recipe']);expected=np.load(SOURCE/'smooth400/fit.npz')['vertices'];difference=np.linalg.norm(replay-expected,axis=1);np.savez_compressed(ROOT/'train_only_replay.npz',vertices=replay,difference=difference)
    print('Train-only replay maxdifference',difference.max(),flush=True)
    datasets=[('input_prior',base)]+[(a,np.load(SOURCE/a/'fit.npz')['vertices']) for a in ARMS];pairsets={};records=[];oldcross=np.cross(base[tri[:,1]]-base[tri[:,0]],base[tri[:,2]]-base[tri[:,0]])
    rim=np.load(PARENT/'probe_head20_neck6/head20_neck6.npz')['rim_points'];rows=read(PARENT/'anchors.json')['cameras'];predictions={r['camera']:r for r in read(RGB/'inference.json')['records'] if r['frame']=='001193' and r['detected']==1};files=[]
    for name,v in datasets:
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(tri));mesh.compute_triangle_normals();t0=time.monotonic();pairs=np.asarray(mesh.get_self_intersecting_triangles(),dtype=np.int64).reshape(-1,2);pairs=np.sort(pairs,axis=1)
        # Explicitly exclude all topologically adjacent pairs from the proof set.
        independent=np.array([not(set(tri[a])&set(tri[b])) for a,b in pairs],bool);pairs=pairs[independent];pairset=set(map(tuple,pairs));pairsets[name]=pairset;newpairs=np.array(sorted(pairset-pairsets['input_prior']),dtype=np.int64).reshape(-1,2)
        cross=np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]]);norm=np.linalg.norm(cross,axis=1);oldnorm=np.linalg.norm(oldcross,axis=1);cos=np.sum(cross*oldcross,axis=1)/(norm*oldnorm);reversed=np.flatnonzero(cos<=0);ratio=norm/oldnorm
        lower=(centers[:,1]>=135)&(centers[:,1]<153);newinlower=np.unique(newpairs[lower[newpairs].any(1)]) if len(newpairs) else np.array([],int)
        record=dict(arm=name,nonadjacent_self_intersection_pairs=len(pairs),new_pair_ids_vs_input=len(newpairs),new_pairs_touching_lower_head_or_neck=int(lower[newpairs].any(1).sum()) if len(newpairs) else 0,
            normal_dot_reversals=len(reversed),reversals_below_neutral_y153=int((centers[reversed,1]<153).sum()),
            reversal_neutral_bounds=[centers[reversed].min(0).tolist(),centers[reversed].max(0).tolist()] if len(reversed) else None,
            reversal_area_ratio=quantiles(ratio[reversed]),reversal_normal_cosine=quantiles(cos[reversed]),
            reversed_triangles_in_new_intersection_pairs=int(np.isin(reversed,newpairs).sum()),intersection_seconds=time.monotonic()-t0)
        if len(reversed):
            revscene=scene_for(v,tri[reversed]);nearest=revscene.compute_closest_points(o3d.core.Tensor(rim.astype(np.float32)))['points'].numpy();record['minimum_reversal_surface_distance_to_requested_rim']=float(np.linalg.norm(nearest-rim,axis=1).min())
        np.savez_compressed(ROOT/(name+'_geometry.npz'),reversed_triangles=reversed,normal_cosine=cos,area_ratio=ratio,neutral_centers=centers,self_intersection_pairs=pairs,new_intersection_pairs=newpairs)
        records.append(record);print(name,record,flush=True)
        if name=='input_prior':continue
        scene=scene_for(v,tri);normals=np.asarray(mesh.triangle_normals);revmask=np.zeros(len(tri),bool);revmask[reversed]=True;intmask=np.zeros(len(tri),bool);intmask[np.unique(newpairs)]=True
        for prefix in ['G004_B','M004_B','E004_B']:
            row=next(r for r in rows if r['physical_camera'].startswith(prefix));cam=row['physical_camera'];x0,y0,x1,y1=predictions[cam]['native_review_box'];y1=min(1550,y1+80);w=x1-x0;h=y1-y0;yy,xx=np.mgrid[y0:y1,x0:x1];xy=portrait_to_native(np.column_stack((xx.ravel(),yy.ravel())));c=np.asarray(row['transform_matrix'])[:3,3];d=unproject(row,xy[:,0],xy[:,1],np.ones(len(xy)))-c;unit=d/np.linalg.norm(d,axis=1,keepdims=True)
            hit=scene.cast_rays(o3d.core.Tensor(np.column_stack((np.broadcast_to(c,d.shape),d)).astype(np.float32)));depth=hit['t_hit'].numpy();tid=hit['primitive_ids'].numpy();ok=np.isfinite(depth);rgb=np.full((len(d),3),20,np.uint8);rgb[ok]=(70+170*abs(np.sum(normals[tid[ok]]*-unit[ok],axis=1)))[:,None];rgb[ok&np.isin(tid,np.flatnonzero(intmask))]=[255,180,0];rgb[ok&np.isin(tid,reversed)]=[255,30,30]
            clay=Image.fromarray(rgb.reshape(h,w,3));original=Image.open(RGB/'001193'/(cam+'.png')).crop((x0,y0-CROP[1],x1,y1-CROP[1]));panel=Image.new('RGB',(2*w,h+24));panel.paste(original,(0,24));panel.paste(clay,(w,24));draw=ImageDraw.Draw(panel);draw.text((3,4),cam+' train RGB',fill='white');draw.text((w+3,4),name+' red: normal change; amber: new intersection',fill='white');path=ROOT/(name+'_'+cam+'.png');panel.save(path);files.append(dict(path=str(path),sha256=sha(path)))
    save(ROOT/'result.json',dict(geometry=records,validation_rows_removed_before_replay=int(obs['validation'][obs['camera']].sum()),train_only_replay_max_vertex_difference=float(difference.max()),
        independently_replayed_arm='smooth400',source_protocol_sha256=sha(SOURCE/'protocol.json'),source_fit_hashes={a:sha(SOURCE/a/'fit.npz') for a in ARMS},
        original_mesh_sha256=sha(protocol['original_mesh']),original_unchanged=True,files=files,script_sha256=sha(__file__),elapsed_seconds=time.monotonic()-started,
        normal_reversal_is_not_alone_proof_of_self_intersection=True,intersection_test='Open3D triangle intersections excluding shared-vertex pairs; new pair IDs relative to input prior, not proof newly penetrating vs existing contact',
        validation_caveat='Eight reserved fitting cameras, not independent of existing62-camera COLMAP mesh. Closest-surface evaluation does not establish visibility/free-space safety.',production_approval=False))

if __name__=='__main__':main()
