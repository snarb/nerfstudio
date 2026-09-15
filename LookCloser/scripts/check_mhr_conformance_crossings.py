"""Strict transverse triangle-crossing witnesses, separate from normal changes."""
import numpy as np
from study_multiview_face_prior import read,save,sha
from review_mhr_conformance_independent import ROOT,SOURCE,PARENT,ARMS

def transverse_crossings(a,b,epsilon=1e-6):
    """Noncoplanar segment-through-triangle interior, excluding mere edge contact."""
    answer=np.zeros(len(a),bool)
    for source,target in [(a,b),(b,a)]:
        e1=target[:,1]-target[:,0];e2=target[:,2]-target[:,0]
        for i,j in [(0,1),(1,2),(2,0)]:
            direction=source[:,j]-source[:,i];h=np.cross(direction,e2);det=np.sum(e1*h,axis=1);scale=np.linalg.norm(direction,axis=1)*np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1);valid=abs(det)>1e-10*scale
            inverse=np.divide(1.,det,out=np.zeros_like(det),where=valid);s=source[:,i]-target[:,0];u=inverse*np.sum(s*h,axis=1);q=np.cross(s,e1);v=inverse*np.sum(direction*q,axis=1);t=inverse*np.sum(e2*q,axis=1)
            answer|=valid&(u>epsilon)&(v>epsilon)&(u+v<1-epsilon)&(t>epsilon)&(t<1-epsilon)
    return answer

def main():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    initial=np.load(SOURCE/'initial.npz');tri=initial['triangles'];neutral=initial['neutral'];centers=neutral[tri].mean(1);records=[];basepairs=set()
    rim=np.load(PARENT/'probe_head20_neck6/head20_neck6.npz')['rim_points']
    for arm in ['input_prior',*ARMS]:
        v=np.load(PARENT/'head20_neck6/fit.npz')['vertices'] if arm=='input_prior' else np.load(SOURCE/arm/'fit.npz')['vertices'];g=np.load(ROOT/(arm+'_geometry.npz'));pairs=g['self_intersection_pairs'];passed=transverse_crossings(v[tri[pairs[:,0]]],v[tri[pairs[:,1]]]);proper=pairs[passed];ps=set(map(tuple,proper))
        if arm=='input_prior':basepairs=ps
        new=np.array(sorted(ps-basepairs),dtype=np.int64).reshape(-1,2);mid=centers[proper].mean(1);labels=np.full(len(mid),'other',dtype='<U16')
        labels[(mid[:,1]>=154)&(mid[:,1]<157)&(abs(mid[:,0])<5)]='mouth_lip';labels[(mid[:,1]>=157)&(mid[:,1]<160)&(abs(mid[:,0])<3)]='nose';labels[(mid[:,1]>=160)&(mid[:,1]<164)&(abs(mid[:,0])<6)]='eye';labels[abs(mid[:,0])>=6]='ear_side'
        record=dict(arm=arm,strict_transverse_pairs=len(proper),new_strict_pair_ids=len(new),strict_pairs_touching_y135_153=int((((centers[proper,1]>=135)&(centers[proper,1]<153)).any(1)).sum()),
            neutral_region_pair_counts={x:int((labels==x).sum()) for x in np.unique(labels)},
            note='A strict crossing is an intersection witness, not a normal-dot test. New pair IDs can be a changed contact pattern in an already intersecting anatomical region.')
        intersecting=np.unique(proper);surface=scene_for(v,tri[intersecting]);nearest=surface.compute_closest_points(o3d.core.Tensor(rim.astype(np.float32)))['points'].numpy()
        record['minimum_intersecting_surface_distance_to_requested_rim']=float(np.linalg.norm(nearest-rim,axis=1).min())
        record['minimum_neutral_vertex_y_of_intersecting_triangles']=float(neutral[tri[intersecting],1].min())
        np.savez_compressed(ROOT/(arm+'_strict_crossings.npz'),pairs=proper,new_pairs=new,neutral_region=labels);records.append(record);print(record,flush=True)
    save(ROOT/'strict_crossings.json',dict(records=records,script_sha256=sha(__file__),review_result_sha256=sha(ROOT/'result.json'),barycentric_and_segment_margin=1e-6,
        classification='Approximate neutral-coordinate anatomical bands, independently visually checked',source_geometry_changed=False))

if __name__=='__main__':main()
