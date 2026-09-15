"""Replay direct evidence, measured seed certificates and all native depth vetoes."""
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from admit_mhr_local_patch_depth import OUT,CANDIDATES,PRIOR,ARMS,inputs,Scene2,interpolation_admission
from study_multiview_face_prior import read,save,sha
from study_confidence_depth_prior import unproject
from study_jaw_repair_transfer import mask_votes
from study_jaw_depth_footprint import train_reference_votes
from study_jaw_train_confidence import footprint_veto
from guard_jaw_measured_depth import initial_admission,measured_pixel_veto
from diagnose_jaw_measured_depth import barycentric_samples
from local_surface_certificate import certify

def main():
    import open3d as o3d
    request=read(OUT/'request.json');summary=read(OUT/'result.json');cq,rows,depths,masks,names,binding=inputs();assert binding==request['inputs'];assert summary['request_sha256']==sha(OUT/'request.json');checked=[]
    def check(path,digest):assert sha(path)==digest,str(path);checked.append(str(path))
    check(Path(__file__).with_name('admit_mhr_local_patch_depth.py'),request['script_sha256'])
    for name,digest in request['helpers'].items():check(Path(__file__).with_name(name),digest)
    check(CANDIDATES/'request.json',request['candidate_request_sha256']);check(CANDIDATES/'result.json',request['candidate_result_sha256'])
    old=o3d.io.read_triangle_mesh(cq['source_mesh']);old.compute_triangle_normals();ov=np.asarray(old.vertices);ot=np.asarray(old.triangles);oldscene=Scene2(ov,ot);initial=np.load(PRIOR/'initial.npz');total_samples=total_certificates=nativechecks=0;details=[]
    for arm in ARMS:
        root=OUT/arm;sr=read(root/'result.json');check(root/'admission.npz',sr['admission_sha256']);raw=o3d.io.read_triangle_mesh(str(CANDIDATES/arm/'local_raw.ply'));v=np.asarray(raw.vertices);pp=np.load(CANDIDATES/arm/'proposal_evidence.npz')['proposals'];a=np.load(root/'admission.npz');ms,mo=mask_votes(v,pp,rows,masks,names);np.testing.assert_array_equal(ms,a['mask_support']);np.testing.assert_array_equal(mo,a['mask_outside']);semantic=np.flatnonzero((ms>=2)&(mo==0));np.testing.assert_array_equal(semantic,a['semantic_ids']);p=pp[semantic];points=barycentric_samples(v[p]);np.testing.assert_array_equal(points,a['points']);votes,refs=train_reference_votes(points.reshape(-1,3),rows,depths);np.testing.assert_array_equal(votes.reshape(-1,10),a['votes']);np.testing.assert_array_equal(refs.reshape(-1,10),a['references']);free=footprint_veto(points.reshape(-1,3),rows,depths).reshape(62,-1,10);np.testing.assert_array_equal(free,a['trusted_free']);strict=initial_admission(votes.reshape(-1,10),free,ms[semantic],mo[semantic]);np.testing.assert_array_equal(strict,a['strict']);total_samples+=len(votes)
        c=np.load(root/'certificates.npz');used=np.unique(p);np.testing.assert_array_equal(used,c['query_ids']);prior_v=np.load(PRIOR/arm/'fit.npz')['vertices'];mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(prior_v),o3d.utility.Vector3iVector(initial['triangles']));mesh.compute_triangle_normals();scene=Scene2(prior_v,initial['triangles']);qh=scene.compute_closest_points(o3d.core.Tensor(v[used].astype(np.float32)));qn=np.asarray(mesh.triangle_normals)[qh['primitive_ids'].numpy()];np.testing.assert_array_equal(qn,c['query_normals'])
        seeds=c['observed_seed_points'];seedcamera=c['observed_seed_camera'];seedpixel=c['observed_seed_pixel'];assert len(np.unique(np.column_stack((seedcamera,seedpixel)),axis=0))==len(seeds)
        for ci in np.unique(seedcamera):
            take=seedcamera==ci;xy=seedpixel[take];d=depths[ci][xy[:,1],xy[:,0]];np.testing.assert_array_equal(seeds[take],unproject(rows[ci],xy[:,0],xy[:,1],d))
        sv,_=train_reference_votes(seeds,rows,depths);np.testing.assert_array_equal(sv,c['observed_seed_votes']);h=scene.compute_closest_points(o3d.core.Tensor(seeds.astype(np.float32)));pd=np.linalg.norm(h['points'].numpy()-seeds,axis=1);h=oldscene.compute_closest_points(o3d.core.Tensor(seeds.astype(np.float32)));od=np.linalg.norm(h['points'].numpy()-seeds,axis=1);sn=np.asarray(old.triangle_normals)[h['primitive_ids'].numpy()];np.testing.assert_array_equal(pd,c['seed_prior_distance']);np.testing.assert_array_equal(od,c['seed_original_distance']);np.testing.assert_array_equal(sn,c['seed_normals']);valid=(sv>=3)&(pd<=.0005)&(od<=.001);np.testing.assert_array_equal(valid,c['valid_seed_mask']);seeds=seeds[valid];sn=sn[valid];certificate=np.zeros(len(used),bool);notes=[]
        d,neighbors=cKDTree(seeds).query(v[used],k=min(24,len(seeds)))
        for i in range(len(used)):
            index=neighbors[i][(d[i]<=.003)&((sn[neighbors[i]]@qn[i])>=.5)];certificate[i],note=certify(v[used[i]],qn[i],seeds[index],tolerance=.0005);notes.append(note)
        np.testing.assert_array_equal(certificate,c['certificate']);assert notes==read(root/'certificates.json')['notes'];lookup=np.zeros(len(v),bool);lookup[used]=certificate;keep,prior=interpolation_admission(strict,lookup,p,free,ms[semantic],mo[semantic]);np.testing.assert_array_equal(keep,a['interpolated']);np.testing.assert_array_equal(prior,a['certified_prior']);total_certificates+=len(used)
        branchstats={}
        for branch,admitted in [('strict',strict),('interpolated',keep)]:
            folder=root/branch;r=read(folder/'result.json')
            for name,digest in r['hashes'].items():check(folder/name,digest)
            final=o3d.io.read_triangle_mesh(str(folder/'mesh.ply'));fv=np.asarray(final.vertices);ft=np.asarray(final.triangles);ids=np.load(folder/'evidence.npz')['retained_proposal_ids'];assert np.isin(ids,semantic[admitted]).all();np.testing.assert_array_equal(fv,v);np.testing.assert_array_equal(fv[:len(ov)],ov);np.testing.assert_array_equal(ft,np.concatenate((ot,pp[ids])));scene=Scene2(fv,ft);checks=[]
            for ci,(camera,depth) in enumerate(zip(rows,depths)):
                for offset in [0,.5]:
                    implicated,count,rawcount=measured_pixel_veto(scene,camera,depth,rows,depths,len(ot),len(ft),offset);assert not len(implicated) and count==0;checks.append(dict(camera=camera['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=rawcount));nativechecks+=1
                if (ci+1)%20==0:print('audit',arm,branch,ci+1,flush=True)
            branchstats[branch]=dict(added=len(ids),native_checks=checks)
        details.append(dict(arm=arm,branches=branchstats,certificate_failure_reasons=dict(Counter(x['reason'] for x in notes)),strict_and_interpolated_mesh_identical=sha(root/'strict/mesh.ply')==sha(root/'interpolated/mesh.ply')))
    save(OUT/'audit.json',dict(status='passed',checked_bindings=len(checked),checked_paths=checked,source_geometric_depth_hashes=len(binding['depth_receipt']['depth_sha256']),sample_votes_and_footprints_replayed=total_samples,vertex_certificates_replayed=total_certificates,native_ray_checks_replayed=nativechecks,details=details,
        source_rebinding_explicit=True,original_prefix_exact=True,production_accepted=False,script_sha256=sha(__file__),
        inventory={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='audit.json'}))
    print('audit passed',total_samples,'samples',total_certificates,'certificates',nativechecks,'nativechecks',flush=True)

if __name__=='__main__':main()
