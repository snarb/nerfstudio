"""Check observed-space safety and locate where residual jaw coverage is lost."""
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
from joint_temporal_texture import read,sha,atomic_json,cameras
from study_poisson_jaw_completion import OUT,SOURCE,FRAME
from study_confidence_depth_prior import load_real
from guard_jaw_measured_depth import measured_pixel_veto
from guard_poisson_jaw_completion import anchored_admission
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth


def run():
    req=read(OUT/'request.json');r=read(OUT/'result.json');bq=read(SOURCE/'request.json')
    assert sha(OUT/'request.json')==r['request_sha256']
    for p,h in r['hashes'].items():assert sha(OUT/p)==h
    assert sha(req['source_mesh'])==req['source_mesh_sha256']
    assert sha(Path(__file__).with_name('study_poisson_jaw_completion.py'))==req['script_sha256']
    rows,depths,receipt=load_real(Path(bq['depth_root']),FRAME);assert receipt==bq['depth_receipt']
    base=o3d.io.read_triangle_mesh(req['source_mesh']);v=np.asarray(base.vertices);t=np.asarray(base.triangles)
    local=o3d.io.read_triangle_mesh(str(OUT/'local_raw.ply'));lv=np.asarray(local.vertices);lt=np.asarray(local.triangles)
    raw=o3d.io.read_triangle_mesh(str(OUT/'poisson_raw.ply'));rv=np.asarray(raw.vertices);rt=np.asarray(raw.triangles)
    evidence=np.load(OUT/'proposal_evidence.npz');used=evidence['raw_vertex_ids'];pi=evidence['raw_triangle_ids']
    np.testing.assert_array_equal(lv,np.concatenate([v,rv[used]]));np.testing.assert_array_equal(lt[:len(t)],t)
    np.testing.assert_array_equal(lv[lt[len(t):]],rv[rt[pi]])
    nearest=scene_for(v,t).compute_closest_points(o3d.core.Tensor(rv[used].astype(np.float32)))['points'].numpy()
    np.testing.assert_allclose(nearest,evidence['closest_points'],atol=1e-7,rtol=0)
    assert np.linalg.norm(rv[used]-nearest,axis=1).max()<=req['maximum_original_surface_distance']+1e-7
    admission=read(OUT/'admission/result.json');ar=read(OUT/'admission/request.json')
    assert sha(OUT/'admission/samples.npz')==admission['arrays_sha256']
    assert sha(OUT/'admission/request.json')==admission['request_sha256']
    assert sha(Path(__file__).with_name('guard_poisson_jaw_completion.py'))==ar['script_sha256']
    a=np.load(OUT/'admission/samples.npz');s=a['semantic_ids']
    np.testing.assert_array_equal(a['anchored'],anchored_admission(a['strict'],a['anchor_votes'],a['free'],a['mask_support'][s],a['mask_outside'][s]))
    audits=[]
    for arm in ['strict','anchored']:
        folder=OUT/arm/FRAME;result=read(folder/'result.json')
        assert sha(folder/'request.json')==result['request_sha256']
        for p,h in result['hashes'].items():assert sha(folder/p)==h
        candidate=o3d.io.read_triangle_mesh(str(folder/'mesh.ply'));cv=np.asarray(candidate.vertices);ct=np.asarray(candidate.triangles)
        retained=np.load(folder/'evidence.npz')['retained_proposal_ids']
        np.testing.assert_array_equal(cv,lv);np.testing.assert_array_equal(ct,np.concatenate([t,evidence['proposals'][retained]]))
        scene=scene_for(cv,ct);checks=[]
        for camera,depth in zip(rows,depths):
            for offset in [0,.5]:
                ids,count,raw_count=measured_pixel_veto(scene,camera,depth,rows,depths,len(t),len(ct),offset)
                assert len(ids)==0 and count==0
                checks.append(dict(camera=camera['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw_count))
        _,components,_=candidate.cluster_connected_triangles()
        record=dict(arm=arm,mesh_sha256=sha(folder/'mesh.ply'),original_prefix_exact=True,native_ray_checks=checks,
            components=len(components),nonmanifold_edges=len(candidate.get_non_manifold_edges(allow_boundary_edges=True)))
        atomic_json(folder/'audit.json',record);audits.append(record);print(arm,'124 checks passed',flush=True)
    camera=next(c for c in rows if c['physical_camera']=='F004_E005_1210FP')
    polygon=[[610,1090],[655,1120],[693,1134],[710,1146],[719,1152],[711,1215],[705,1230],[690,1200],[668,1175],[635,1145],[610,1125]]
    mask=np.zeros((1920,1080),np.uint8);cv2.fillPoly(mask,[np.array(polygon,np.int32)],1);mask=mask.astype(bool)
    curved=np.rot90(np.load('/mnt/data/dec5_subdivided_jaw_caps/curved/rgb/001193/F004_E005_1210FP/repaired/frames/001193/target_depth.npz')['depth'])
    missing=mask&(curved==0);coverage=[];local_ids=None
    for name,sv,st in [('full_poisson',np.concatenate([v,rv]),np.concatenate([t,rt+len(v)])),('local_proposals',lv,lt)]:
        d,ids,_=camera_depth(scene_for(sv,st),camera);d,ids=np.rot90(d),np.rot90(ids)
        coverage.append(dict(stage=name,misses=int((missing&~np.isfinite(d)).sum()),newly_covered=int((missing&np.isfinite(d)).sum())))
        if name=='local_proposals':local_ids=ids[missing&np.isfinite(d)]-len(t)
    reasons=[];lookup={int(raw_id):i for i,raw_id in enumerate(s)}
    for idx in np.unique(local_ids):
        i=int(idx);record=dict(proposal=i,pixels=int((local_ids==idx).sum()),semantic_pass=i in lookup)
        if i in lookup:
            j=lookup[i];record.update(strict=bool(a['strict'][j]),anchored=bool(a['anchored'][j]),
                anchor_votes=a['anchor_votes'][j].tolist(),votes=a['votes'][j].tolist(),free_veto=bool(a['free'][:,j].any()))
        else:record.update(mask_support=int(a['mask_support'][i]),mask_veto=int(a['mask_outside'][i]))
        reasons.append(record)
    atomic_json(OUT/'audit.json',dict(script_sha256=sha(__file__),fresh_native_checks=248,audits=audits,
        original_curved_remaining=int(missing.sum()),coverage=coverage,proposal_reasons=reasons,
        poisson_solve_repeated=False,initial_depth_sample_evidence_recomputed=False,heldout_used=False,production_accepted=False))
    print('Residual curved-cap misses',int(missing.sum()),coverage,reasons,flush=True)


if __name__=='__main__':run()
