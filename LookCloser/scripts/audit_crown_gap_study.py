"""Replay the crown stage attribution and rejected local-consensus screen."""
import argparse
from pathlib import Path
import subprocess
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from joint_temporal_texture import read, sha, atomic_json, cameras
from diagnose_crown_completion_gap import ROOT as ATTR, INSET, RAW, FRAMES, LABELS
from probe_consensus_head_normals import ROOT, SETTINGS, MOVIE, MASKS
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from study_confidence_depth_prior import REGIONS, region_masks
from study_jaw_repair_transfer import mask_votes


def run():
    hashes={};records=[]
    def bind(path, expected=None):
        actual=sha(path)
        if expected is not None: assert actual==expected,str(path)
        hashes[str(Path(path).resolve())]=actual
    for frame in FRAMES:
        folder=ATTR/frame;q=read(folder/'request.json');r=read(folder/'result.json');a=np.load(folder/'evidence.npz')
        bind(folder/'request.json',r['request_sha256'])
        for p,h in q['scripts'].items():bind(p,h)
        for p,h in r['hashes'].items():bind(folder/p,h)
        config=read(INSET/frame/'request.json');bind(INSET/frame/'request.json',q['inset_config_sha256'])
        bind(config['source_mesh'],q['source_mesh_sha256']);bind(RAW/frame/'poisson_raw.ply',q['raw_mesh_sha256'])
        bind(q['gt_path'],q['gt_sha256'])
        for path in [INSET/frame/'inset_001000'/'result.json',INSET/frame/'guarded'/'result.json']:
            for p,h in read(path)['hashes'].items():bind(path.parent/p,h)
        local=np.load(INSET/frame/'inset_001000'/'evidence.npz');g=np.load(INSET/frame/'guarded'/'evidence.npz')
        kept=local['retained_raw_triangle_ids'][g['retained_candidate_triangle_ids']]
        ids=a['raw_triangle_ids']
        # Independent nested membership arithmetic (not producer's stages helper).
        label=np.isin(ids,local['proposal_ids']).astype(np.uint8)+np.isin(ids,local['retained_raw_triangle_ids'])+np.isin(ids,kept)
        np.testing.assert_array_equal(label,a['triangle_stage'])
        for rec in r['records']:
            mask=a[rec['region']+'_region'];query=a['diagnostic_query'];remain=a['remaining_query']
            assert int((mask&query).sum())==rec['raw_prior_covers_old_miss']
            assert int((mask&remain).sum())==rec['still_missing_after_guard']
            for i,name in enumerate(LABELS):
                assert int((mask&remain&(a['pixel_stage']==i)).sum())==rec['remaining_stages'][name]
        c=read(folder/'constraints'/'result.json');e=np.load(folder/'constraints'/'evidence.npz')
        bind(folder/'constraints'/'evidence.npz',c['evidence_sha256']);bind(Path(__file__).with_name('inspect_crown_gap_constraints.py'),c['script_sha256'])
        assert not c['rgb_receipt']['heldout_rgb_loaded']
        for p,h in c['rgb_receipt']['source_rgb_hashes'].items():bind(p,h)
        for name,rec in c['local_failure_counts'].items():
            failed=e[name+'_fail'];assert rec['triangles']==int(failed.sum())
            assert rec['remaining_crown_pixels']==int(e['pixel_weights'][failed].sum())
        for example in c['mask_examples']:bind(folder/'constraints'/example['panel'],example['panel_sha256'])

        out=ROOT/frame;sq=read(out/'request.json');sr=read(out/'result.json');se=np.load(out/'evidence.npz')
        assert sq['settings']==SETTINGS and not sr['measured_guard_passed'] and not sr['production_updated']
        bind(out/'request.json',sr['request_sha256']);bind(Path(__file__).with_name('probe_consensus_head_normals.py'),sq['script_sha256'])
        for p,h in sq['helpers'].items():bind(Path(__file__).with_name(p),h)
        for p,h in sr['hashes'].items():bind(out/p,h)
        original=o3d.io.read_triangle_mesh(config['source_mesh']);original.compute_triangle_normals()
        ov,ot=np.asarray(original.vertices),np.asarray(original.triangles)
        baseline=o3d.io.read_triangle_mesh(str(INSET/frame/'guarded'/'mesh.ply'))
        v,t=np.asarray(baseline.vertices),np.asarray(baseline.triangles)
        mesh=o3d.io.read_triangle_mesh(str(out/'unchecked_mesh.ply'));cv,ct=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
        np.testing.assert_array_equal(cv,v);np.testing.assert_array_equal(ct[:len(t)],t)
        raw=o3d.io.read_triangle_mesh(str(RAW/frame/'poisson_raw.ply'));raw.compute_vertex_normals()
        rt=np.asarray(raw.triangles)
        expected=np.concatenate([t,rt[se['retained_raw_triangle_ids']]+len(ov)])
        np.testing.assert_array_equal(ct,expected)
        rows,_,_=cameras(frame);assert len(rows)==len({x['physical_camera'] for x in rows})==62
        masks=np.load(MASKS/frame/'masks.npz')['masks'];names=read(MASKS/frame/'cameras.json')
        maskresult=read(MASKS/frame/'result.json');bind(MASKS/frame/'result.json',sq['masks_result_sha256'])
        for p,h in maskresult['hashes'].items():bind(MASKS/frame/p,h)
        support,outside=mask_votes(cv[len(ov):],rt[se['rescued']],rows,masks,names)
        np.testing.assert_array_equal(support,se['mask_support']);np.testing.assert_array_equal(outside,se['mask_outside'])
        np.testing.assert_array_equal(se['rescued'][(support>=2)&(outside==0)],se['retained_raw_triangle_ids'])
        # Fresh brute-force distances, independent of the producer's KD-tree.
        selected=np.linspace(0,len(se['vertex_ids'])-1,64,dtype=int)
        head=(ov[ot][:,:,0]>-.03).all(1);centers=ov[ot[head]].mean(1);normals=np.asarray(original.triangle_normals)[head]
        areas=.5*np.linalg.norm(np.cross(ov[ot[head,1]]-ov[ot[head,0]],ov[ot[head,2]]-ov[ot[head,0]]),axis=1)
        pts=cv[len(ov):][se['vertex_ids'][selected]];distances=cdist(pts,centers)
        nearest=np.argsort(distances,axis=1)[:,:32];d=np.take_along_axis(distances,nearest,axis=1)
        weights=np.minimum(areas[nearest],np.median(areas[nearest],axis=1)[:,None])*np.exp(-.5*(d/.0015)**2)*(d<=.003)
        vec=(weights[:,:,None]*normals[nearest]).sum(1);length=np.linalg.norm(vec,axis=1)
        directions=np.asarray(raw.vertex_normals)[se['vertex_ids'][selected]]
        coherence=length/np.maximum(weights.sum(1),1e-30)
        dot=(vec*directions).sum(1)/np.maximum(length,1e-30)
        agreement=(weights*((normals[nearest]*directions[:,None]).sum(2)>=.25)).sum(1)/np.maximum(weights.sum(1),1e-30)
        for name,value in [('coherence',coherence),('normal_dot',dot),('weighted_agreement',agreement)]:
            np.testing.assert_allclose(value,se[name][selected],rtol=1e-10,atol=1e-10)
        passed=((d<=.003).sum(1)>=8)&(coherence>=.5)&(dot>=.25)&(agreement>=.7)
        np.testing.assert_array_equal(passed,se['consensus_pass'][selected])
        native=next(x for x in rows if x['physical_camera']==REGIONS[frame]['camera'])
        moving=next(x['camera'] for x in read(MOVIE/'request.json')['inventory'] if x['frame_id']==frame)
        scenes=[scene_for(v,t),scene_for(cv,ct)]
        for (name,cam),record in zip([('native_train',native),('moving',moving)],sr['views']):
            saved=np.load(out/(name+'_depth.npz'));fresh=[camera_depth(s,cam) for s in scenes]
            for field,data in [('baseline',fresh[0][0]),('candidate',fresh[1][0]),('triangle_ids',fresh[1][1])]:
                np.testing.assert_array_equal(saved[field],data)
            old=np.isfinite(fresh[0][0]);now=np.isfinite(fresh[1][0]);gain=~old&now
            assert record['gained_depth']==int(gain.sum()) and record['lost_depth']==0
            if name=='native_train':assert record['coarse_hair_gained_depth']==int((gain&region_masks(frame)['hair']).sum())
        records.append(dict(frame=frame,brute_force_consensus_samples=64,fresh_raycasts=4,
            original_and_guarded_prefix_exact=True,mask_votes_replayed=True,views=sr['views']))
        print(frame,'attribution, masks, normals and raycast replay passed',flush=True)
    atomic_json(ROOT/'audit.json',dict(records=records,source_hashes=hashes,script_sha256=sha(__file__),
        production_promoted=False,measured_depth_safety_audited=False,quality_approval=False))


def seal():
    review=read(ROOT/'visual_review.json');assert review['operator']=='main_LLM_actual_image_inspection'
    assert review['decision']=='reject_as_material_crown_repair' and not review['production_promoted']
    hashes=read(ROOT/'audit.json')['source_hashes']
    for p,h in review['inspected_images'].items():assert sha(p)==h
    live=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
    assert not [x for x in live if any('python scripts/'+n in x for n in
        ['probe_consensus_head_normals.py','diagnose_crown_completion_gap.py','inspect_crown_gap_constraints.py']) and '/bin/bash' not in x]
    atomic_json(ROOT/'completion.json',dict(workers_terminal=True,visual_review_sha256=sha(ROOT/'visual_review.json'),
        audit_sha256=sha(ROOT/'audit.json'),production_promoted=False,measured_depth_safety_audited=False,
        rgb_rerender_skipped_reason='No material improvement in four geometry screens',status='rejected_screen_complete'))
    for root in [ROOT,ATTR]:
        for p in root.rglob('*'):
            if p.is_file() and p.name!='artifact_manifest.json':hashes[str(p)]=sha(p)
    for name in ['test_crown_completion_gap.py','test_consensus_head_normals.py']:
        p=Path(__file__).resolve().parents[1]/'tests'/name;hashes[str(p)]=sha(p)
    p=Path(__file__).resolve().parents[1]/'experiments'/'dec5_crown_completion_gap.md';hashes[str(p)]=sha(p)
    hashes[str(Path(__file__).resolve())]=sha(__file__)
    for p,h in hashes.items():assert sha(p)==h,p
    atomic_json(ROOT/'artifact_manifest.json',dict(hashes=hashes,production_promoted=False))
    print('Sealed',len(hashes),'bindings',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seal',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:
        h=read(ROOT/'artifact_manifest.json')['hashes']
        for path,value in h.items():assert sha(path)==value,path
        print('Rechecked',len(h),'bindings',flush=True)
    elif a.seal:seal()
    else:run()
