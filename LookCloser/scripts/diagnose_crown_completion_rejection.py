"""Attribute raw crown coverage to actual proposal/evidence rejection stages.

Moving rectangles are diagnostic only and contain true background. Native
camera hair polygons were independently drawn on train RGB in the prior pilot.
Neither region changes the geometry or admission rules.
"""
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.ndimage import label
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras,exr,display,ROOT as CAL
from transfer_close_boundary_completion import ROOT as BASE,SOURCE,MOVIE,FRAMES
from study_confidence_depth_prior import REGIONS,region_masks,load_real
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from review_jaw_repair_transfer import verified_image,panel
from diagnose_jaw_measured_depth import barycentric_samples
from study_jaw_depth_footprint import train_reference_votes
from study_jaw_train_confidence import footprint_veto

ROOT=Path('/mnt/data/dec5_crown_completion_rejection')


def run():
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'request.json').exists():
        raise ValueError('Use fresh diagnosis root')
    q=dict(frames=FRAMES,script_sha256=sha(__file__),parent_artifact_sha256=sha(BASE/'artifact_manifest.json'),
        geometry_changed=False,heldout_used=False,diagnostic_regions_not_quality_metrics=True,
        moving_box=[170,450,970,850],native_regions_source='frozen prior manual train hair polygons',
        missing_depth_not_automatically_missing_anatomy=True)
    atomic_json(ROOT/'request.json',q)
    hashes=read(BASE/'artifact_manifest.json')['hashes']
    for p,h in hashes.items():
        assert sha(p)==h,p
    profiles=read(CAL/'camera_profiles.json');gainmap=dict(zip(profiles['physical_cameras'],profiles['rgb_gain']))
    exposure=read(CAL/'exposure.json')['fixed_exposure_gain'];records=[]
    for frame in FRAMES:
        source=read(SOURCE/frame/'request.json');root=BASE/frame
        original=o3d.io.read_triangle_mesh(source['source_mesh']);nt=len(original.triangles)
        raw=o3d.io.read_triangle_mesh(str(root/'local_raw.ply'));v=np.asarray(raw.vertices);t=np.asarray(raw.triangles)
        final=o3d.io.read_triangle_mesh(str(root/'interpolated'/frame/'mesh.ply'))
        scenes=[scene_for(np.asarray(m.vertices),np.asarray(m.triangles)) for m in [original,raw,final]]
        a=np.load(root/'admission/samples.npz');e=np.load(root/'interpolated'/frame/'evidence.npz')
        lookup=np.full(len(t)-nt,-1,int);lookup[a['semantic_ids']]=np.arange(len(a['semantic_ids']))
        retained=e['retained_proposal_ids'];rows,depths,receipt=load_real(Path(source['depth_root']),frame)
        assert receipt==source['depth_receipt']
        moving=next(r for r in read(MOVIE/'request.json')['inventory'] if r['frame_id']==frame)['camera']
        native=next(r for r in rows if r['physical_camera']==REGIONS[frame]['camera'])
        frame_records=[];queries=[]
        for view,camera in [('moving',moving),('native_train',native)]:
            outputs=[camera_depth(s,camera) for s in scenes]
            ds=[np.rot90(o[0]) for o in outputs];ids=np.rot90(outputs[1][1]);hit=[np.isfinite(d) for d in ds]
            if view=='moving':
                region=np.zeros_like(hit[0]);region[450:850,170:970]=True
                rgb,_=verified_image(MOVIE,frame);box=(170,450,970,850)
            else:
                region=np.rot90(region_masks(frame)['hair'])
                rgb=np.rot90(np.rint(display(exr(camera['file_path'])*gainmap[camera['physical_camera']],exposure)*255).clip(0,255).astype(np.uint8)).copy()
                yy,xx=np.nonzero(region);box=(max(0,int(xx.min())-12),max(0,int(yy.min())-12),min(1080,int(xx.max())+13),min(1920,int(yy.max())+13))
            missing=region&~hit[0];covered=missing&hit[1]&(ids>=nt)
            which=ids[covered].astype(int)-nt;unique,counts=np.unique(which,return_counts=True)
            component,n=label(covered);sizes=np.bincount(component.ravel())[1:]
            detail=[]
            for p,npx in zip(unique,counts):
                j=lookup[p]
                detail.append(dict(proposal=int(p),pixels=int(npx),mask_support=int(a['mask_support'][p]),
                    mask_outside=int(a['mask_outside'][p]),semantic_admitted=bool(j>=0),
                    strict_admitted=bool(a['strict'][j]) if j>=0 else None,
                    certificate_admitted=bool(e['prior'][j]) if j>=0 else None,
                    retained=bool(p in retained)))
            overlay=rgb.copy();overlay[missing]=[255,30,30];overlay[covered]=[20,240,240]
            overlay[missing&hit[2]]=[30,255,50]
            oldclay=np.zeros_like(rgb);oldclay[hit[0]]=[130,130,130]
            rawclay=oldclay.copy();rawclay[hit[1]&~hit[0]]=[240,70,30]
            rawclay[region&hit[2]&~hit[0]]=[30,255,50]
            path=ROOT/(frame+'_'+view+'.png')
            panel(path,[rgb,overlay,rawclay],['source RGB / published moving','red misses / cyan raw / green final','gray old / orange raw additions'],box)
            rr=dict(frame=frame,view=view,physical_camera=camera['physical_camera'],
                region='moving_rectangle_contains_background' if view=='moving' else 'prior_manual_train_hair_polygon',
                baseline_missing=int(missing.sum()),raw_covers=int(covered.sum()),final_covers=int((missing&hit[2]).sum()),
                raw_coverage_components=sorted(map(int,sizes),reverse=True),proposals=detail,
                panel=str(path),panel_sha256=sha(path),box=box)
            frame_records.append(rr);queries.extend(unique.tolist())
        unique=np.unique(queries)
        if len(unique):
            points=barycentric_samples(v[t[nt+unique]])
            votes,_=train_reference_votes(points.reshape(-1,3),rows,depths)
            votes=votes.reshape(-1,10)
            free=footprint_veto(points.reshape(-1,3),rows,depths).reshape(62,-1,10)
            for rr in frame_records:
                for item in rr['proposals']:
                    i=int(np.searchsorted(unique,item['proposal']))
                    item.update(recomputed_sample_votes=votes[i].tolist(),
                        geometric_sample_gate=bool((votes[i,:3]>=2).sum()>=2 and np.median(votes[i])>=2),
                        sample_free_veto=bool(free[:,i,:].any()))
            np.savez_compressed(ROOT/(frame+'_queried_evidence.npz'),proposals=unique,points=points,votes=votes,free=free)
        records.extend(frame_records)
        print(frame,[(r['view'],r['baseline_missing'],r['raw_covers'],r['final_covers'],len(r['proposals'])) for r in frame_records],flush=True)
    atomic_json(ROOT/'result.json',dict(request_sha256=sha(ROOT/'request.json'),records=records,
        visual_status='pending',production_updated=False,anatomical_completeness_not_proven=True))
    atomic_json(ROOT/'complete.json',dict(hashes={str(p):sha(p) for p in ROOT.iterdir() if p.is_file()}))


if __name__=='__main__':
    run()
