"""Locate the stage that prevents a raw surface prior from filling crown gaps.

Read-only on production/candidates. Manual train hair polygons select diagnostic
pixels only; these candidate-dependent gap counts are NOT image-quality metrics.
"""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras
from transfer_close_boundary_completion import ROOT as RAW,FRAMES
from probe_inset_head_completion import ROOT as INSET,inset_vertices
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from study_confidence_depth_prior import REGIONS,region_masks
from review_jaw_repair_transfer import panel

ROOT=Path('/mnt/data/dec5_crown_completion_gap_attribution')
LABELS=['not_local_proposal','rejected_by_silhouettes','rejected_by_measured_guard','retained']
COLORS=np.array([[255,80,50],[210,60,255],[255,210,30],[40,210,100]],np.uint8)


def stages(count,local,semantic,guarded):
    local=np.asarray(local,dtype=int);semantic=np.asarray(semantic,dtype=int);guarded=np.asarray(guarded,dtype=int)
    if not np.isin(semantic,local).all() or not np.isin(guarded,semantic).all():raise ValueError('Non-nested evidence')
    if any(len(x) and (x.min()<0 or x.max()>=count) for x in [local,semantic,guarded]):raise ValueError('Invalid triangle IDs')
    label=np.zeros(count,np.uint8);label[local]=1;label[semantic]=2;label[guarded]=3
    return label


def run(frame):
    out=ROOT/frame;out.mkdir(parents=True,exist_ok=False)
    config=read(INSET/frame/'request.json');arm=INSET/frame/'inset_001000';guard=INSET/frame/'guarded'
    assert sha(RAW/frame/'poisson_raw.ply')==config['raw_mesh_sha256']
    assert sha(config['source_mesh'])==config['source_mesh_sha256']
    ar=read(arm/'result.json');gr=read(guard/'result.json')
    for root,r in [(arm,ar),(guard,gr)]:
        for p,h in r['hashes'].items():assert sha(root/p)==h
    raw=o3d.io.read_triangle_mesh(str(RAW/frame/'poisson_raw.ply'));rv=np.asarray(raw.vertices);rt=np.asarray(raw.triangles)
    shifted=inset_vertices(rv,np.array(config['center']),.001)
    original=o3d.io.read_triangle_mesh(config['source_mesh']);old=scene_for(np.asarray(original.vertices),np.asarray(original.triangles))
    prior=scene_for(shifted,rt)
    final=o3d.io.read_triangle_mesh(str(guard/'mesh.ply'));finalscene=scene_for(np.asarray(final.vertices),np.asarray(final.triangles))
    e=np.load(arm/'evidence.npz');g=np.load(guard/'evidence.npz')
    semantic=e['retained_raw_triangle_ids'];kept=semantic[g['retained_candidate_triangle_ids']]
    label=stages(len(rt),e['proposal_ids'],semantic,kept)
    rows,_,metadata=cameras(frame);native=next(r for r in rows if r['physical_camera']==REGIONS[frame]['camera'])
    od,_,_=camera_depth(old,native);pd,hit,_=camera_depth(prior,native);fd,_,_=camera_depth(finalscene,native)
    hair=region_masks(frame)['hair'];portrait=np.rot90(hair);yy,xx=np.nonzero(portrait)
    crown=portrait.copy();crown[np.indices(crown.shape)[0]>yy.min()+.25*(yy.max()-yy.min())]=False;crown=np.rot90(crown,-1)
    query=hair&~np.isfinite(od)&np.isfinite(pd);remaining=query&~np.isfinite(fd)
    pixlabels=np.full(pd.shape,255,np.uint8);pixlabels[query]=label[hit[query]]
    gtpath=INSET/'review'/frame/'native_unmasked_gt.png'
    seal=read(INSET/'artifact_manifest.json');assert sha(gtpath)==seal['hashes'][str(gtpath)]
    gt=np.array(Image.open(gtpath));overlay=np.rot90(gt,-1).copy();overlay[query]=COLORS[pixlabels[query]]
    box=(max(0,int(xx.min())-15),max(0,int(yy.min())-15),min(1080,int(xx.max())+16),int(yy.min()+.25*(yy.max()-yy.min()))+35)
    panel(out/'crown_stages.png',[gt,np.rot90(overlay).copy()],['train GT','red local / purple masks / yellow depth guard / green retained'],box)
    records=[]
    for name,region in [('hair',hair),('crown',crown)]:
        records.append(dict(region=name,old_depth_miss=int((region&~np.isfinite(od)).sum()),
            raw_prior_covers_old_miss=int((region&query).sum()),still_missing_after_guard=int((region&remaining).sum()),
            stages={n:int((region&query&(pixlabels==i)).sum()) for i,n in enumerate(LABELS)},
            remaining_stages={n:int((region&remaining&(pixlabels==i)).sum()) for i,n in enumerate(LABELS)},
            raw_prior_also_missing=int((region&~np.isfinite(od)&~np.isfinite(pd)).sum())))
    selected=np.unique(hit[crown&remaining]).astype(int)
    # Save actual 3D locations for the next evidence/fitting step, not just a
    # screen-space image mask that could hide errors in subsequent metrics.
    np.savez_compressed(out/'evidence.npz',raw_triangle_ids=selected,triangle_stage=label[selected],
        triangle_points=shifted[rt[selected]],native_original_depth=od,native_full_prior_depth=pd,
        native_guarded_depth=fd,native_full_prior_triangle_ids=hit,hair_region=hair,crown_region=crown,
        diagnostic_query=query,remaining_query=remaining,pixel_stage=pixlabels)
    q=dict(frame=frame,source_mesh_sha256=config['source_mesh_sha256'],raw_mesh_sha256=config['raw_mesh_sha256'],
        inset_config_sha256=sha(INSET/frame/'request.json'),inset_arm_result_sha256=sha(arm/'result.json'),
        inset_guard_result_sha256=sha(guard/'result.json'),native_camera=native,regions=REGIONS[frame],
        gt_path=str(gtpath),gt_sha256=sha(gtpath),metadata_sha256=sha(metadata),
        scripts={str(Path(__file__).resolve().with_name(n)):sha(Path(__file__).with_name(n)) for n in
            [Path(__file__).name,'probe_inset_head_completion.py','bake_joint_temporal_mesh.py','study_confidence_depth_prior.py']},
        geometry_changed=False,heldout_used=False,quality_metrics=False,diagnostic_selection_not_geometry_policy=True)
    atomic_json(out/'request.json',q)
    atomic_json(out/'result.json',dict(request_sha256=sha(out/'request.json'),records=records,
        selected_remaining_raw_triangles=len(selected),hashes={n:sha(out/n) for n in ['evidence.npz','crown_stages.png']},
        production_promoted=False,visual_status='pending'))
    print(frame,records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True,choices=FRAMES);a=p.parse_args();run(a.frame)
