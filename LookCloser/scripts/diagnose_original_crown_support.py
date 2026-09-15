"""Read-only train evidence for existing hair/crown triangles, not deletion.

Query triangles visible in a previously drawn train hair polygon. The top
quarter of that polygon's portrait bounding box is a diagnostic crown band.
Missing depth alone is not proof that an existing triangle is erroneous.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras,project
from transfer_close_boundary_completion import SOURCE,MOVIE,FRAMES
from refine_measured_head_masks import ROOT as MASKS
from study_confidence_depth_prior import load_real,REGIONS,region_masks
from study_jaw_depth_footprint import train_reference_votes
from study_jaw_repair_transfer import mask_votes
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from review_jaw_repair_transfer import panel
from calibrated_depth_witness import load_images

ROOT=Path('/mnt/data/dec5_original_crown_support')


def classify(votes,outside):
    """Descriptive labels only; no deletion policy is implied."""
    low=np.median(votes,axis=1)<2
    multi=np.asarray(outside)>=3
    return low,multi,low&multi


def verify_original_surface(vertices,triangles,prior_vertices,prior_triangles):
    # The earlier boundary repair split non-manifold vertices at zero distance.
    # Vertex IDs can change while triangle order and actual surface stay exact.
    np.testing.assert_array_equal(vertices[triangles[:len(prior_triangles)]],prior_vertices[prior_triangles])


def display_pixels(value):
    # load_images already applies the frozen display response and quantizes.
    value=np.asarray(value)
    if value.dtype!=np.uint8 or value.ndim!=3 or value.shape[-1]!=3:
        raise ValueError('Expected calibrated uint8 RGB, not linear/normalized pixels')
    return value.copy()


def run(frame):
    root=ROOT/frame;root.mkdir(parents=True,exist_ok=False)
    base=read(SOURCE/frame/'request.json');rows,depths,receipt=load_real(Path(base['depth_root']),frame)
    assert receipt==base['depth_receipt'] and sha(base['source_mesh'])==base['source_mesh_sha256']
    entry=next(r for r in read(MOVIE/'request.json')['inventory'] if r['frame_id']==frame)
    maskspec=entry['source_masks'];oldroot=Path(maskspec['root'])
    assert sha(oldroot/'masks.npz')==maskspec['masks_sha256']
    names=read(oldroot/'cameras.json');oldmasks=np.load(oldroot/'masks.npz')['masks']
    mq=read(MASKS/frame/'request.json');mr=read(MASKS/frame/'result.json')
    assert sha(MASKS/frame/'request.json')==mr['request_sha256']
    assert mq['original_masks_sha256']==sha(oldroot/'masks.npz')
    for p,h in mr['hashes'].items():assert sha(MASKS/frame/p)==h
    refined=np.load(MASKS/frame/'masks.npz')['masks']
    assert names==read(MASKS/frame/'cameras.json')
    images,_,rgb_receipt=load_images(frame)
    mesh=o3d.io.read_triangle_mesh(base['source_mesh']);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
    native=next(r for r in rows if r['physical_camera']==REGIONS[frame]['camera'])
    scene=scene_for(v,t);depth,hit,_=camera_depth(scene,native)
    region=region_masks(frame)['hair'];portrait=np.rot90(region);yy,xx=np.nonzero(portrait)
    crown=portrait.copy();crown[np.indices(crown.shape)[0]>yy.min()+.25*(yy.max()-yy.min())]=False
    crown=np.rot90(crown,-1);visible=region&np.isfinite(depth)
    query=np.unique(hit[visible]).astype(int);crown_ids=np.unique(hit[crown&np.isfinite(depth)]).astype(int)
    points=np.concatenate([v[t[query]],v[t[query]].mean(1)[:,None]],axis=1)
    scripts=[Path(__file__).name,'study_jaw_depth_footprint.py','study_jaw_repair_transfer.py','calibrated_depth_witness.py']
    request=dict(frame=frame,source_mesh=base['source_mesh'],source_mesh_sha256=sha(base['source_mesh']),
        depth_receipt=receipt,rgb_receipt=rgb_receipt,original_masks_sha256=sha(oldroot/'masks.npz'),
        refined_masks_sha256=sha(MASKS/frame/'masks.npz'),native_camera=native,
        regions=REGIONS[frame],crown_top_fraction=.25,query_samples='vertices and centroid',
        depth_tolerance=.001,low_support='median of four samples <2',mask_disagreement='at least3 cameras',
        geometry_changed=False,heldout_used=False,diagnostic_not_deletion_policy=True,
        scripts={str(Path(__file__).resolve().with_name(n)):sha(Path(__file__).with_name(n)) for n in scripts})
    atomic_json(root/'request.json',request)
    atomic_json(root/'progress.json',dict(stage='depth_votes',query_triangles=len(query),unix_time=time.time()))
    votes,refs=train_reference_votes(points.reshape(-1,3),rows,depths);votes=votes.reshape(-1,4)
    ms,mo=mask_votes(v,t[query],rows,oldmasks,names)
    rs,ro=mask_votes(v,t[query],rows,refined,names)
    low,multi,both=classify(votes,ro)
    np.savez_compressed(root/'evidence.npz',query_triangles=query,crown_triangles=crown_ids,points=points,
        depth_votes=votes,depth_references=refs.reshape(-1,4),original_mask_support=ms,original_mask_outside=mo,
        refined_mask_support=rs,refined_mask_outside=ro,low_support=low,multi_mask_outside=multi,both=both,
        native_depth=depth,native_triangle_ids=hit,hair_region=region,crown_region=crown)
    lookup=np.full(len(t),-1,int);lookup[query]=np.arange(len(query));pixel=lookup[hit[visible]]
    gt=display_pixels(images[native['physical_camera']])
    overlay=gt.copy();colors=np.full((len(query),3),[40,210,90],np.uint8)
    colors[low]=[255,70,50];colors[multi]=[190,65,255];colors[both]=[255,210,25]
    overlay[visible]=colors[pixel]
    box=(max(0,int(xx.min())-15),max(0,int(yy.min())-15),min(1080,int(xx.max())+16),min(1920,int(yy.max())+16))
    panel(root/'native_hair.png',[np.rot90(gt).copy(),np.rot90(overlay).copy()],
        ['real train RGB','red low depth / violet mask disagreement / yellow both'],box)
    crownbox=(box[0],box[1],box[2],int(yy.min()+.25*(yy.max()-yy.min()))+35)
    panel(root/'native_crown.png',[np.rot90(gt).copy(),np.rot90(overlay).copy()],['real train RGB','diagnostic labels, not a deletion mask'],crownbox)
    records=[]
    for name,ids in [('hair',query),('crown',crown_ids)]:
        selected=np.isin(query,ids);pix=(crown if name=='crown' else region)&np.isfinite(depth)
        pindex=lookup[hit[pix]]
        records.append(dict(region=name,triangles=int(selected.sum()),visible_pixels=int(pix.sum()),
            low_depth_triangles=int((low&selected).sum()),mask_disagreement_triangles=int((multi&selected).sum()),
            both_triangles=int((both&selected).sum()),both_visible_pixels=int(both[pindex].sum()),
            low_depth_visible_pixels=int(low[pindex].sum()),mask_disagreement_visible_pixels=int(multi[pindex].sum()),
            median_depth_votes_quantiles=np.quantile(np.median(votes[selected],axis=1),[0,.25,.5,.75,1]).tolist(),
            refined_outside_quantiles=np.quantile(ro[selected],[0,.25,.5,.75,1]).tolist()))
    # Determine whether the latest pre-existing boundary repair created the fringe.
    repair=read(entry['head_repair_receipt']);prior=o3d.io.read_triangle_mesh(repair['request']['source_mesh'])
    assert sha(repair['request']['source_mesh'])==repair['request']['source_sha256']
    verify_original_surface(v,t,np.asarray(prior.vertices),np.asarray(prior.triangles))
    provenance=dict(latest_head_repair_added=repair['added_triangles'],
        queried_triangles_from_latest_repair=int((query>=len(prior.triangles)).sum()),
        crown_triangles_from_latest_repair=int((crown_ids>=len(prior.triangles)).sum()))
    atomic_json(root/'result.json',dict(request_sha256=sha(root/'request.json'),records=records,provenance=provenance,
        hashes={p.name:sha(p) for p in root.iterdir() if p.is_file() and p.name not in ['result.json','progress.json']},
        geometry_changed=False,visual_status='pending'))
    print(frame,records,provenance,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True,choices=FRAMES);run(p.parse_args().frame)
