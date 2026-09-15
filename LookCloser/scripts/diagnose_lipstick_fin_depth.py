"""Read-only attribution of the retained 000995 lipstick fin to train depths.

A manually observed train-view polygon selects diagnostic faces only. Missing
depth is unknown. No geometry, mask, radiometry or production artifact is edited.
"""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json, project
from review_full_block_transfer import ROOT as DEPTH_ROOT, VIDEO
from study_confidence_depth_prior import load_real, project_integer, support, unproject
from study_jaw_depth_footprint import train_reference_votes
from diagnose_jaw_measured_depth import observed_at
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from calibrated_depth_witness import load_images
from review_jaw_repair_transfer import panel, verified_image

ROOT = Path('/mnt/data/dec5_lipstick_fin_depth/000995')
FRAME = '000995'
CAMERA = 'K004_B005_1210DS'
POLYGON = [(150,1184),(174,1190),(177,1209),(219,1219),(225,1234),(191,1232),(162,1210)]
BOX = (50,1140,300,1400)


def run():
    ROOT.mkdir(parents=True, exist_ok=False)
    parent = read(VIDEO/'request.json')
    entry = next(r for r in parent['inventory'] if r['frame_id']==FRAME)
    assert sha(entry['mesh'])==entry['mesh_sha256']
    rows, depths, receipt = load_real(DEPTH_ROOT, FRAME)
    assert len(rows)==62 and len({r['physical_camera'] for r in rows})==62
    received=read(DEPTH_ROOT/FRAME/'received.json')
    for name, digest in received['depth_hashes'].items(): assert sha(DEPTH_ROOT/FRAME/name)==digest
    mesh=o3d.io.read_triangle_mesh(entry['mesh']); v=np.asarray(mesh.vertices); t=np.asarray(mesh.triangles)
    camera=next(r for r in rows if r['physical_camera']==CAMERA)
    scene=scene_for(v,t); depth, ids, _=camera_depth(scene,camera)
    portrait_ids=np.rot90(ids); portrait_depth=np.rot90(depth)
    mask=Image.new('L',(1080,1920));ImageDraw.Draw(mask).polygon(POLYGON,fill=1)
    selected=np.unique(portrait_ids[np.asarray(mask,dtype=bool)&np.isfinite(portrait_depth)]).astype(int)
    assert len(selected)>0
    points=np.concatenate([v[t[selected]],v[t[selected]].mean(1)[:,None]],axis=1)
    flat=points.reshape(-1,3)
    votes, refs=train_reference_votes(flat,rows,depths)
    votes=votes.reshape(-1,4);refs=refs.reshape(-1,4)
    shape=(62,len(selected),4)
    near=np.zeros(shape,bool);far=np.zeros(shape,bool);occluded=np.zeros(shape,bool)
    available=np.zeros(shape,bool);trusted_free=np.zeros(shape,bool)
    deltas=np.zeros(shape,np.float32)
    maskspec=entry['source_masks'];maskroot=Path(maskspec['root'])
    assert sha(maskroot/'masks.npz')==maskspec['masks_sha256']
    masks=np.load(maskroot/'masks.npz')['masks'];names=read(maskroot/'cameras.json')
    outside=np.zeros(shape,bool)
    for ci,(row,d) in enumerate(zip(rows,depths)):
        xy,z,obs,ok=observed_at(flat,row,d)
        available[ci]=ok.reshape(-1,4);delta=obs-z
        deltas[ci]=np.where(ok,delta,0).reshape(-1,4)
        near[ci]=(ok&(np.abs(delta)<=.001)).reshape(-1,4)
        far[ci]=(ok&(delta>.003)).reshape(-1,4)
        occluded[ci]=(ok&(delta<-.003)).reshape(-1,4)
        chosen=np.flatnonzero(ok&(delta>.003))
        if len(chosen):
            corroboration,_=support(unproject(row,xy[chosen,0],xy[chosen,1],obs[chosen]),row,rows,depths)
            trusted_free[ci].reshape(-1)[chosen]=corroboration>=3
        uv,z=project(flat,[row]);xy=np.rint(uv[0]).astype(int)
        valid=(z[0]>0)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
        j=np.flatnonzero(valid)
        outside[ci].reshape(-1)[j]=~masks[names.index(row['physical_camera'])][xy[j,1],xy[j,0]].astype(bool)
    images,_,rgb_receipt=load_images(FRAME)
    native_root=DEPTH_ROOT/FRAME/'rgb'/CAMERA/'production'
    prediction,_=verified_image(native_root,FRAME)
    gt=np.rot90(images[CAMERA]);overlay=prediction.copy()
    overlay[np.isin(portrait_ids,selected)]=[250,40,50]
    panel(ROOT/'selected_native.png',[gt,prediction,overlay],['real train GT','production','diagnostic faces'],BOX)
    np.savez_compressed(ROOT/'evidence.npz',triangle_ids=selected,points=points,
        votes=votes,references=refs,near=near,far=far,occluded=occluded,available=available,
        trusted_free=trusted_free,deltas=deltas,source_mask_outside=outside,
        native_depth=depth,native_triangle_ids=ids)
    details=[]
    for i,ti in enumerate(selected):
        details.append(dict(triangle=int(ti),depth_votes=votes[i].tolist(),
            near_views=near[:,i].sum(0).tolist(),far_views=far[:,i].sum(0).tolist(),
            trusted_free_views=trusted_free[:,i].sum(0).tolist(),
            unavailable_views=(~available[:,i]).sum(0).tolist(),
            source_mask_outside=outside[:,i].sum(0).tolist(),centroid=points[i,3].tolist()))
    # Largest selected projected face is a deterministic representative, not
    # whichever point produces the strongest support or contradiction.
    pixel_counts=np.array([(portrait_ids==ti).sum() for ti in selected])
    ri=int(pixel_counts.argmax());point=points[ri,3]
    crops=[];observations=[]
    for ci,row in enumerate(rows):
        uv,z=project(point[None],[row]);u,w=uv[0,0];x,y=int(round(u)),int(round(w))
        crop=Image.fromarray(images[row['physical_camera']]).crop((x-60,y-60,x+60,y+60))
        draw=ImageDraw.Draw(crop);draw.ellipse((57,57,63,63),outline='red',width=1)
        tile=Image.new('RGB',(240,160));tile.paste(crop,(60,40));draw=ImageDraw.Draw(tile)
        draw.text((2,2),row['physical_camera'],fill='white')
        state=('missing' if not available[ci,ri,3] else
            f'dz={deltas[ci,ri,3]:+.5f} near={int(near[ci,ri,3])} free={int(trusted_free[ci,ri,3])}')
        draw.text((2,19),state,fill='white');crops.append(tile)
        observations.append(dict(camera=row['physical_camera'],uv=[float(u),float(w)],
            depth=float(z[0,0]),available=bool(available[ci,ri,3]),
            delta=float(deltas[ci,ri,3]),near=bool(near[ci,ri,3]),
            trusted_free=bool(trusted_free[ci,ri,3]),mask_outside=bool(outside[ci,ri,3])))
    for start in range(0,62,16):
        sheet=Image.new('RGB',(960,640))
        for i,tile in enumerate(crops[start:start+16]):sheet.paste(tile,((i%4)*240,(i//4)*160))
        sheet.save(ROOT/f'train_witnesses_{start:02d}.png')
    atomic_json(ROOT/'request.json',dict(frame=FRAME,source_mesh=entry['mesh'],
        source_mesh_sha256=entry['mesh_sha256'],source_metadata_sha256=entry['metadata_sha256'],
        source_video_sha256=sha(VIDEO/'request.json'),depth_receipt=receipt,rgb_receipt=rgb_receipt,
        source_masks=maskspec,diagnostic_camera=CAMERA,diagnostic_polygon=POLYGON,
        manual_polygon_is_not_independent_evidence=True,geometry_changed=False,heldout_used=False,
        thresholds=dict(near=.001,far=.003,trusted_far_other_views=3,roundtrip=1.5,min_parallax_deg=1),
        scripts={str(Path(__file__).resolve().with_name(n)):sha(Path(__file__).with_name(n)) for n in
            [Path(__file__).name,'study_confidence_depth_prior.py','study_jaw_depth_footprint.py',
             'diagnose_jaw_measured_depth.py','joint_temporal_texture.py','calibrated_depth_witness.py']}))
    atomic_json(ROOT/'result.json',dict(request_sha256=sha(ROOT/'request.json'),triangles=details,
        representative_triangle=int(selected[ri]),representative_observations=observations,
        hashes={p.name:sha(p) for p in ROOT.iterdir() if p.is_file() and p.name!='result.json'},
        geometry_changed=False,visual_status='pending'))
    print('Diagnostic triangles',len(selected),'representative',int(selected[ri]),flush=True)
    print(details,flush=True)


if __name__=='__main__':run()
