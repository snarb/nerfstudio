"""Native RGB review of actual sampled-depth vetoes in the fixed forearm ROI."""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras
from study_body_neighborhood_completion import ROOT as BODY,load_inputs
from study_body_single_depth_seed import ROOT as WEAK
from study_confidence_depth_prior import project_integer,unproject
from study_jaw_train_confidence import enclosing_samples
from diagnose_jaw_measured_depth import barycentric_samples
from calibrated_depth_witness import load_images,event_votes
from forearm_rgb_witnesses import color_errors,patch_rgb
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
import study_forearm_plane_transfer_v3 as prior

ROOT=Path('/mnt/data/dec5_body_rgb_veto_diagnosis')


def crop(image,xy,size=160):
    x,y=np.rint(xy).astype(int);im=Image.fromarray(image).crop((x-size//2,y-size//2,x+size//2,y+size//2))
    draw=ImageDraw.Draw(im);c=size//2;draw.rectangle((c-3,c-3,c+3,c+3),outline=(255,40,40),width=1)
    return im


def run(frame):
    out=ROOT/frame;out.mkdir(parents=True,exist_ok=False);src=BODY/frame
    request=read(src/'request.json');result=read(src/'result.json')
    for n,h in result['hashes'].items():assert sha(src/n)==h
    rows,depths,hashes,_,_=load_inputs(frame);assert hashes==request['source_depth_sha256']
    images,legacy,metadata=load_images(frame,include_legacy=True)
    image_differences={n:int(np.any(im!=legacy[n],axis=2).sum()) for n,im in images.items()}
    images_identical=not any(image_differences.values())
    e=np.load(src/'admission.npz');p=np.load(src/'proposal.npz');mv=p['vertices'];semantic=e['semantic_ids'];pp=p['proposals'][semantic]
    old=o3d.io.read_triangle_mesh(request['source_mesh']);v=np.asarray(old.vertices);t=np.asarray(old.triangles);nt=len(t)
    camera=next(r for r in rows if r['physical_camera']=='H004_A005_1210M6');prior.configure();mask=np.rot90(prior.v2.v1.masks(frame)[camera['physical_camera']])
    od,_,_=camera_depth(scene_for(v,t),camera);d,ids,_=camera_depth(scene_for(mv,np.concatenate([t,pp])),camera)
    od,d,ids=map(np.rot90,[od,d,ids]);hit=ids[mask&~np.isfinite(od)&np.isfinite(d)&(ids>=nt)]-nt
    selected,weights=np.unique(hit,return_counts=True);lookup_weight=dict(zip(selected.tolist(),weights.tolist()))
    points=barycentric_samples(mv[pp]);ci,local,si=np.nonzero(e['free'][:,selected,:]);tri=selected[local]
    events=points[tri,si];qualified=np.zeros(len(events),bool);old_qualified=np.zeros(len(events),bool)
    all_votes={k:np.zeros((len(events),4),int) for k in ['geometric','rgb_qualified','comparable','decisive']}
    legacy_votes={k:np.zeros((len(events),4),int) for k in all_votes}
    for index in np.unique(ci):
        which=np.flatnonzero(ci==index);votes,keep=event_votes(events[which],rows[index],depths[index],rows,depths,images)
        qualified[which]=keep
        ov,ok=(votes,keep) if images_identical else event_votes(events[which],rows[index],depths[index],rows,depths,legacy)
        old_qualified[which]=ok
        for k in all_votes:all_votes[k][which]=votes[k].reshape(-1,4);legacy_votes[k][which]=ov[k].reshape(-1,4)
    new_free=np.zeros(len(pp),bool);new_free[tri[qualified]]=True
    old_free=e['free'].any(axis=(0,2));summary=[]
    for label,cert in [('three_depth',e['certificate']),('one_depth',np.load(WEAK/frame/'certificate_evidence.npz')['certificate'])]:
        lookup=np.zeros(len(mv),bool);lookup[e['query_ids']]=cert;shape=lookup[pp].all(1)
        summary.append(dict(variant=label,semantic_covered_old_misses=len(hit),shape_pass=int(shape[hit].sum()),
            old_shape_and_free_pass=int((shape[hit]&~old_free[hit]).sum()),
            new_shape_and_free_pass=int((shape[hit]&~new_free[hit]).sum()),
            old_sample_free_pass=int((~old_free[hit]).sum()),new_sample_free_pass=int((~new_free[hit]).sum())))
    arrays=dict(camera_ids=ci,semantic_indices=tri,sample_ids=si,points=events,qualified=qualified,legacy_qualified=old_qualified)
    arrays.update({'centered_'+k:x for k,x in all_votes.items()});arrays.update({'legacy_'+k:x for k,x in legacy_votes.items()})
    np.savez_compressed(out/'events.npz',**arrays)
    rgb_ok=(all_votes['rgb_qualified']>=3).all(1);cases=[];chosen=[]
    for kind,selection in [('retained',qualified),('rgb_mismatch',~rgb_ok),('ambiguous',rgb_ok&~qualified)]:
        order=sorted(np.flatnonzero(selection),key=lambda j:(-lookup_weight[int(tri[j])],int(ci[j]),int(tri[j]),int(si[j])))
        used_cameras=set()
        for j in order:
            if int(ci[j]) in used_cameras:continue
            used_cameras.add(int(ci[j]));chosen.append((kind,int(j)))
            if len(used_cameras)==2:break
    for number,(kind,j) in enumerate(chosen):
        ref=rows[int(ci[j])];depth=depths[int(ci[j])];uv,z=project_integer(ref,events[j:j+1]);taps,obs,far=enclosing_samples(uv,z,depth)
        xy=taps[0,0];observed=unproject(ref,xy[0:1],xy[1:2],obs[0,0:1])
        proposed=unproject(ref,xy[0:1],xy[1:2],z[0:1])
        chroma,rgb=color_errors(observed,ref,rows,depths,images)
        witnesses=np.flatnonzero(np.isfinite(chroma[:,0]));witnesses=sorted(witnesses,key=lambda k:(float(rgb[k,0]),rows[k]['physical_camera']))[:4]
        panel=Image.new('RGB',(740,190*(1+len(witnesses))));draw=ImageDraw.Draw(panel)
        panel.paste(crop(images[ref['physical_camera']],xy),(0,25));draw.text((3,4),'query '+ref['physical_camera'],fill='white')
        draw.text((180,40),kind+'; event '+str(j),fill='white');draw.text((180,60),'obs z %.6f / prior z %.6f'%(obs[0,0],z[0]),fill='white')
        draw.text((180,80),'4-tap RGB witnesses '+str(all_votes['rgb_qualified'][j].tolist()),fill='white')
        draw.text((180,100),'4-tap decisive '+str(all_votes['decisive'][j].tolist()),fill='white')
        reference_rgb=patch_rgb(images[ref['physical_camera']],xy[None])[0];details=[]
        for k,index in enumerate(witnesses):
            row=rows[index];a,_=project_integer(row,observed);b,_=project_integer(row,proposed)
            y=190*(k+1);panel.paste(crop(images[row['physical_camera']],a[0]),(0,y+25));panel.paste(crop(images[row['physical_camera']],b[0]),(180,y+25))
            draw.text((3,y+4),row['physical_camera']+' observed',fill='white');draw.text((183,y+4),'nearer prior',fill='white')
            alt=float(np.abs(patch_rgb(images[row['physical_camera']],np.rint(b).astype(int))[0]-reference_rgb).mean())
            draw.text((360,y+45),'RGB observed %.4f / prior %.4f'%(rgb[index,0],alt),fill='white')
            draw.text((360,y+65),'chroma observed %.4f'%chroma[index,0],fill='white')
            details.append(dict(camera=row['physical_camera'],observed_uv=a[0].tolist(),proposed_uv=b[0].tolist(),
                chroma=float(chroma[index,0]),observed_rgb_error=float(rgb[index,0]),alternative_rgb_error=alt))
        path=out/('case_%02d_%s.png'%(number,kind));panel.save(path)
        cases.append(dict(category=kind,event=j,camera=ref['physical_camera'],semantic_index=int(tri[j]),sample=int(si[j]),
            weighted_old_missing_rays=lookup_weight[int(tri[j])],panel=str(path),panel_sha256=sha(path),first_tap_witnesses=details))
    atomic_json(out/'result.json',dict(script_sha256=sha(__file__),source_admission_sha256=sha(src/'admission.npz'),
        weak_certificate_sha256=sha(WEAK/frame/'certificate_evidence.npz'),source_depth_sha256=hashes,**metadata,
        profiles_control_image_pixel_differences=image_differences,profile_control_decision_differences=int((qualified!=old_qualified).sum()),
        old_events=len(events),new_events=int(qualified.sum()),semantic_triangles=len(selected),
        previous_veto_triangles=int(old_free[selected].sum()),new_veto_triangles=int(new_free[selected].sum()),
        fixed_forearm_nearest_ray_summary=summary,cases=cases,events_sha256=sha(out/'events.npz'),
        geometry_changed=False,no_quality_metrics=True,not_an_admission_or_video_approval=True,visual_status='pending'))
    print('events',len(events),'qualified',int(qualified.sum()),'profile decisions changed',int((qualified!=old_qualified).sum()),summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=['001029','001033','001037'],required=True)
    run(p.parse_args().frame)
