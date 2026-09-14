"""Train-referenced confidence and native-pixel evidence at rejected jaw vertices.

Read-only study: no mesh edit, no held-out RGB, no virtual camera in confidence.
"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,exr,display,ROOT
from study_confidence_depth_prior import load_real,support,project_integer,unproject
from diagnose_jaw_measured_depth import observed_at,ATTRIBUTION
from guard_jaw_measured_depth import initial_admission

BASE=Path('/mnt/data/dec5_jaw_measured_depth')


def train_reference_votes(points,rows,depths,tolerance=.001):
    """Nearest agreeing measured depth chooses a deterministic physical anchor.

The anchor counts once; other cameras must meet the existing roundtrip and
parallax rule. Missing depth cannot establish an anchor. Ties use camera name.
"""
    order=sorted(range(len(rows)),key=lambda i:rows[i]['physical_camera'])
    errors=[]
    for i in order:
        _,z,d,valid=observed_at(points,rows[i],depths[i])
        errors.append(np.where(valid&(np.abs(d-z)<=tolerance),np.abs(d-z),np.inf))
    errors=np.array(errors);best=errors.argmin(0);available=np.isfinite(errors.min(0))
    refs=np.full(len(points),-1,np.int16);refs[available]=np.asarray(order)[best[available]]
    votes=np.zeros(len(points),np.uint8)
    for index in np.unique(refs[refs>=0]):
        ids=np.flatnonzero(refs==index)
        counts,_=support(points[ids],rows[index],rows,depths,tolerance=tolerance)
        votes[ids]=counts+1
    return votes,refs


def run(output,frame):
    out=output/frame;out.mkdir(parents=True,exist_ok=True)
    source=BASE/'analysis'/frame;record=read(source/'result.json')
    if sha(source/'evidence.npz')!=record['evidence_sha256']:raise ValueError('Changed sample evidence')
    rows,depths,receipt=load_real(BASE/'analysis',frame)
    if receipt!=read(source/'request.json')['real_depth_receipt']:raise ValueError('Changed observed maps')
    request=dict(frame=frame,script_sha256=sha(__file__),source_result_sha256=sha(source/'result.json'),
        real_depth_receipt=receipt,reference='nearest agreeing real train depth; camera-name tie break',
        virtual_camera_used=False,heldout_used=False,geometry_changed=False,
        footprint_radius=4,neighbor_support_other_views=3,free_space_tolerance=.003)
    if (out/'request.json').exists() and read(out/'request.json')!=request:raise ValueError('Immutable footprint request mismatch')
    atomic_json(out/'request.json',request)
    a=np.load(source/'evidence.npz');points=a['samples'].reshape(-1,3)
    votes,refs=train_reference_votes(points,rows,depths)
    masks=np.load(ATTRIBUTION/frame/'admission.npz')
    keep=initial_admission(votes.reshape(-1,10),a['trusted_free'],masks['support'],masks['outside'])
    np.savez_compressed(out/'train_reference.npz',votes=votes.reshape(-1,10),references=refs.reshape(-1,10),initial_admitted=keep)
    spot=[r['proposal'] for r in record['selected_spot']];samples=a['samples'];veto=a['trusted_free']
    locations=[];seen=set();profiles=read(ROOT/'camera_profiles.json');gains=dict(zip(profiles['physical_cameras'],profiles['rgb_gain']))
    gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    for ci,ti,si in zip(*np.nonzero(veto)):
        if ti not in spot:continue
        point=samples[ti,si];key=(int(ci),tuple(point))
        if key in seen:continue
        seen.add(key);camera=rows[ci];depth=depths[ci]
        uv,z=project_integer(camera,point[None]);u,v=uv[0];x,y=np.rint(uv[0]).astype(int)
        yy,xx=np.mgrid[y-4:y+5,x-4:x+5];d=depth[yy,xx];valid=np.isfinite(d)&(d>0)
        neighbors=np.zeros(d.shape,np.uint8)
        counts,_=support(unproject(camera,xx[valid],yy[valid],d[valid]),camera,rows,depths)
        neighbors[valid]=counts
        delta=np.where(valid,d-z[0],np.nan)
        rgb=np.rint(display(exr(camera['file_path'])*np.array(gains[camera['physical_camera']]),gain)*255).clip(0,255).astype(np.uint8)
        panel=Image.new('RGB',(730,405));draw=ImageDraw.Draw(panel)
        crop=Image.fromarray(rgb).crop((x-80,y-80,x+80,y+80));panel.paste(crop,(0,35))
        draw.ellipse((u-x+77,v-y+112,u-x+83,v-y+118),outline='red',width=1)
        zoom=Image.fromarray(rgb).crop((x-8,y-8,x+9,y+9)).resize((170,170),Image.Resampling.NEAREST);panel.paste(zoom,(0,220))
        draw.text((3,5),camera['physical_camera'],fill='white')
        draw.text((185,5),'delta z x1000 / other-view votes; red=far',fill='white')
        for iy in range(9):
            for ix in range(9):
                color=(180,35,35) if valid[iy,ix] and delta[iy,ix]>.003 else ((40,110,65) if valid[iy,ix] else (50,50,50))
                box=(185+ix*60,35+iy*40,244+ix*60,74+iy*40);draw.rectangle(box,fill=color)
                text=f'{delta[iy,ix]*1000:.1f}/{neighbors[iy,ix]}' if valid[iy,ix] else 'missing'
                draw.text((box[0]+2,box[1]+12),text,fill='white')
        draw.rectangle((185+4*60,35+4*40,244+4*60,74+4*40),outline='yellow',width=2)
        path=out/f'veto_{len(locations):02d}.png';panel.save(path)
        locations.append(dict(camera=camera['physical_camera'],proposal=int(ti),sample=int(si),point=point.tolist(),
            uv=uv[0].tolist(),candidate_depth=float(z[0]),integer_xy=[int(x),int(y)],depth=np.where(valid,d,0).tolist(),
            delta_milli=np.where(valid,delta*1000,0).tolist(),valid=valid.tolist(),other_view_support=neighbors.tolist(),
            source_rgb_sha256=sha(camera['file_path']),panel=path.name,panel_sha256=sha(path)))
    summary=[dict(proposal=k,old_votes=a['votes'][k].tolist(),train_votes=votes.reshape(-1,10)[k].tolist(),
                  admitted=bool(keep[k]),trusted_free_samples=int(veto[:,k].sum())) for k in spot]
    atomic_json(out/'result.json',dict(request_sha256=sha(out/'request.json'),reference_npz_sha256=sha(out/'train_reference.npz'),
        admitted_triangles=int(keep.sum()),spot=summary,free_space_locations=locations,visual_status='pending',production_accepted=False))
    print(frame,'spot',summary,'unique_veto_points',len(locations),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_jaw_depth_footprint'))
    p.add_argument('--frame',required=True,choices=['001193','001195']);a=p.parse_args();run(a.output,a.frame)
