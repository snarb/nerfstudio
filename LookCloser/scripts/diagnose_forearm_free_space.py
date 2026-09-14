"""Inspect observed-depth veto evidence without weakening or modifying any guard."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,exr,display,cameras,ROOT
from study_confidence_depth_prior import support,unproject,project_integer
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
import study_forearm_plane_transfer_v3 as prior


def run(root,frame):
    out=root/'free_space_diagnosis'/frame;out.mkdir(parents=True,exist_ok=True)
    prior.configure();v1=prior.v2.v1;rows,depths,hashes=v1.load_real(frame)
    result=read(root/frame/'geometry_result.json')
    if hashes!=result['depth_hashes']:raise ValueError('Changed depth evidence')
    mesh=o3d.io.read_triangle_mesh(str(root/frame/'transferred.ply'))
    if sha(root/frame/'transferred.ply')!=result['hashes']['transferred.ply']:raise ValueError('Changed candidate')
    scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
    parent=read('/mnt/data/dec5_phase30_dynamic_150/request.json')
    source=next(r for r in parent['inventory'] if r['frame_id']==frame)
    old=o3d.io.read_triangle_mesh(source['mesh']);masks=v1.masks(frame)
    offset=0
    counts={c['camera']:c['trusted_free_pixels'] for c in result['rounds'][0]['checks'] if c['offset']==offset}
    if not any(counts.values()):
        offset=.5;counts={c['camera']:c['trusted_free_pixels'] for c in result['rounds'][0]['checks'] if c['offset']==offset}
    selected=[n for n in sorted(counts,key=counts.get,reverse=True) if counts[n]>0][:3]
    actual,_,_=cameras(frame);lookup={r['physical_camera']:r for r in actual}
    profiles=np.load(ROOT/'parameters.npz')['log_gain'];gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    def rgb(name):
        idx=next(i for i,r in enumerate(actual) if r['physical_camera']==name)
        image=display(exr(lookup[name]['file_path'])*np.exp(profiles[idx]),gain)
        return np.rint(image*255).clip(0,255).astype(np.uint8)
    reference=next(r for r in rows if r['physical_camera']==v1.NAMES[0]);ref_rgb=rgb(v1.NAMES[0]);records=[]
    for name in selected:
        ci=next(i for i,r in enumerate(rows) if r['physical_camera']==name);camera=rows[ci]
        ray=dict(camera)
        if offset==0:ray['cx']+=.5;ray['cy']+=.5
        d,ids,_=camera_depth(scene,ray)
        y,x=np.nonzero(np.isfinite(d)&(ids>=len(old.triangles))&(ids<len(mesh.triangles)))
        qx=np.rint(x+offset).astype(int);qy=np.rint(y+offset).astype(int)
        valid=(qx<1920)&(qy<1080);x,y,qx,qy=x[valid],y[valid],qx[valid],qy[valid]
        obs=depths[ci][qy,qx];far=np.isfinite(obs)&(obs>d[y,x]+.003)
        x,y,qx,qy,obs=x[far],y[far],qx[far],qy[far],obs[far]
        observed=unproject(camera,qx,qy,obs);votes,_=support(observed,camera,rows,depths)
        keep=votes>=3;x,y,qx,qy,obs,observed=x[keep],y[keep],qx[keep],qy[keep],obs[keep],observed[keep]
        if not len(x):raise ValueError('Recorded positive veto could not be reproduced')
        candidate=unproject(camera,x+offset,y+offset,d[y,x]);semantics={}
        for label,points in [('candidate',candidate),('observed',observed)]:
            evidence=[]
            for row in rows:
                if row['physical_camera'] not in masks:continue
                uv,z=project_integer(row,points);xy=np.rint(uv).astype(int)
                ok=(z>0)&(xy[:,0]>2)&(xy[:,0]<1917)&(xy[:,1]>2)&(xy[:,1]<1077)
                inside=np.zeros(len(points),bool);j=np.flatnonzero(ok)
                inside[j]=masks[row['physical_camera']][xy[j,1],xy[j,0]]
                evidence.append(dict(camera=row['physical_camera'],available=int(ok.sum()),inside_skin=int(inside.sum()),outside_skin=int((ok&~inside).sum())))
            semantics[label]=evidence
        native=rgb(name);marked=native.copy();marked[y,x]=[255,30,30]
        refmark=ref_rgb.copy()
        for points,color in [(candidate,[255,30,30]),(observed,[30,255,30])]:
            uv,z=project_integer(reference,points);q=np.rint(uv).astype(int)
            valid=(z>0)&(q[:,0]>=0)&(q[:,0]<1920)&(q[:,1]>=0)&(q[:,1]<1080)
            refmark[q[valid,1],q[valid,0]]=color
        # Native portrait crops around the vetoed query rays and reference forearm.
        px=y;py=1919-x;cx=int(np.median(px));cy=int(np.median(py))
        box=(max(0,min(540,cx-270)),max(0,min(1400,cy-260)))
        box=(*box,box[0]+540,box[1]+520)
        panel=Image.new('RGB',(1620,550));draw=ImageDraw.Draw(panel)
        for k,(image,crop,label) in enumerate([(native,box,name),(marked,box,'RED: veto rays'),(refmark,(0,1400,540,1920),'reference: RED candidate / GREEN observed')]):
            panel.paste(Image.fromarray(np.rot90(image)).crop(crop),(k*540,30));draw.text((k*540+4,5),label,fill='white')
        panel.save(out/(name+'.png'))
        np.savez_compressed(out/(name+'.npz'),query_xy=np.column_stack([qx,qy]),candidate=candidate,observed=observed,votes=votes[keep])
        records.append(dict(camera=name,ray_offset=offset,veto_pixels=len(x),depth_separation_quantiles=np.quantile(obs-d[y,x],[0,.5,1]).tolist(),skin_projections=semantics))
    atomic_json(out/'result.json',dict(frame=frame,script_sha256=sha(__file__),geometry_result_sha256=sha(root/frame/'geometry_result.json'),
        records=records,geometry_changed=False,guard_changed=False,scope='up to three largest positive veto cameras; integer grid, half-grid if integer has no veto',
        source_rgb_sha256={n:sha(lookup[n]['file_path']) for n in set(selected+[v1.NAMES[0]])},
        hashes={p.name:sha(p) for p in sorted(out.iterdir()) if p.suffix in ['.png','.npz']}))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',default='001037');a=p.parse_args();run(a.root,a.frame)
