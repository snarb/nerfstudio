"""Collect train-only multiview anchors near the frozen missing-forearm patch.

This diagnoses whether a curved completion has observed support. Fits are only
depth diagnostics; no proposed mesh, renderer settings, or masks are changed.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,project
from study_confidence_depth_prior import project_integer,unproject,support,robust_fit
import study_forearm_plane_transfer_v3 as prior


def run(output,frame):
    prior.configure();v1=prior.v2.v1;root=prior.OUT
    out=output/frame;out.mkdir(parents=True,exist_ok=True)
    rows,depths,hashes=v1.load_real(frame)
    if hashes!=read(root/frame/'analysis.json')['source_depth_sha256']:raise ValueError('Changed observed depths')
    ev=np.load(root/frame/'plane/evidence.npz');accepted=ev['accepted'];distance=distance_transform_edt(~accepted)
    reference=next(r for r in rows if r['physical_camera']==v1.NAMES[0]);masks=v1.masks(frame)
    coef=np.array(read(root/frame/'analysis.json')['plane_inverse_coefficients'])
    ay,ax=np.nonzero(accepted);az=ev['depth'][ay,ax];surface=unproject(reference,ax,ay,az)
    request=dict(frame=frame,script_sha256=sha(__file__),prior_protocol_sha256=sha(root/'protocol.json'),
        source_depth_sha256=hashes,prior_evidence_sha256=sha(root/frame/'plane/evidence.npz'),
        context_distance_px=32,max_depth_distance_from_plane=.012,max_candidates_per_camera=2000,
        minimum_other_observed_views=3,minimum_available_skin_views=2,available_skin_disagreement_veto=True,
        heldout_used=False,geometry_changed=False)
    if (out/'request.json').exists() and read(out/'request.json')!=request:raise ValueError('Frozen anchor collection mismatch')
    atomic_json(out/'request.json',request);allpoints=[];alluv=[];allz=[];allvotes=[];allsources=[];counts=[]
    for ci,(camera,depth) in enumerate(zip(rows,depths)):
        uv,_=project_integer(camera,surface)
        x0,y0=np.maximum(np.floor(uv.min(0)).astype(int)-32,0);x1,y1=np.minimum(np.ceil(uv.max(0)).astype(int)+33,[1920,1080])
        if x1<=x0 or y1<=y0:continue
        y,x=np.nonzero(depth[y0:y1,x0:x1]>0);y+=y0;x+=x0
        points=unproject(camera,x,y,depth[y,x]);q,z=project_integer(reference,points);xy=np.rint(q).astype(int)
        valid=(z>0)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
        ids=np.flatnonzero(valid);xy=xy[ids];points=points[ids];q=q[ids];z=z[ids]
        plane=1/(np.column_stack([q/100,np.ones(len(q))])@coef)
        take=masks[v1.NAMES[0]][xy[:,1],xy[:,0]]&(distance[xy[:,1],xy[:,0]]<=32)&(np.abs(z-plane)<=.012)
        points,q,z=points[take],q[take],z[take]
        skin=np.zeros(len(points),np.uint8);outside=np.zeros(len(points),bool)
        for row in [r for r in rows if r['physical_camera'] in masks]:
            proj,cz=project(points,[row]);p=proj[0];cz=cz[0];pix=np.rint(p).astype(int)
            available=(cz>0)&(p[:,0]>2)&(p[:,0]<1917)&(p[:,1]>2)&(p[:,1]<1077)
            inside=np.zeros(len(points),bool);j=np.flatnonzero(available)
            inside[j]=masks[row['physical_camera']][pix[j,1],pix[j,0]]
            skin+=inside;outside|=available&~inside
        take=(skin>=2)&~outside;points,q,z=points[take],q[take],z[take]
        index=np.arange(0,len(points),max(1,int(np.ceil(len(points)/2000))))
        points,q,z=points[index],q[index],z[index]
        votes,_=support(points,camera,rows,depths);keep=votes>=3
        counts.append(dict(camera=camera['physical_camera'],tested=len(points),trusted=int(keep.sum())))
        allpoints.append(points[keep]);alluv.append(q[keep]);allz.append(z[keep]);allvotes.append(votes[keep]);allsources.append(np.full(int(keep.sum()),ci,int))
        if (ci+1)%10==0:print(frame,'anchors camera',ci+1,flush=True)
    points=np.concatenate(allpoints);uv=np.concatenate(alluv);z=np.concatenate(allz);votes=np.concatenate(allvotes);sources=np.concatenate(allsources)
    xy=np.rint(uv).astype(int);inside=accepted[xy[:,1],xy[:,0]]
    np.savez_compressed(out/'anchors.npz',points=points,reference_uv=uv,reference_z=z,other_votes=votes,source_index=sources,inside_old_patch=inside)
    fit=[];center=np.array([ax.mean(),ay.mean()]);q=(uv-center)/100
    design=np.column_stack([q,np.ones(len(q)),q[:,0]**2,q[:,0]*q[:,1],q[:,1]**2])
    split=sources%2==0
    for label,n in [('plane',3),('quadratic',6)]:
        if split.sum()<30 or (~split).sum()<30 or np.linalg.matrix_rank(design[split,:n])<n:
            fit.append(dict(model=label,status='insufficient_independent_camera_coverage'));continue
        coefficients,rmse=robust_fit(design[split,:n],1/z[split]);inv=design[~split,:n]@coefficients
        residual=np.where(inv>0,1/inv-z[~split],np.nan)
        fullcoef,_=robust_fit(design[:,:n],1/z);cq=(np.column_stack([ax,ay])-center)/100
        cdesign=np.column_stack([cq,np.ones(len(cq)),cq[:,0]**2,cq[:,0]*cq[:,1],cq[:,1]**2]);cp=cdesign[:,:n]@fullcoef
        delta=np.where(cp>0,1/cp-az,np.nan)
        fit.append(dict(model=label,status='diagnostic_fit_only',camera_partition_test_median_absolute_depth_error=float(np.nanmedian(np.abs(residual))),
            camera_partition_test_p90_absolute_depth_error=float(np.nanquantile(np.abs(residual),.9)),
            reference_center=center.tolist(),all_camera_coefficients=fullcoef.tolist(),fit_inverse_rmse=rmse,
            candidate_delta_from_plane_quantiles=np.nanquantile(delta,[0,.5,1]).tolist(),
            note='Disjoint train camera partition, not heldout evaluation or independent multiview observations'))
    rgb=np.array(Image.open(root/frame/'rgb'/(v1.NAMES[0]+'.png')).convert('RGB'));overlay=rgb.copy()
    overlay[xy[~inside,1],xy[~inside,0]]=[50,220,70];overlay[xy[inside,1],xy[inside,0]]=[255,40,30]
    panel=Image.new('RGB',(1080,550));draw=ImageDraw.Draw(panel)
    for i,im in enumerate([rgb,overlay]):panel.paste(Image.fromarray(np.rot90(im)).crop((0,1400,540,1920)),(i*540,30))
    draw.text((4,5),'real train RGB',fill='white');draw.text((544,5),'RED: inside old patch; GREEN: context anchors',fill='white')
    panel.save(out/'anchors_native.png')
    atomic_json(out/'result.json',dict(request_sha256=sha(out/'request.json'),anchors_sha256=sha(out/'anchors.npz'),panel_sha256=sha(out/'anchors_native.png'),
        total_anchors=len(points),inside_patch_anchors=int(inside.sum()),inside_patch_unique_reference_pixels=len(np.unique(xy[inside],axis=0)),
        candidate_reference_pixels=int(accepted.sum()),source_counts=counts,fit=fit,geometry_changed=False,visual_status='pending'))
    print(frame,len(points),'anchors;',int(inside.sum()),'inside patch;',fit,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True,choices=['001029','001033','001037'])
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_forearm_multiview_anchors'));a=p.parse_args();run(a.output,a.frame)
