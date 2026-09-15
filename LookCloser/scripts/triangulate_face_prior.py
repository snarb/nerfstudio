"""Calibrated train-only face correspondence gate, with unused train validation views."""
from __future__ import annotations
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from PIL import Image,ImageDraw
from study_multiview_face_prior import OUT,FRAMES,sha,read,save,portrait_to_native

def projection_matrices(rows):
    matrices=[]
    for row in rows:
        pose=np.asarray(row['transform_matrix'],float)@np.diag([1.,-1.,-1.,1.]);ext=np.linalg.inv(pose)[:3]
        k=np.array([[row['fl_x'],0,row['cx']],[0,row['fl_y'],row['cy']],[0,0,1.]])
        matrices.append(k@ext)
    return np.array(matrices)

def project(points,matrices):
    points=np.atleast_2d(points);hom=np.column_stack((points,np.ones(len(points))))
    q=np.einsum('vij,pj->pvi',matrices,hom);z=q[...,2]
    return q[...,:2]/np.where(np.abs(z)>1e-12,z,1)[...,None],z

def linear_point(matrices,pixels):
    a=(pixels[:,:,None]*matrices[:,None,2]-matrices[:,:2]).reshape(-1,4)
    _,s,vh=np.linalg.svd(a)
    if s[-2]<s[0]*1e-10 or abs(vh[-1,3])<1e-10:raise ValueError('Degenerate calibrated rays')
    return vh[-1,:3]/vh[-1,3]

def refine(matrices,pixels,initial):
    fit=least_squares(lambda p:(project(p,matrices)[0][0]-pixels).ravel(),initial,loss='soft_l1',f_scale=2.,max_nfev=100)
    if not fit.success or not np.isfinite(fit.x).all():raise ValueError('Robust triangulation failed')
    return fit.x

def max_parallax(point,rows):
    ray=point-np.array([r['transform_matrix'] for r in rows])[:,:3,3]
    ray/=np.linalg.norm(ray,axis=1)[:,None]
    return float(np.degrees(np.arccos(np.clip((ray@ray.T).min(),-1,1))))

def triangulate(rows,pixels,arm):
    pixels=np.asarray(pixels,float)
    if len(rows)<3 or pixels.shape!=(len(rows),2) or not np.isfinite(pixels).all():raise ValueError('Need three finite fit observations')
    p=projection_matrices(rows)
    if arm=='all_fit_robust':
        point=refine(p,pixels,linear_point(p,pixels));selected=np.ones(len(rows),bool)
    elif arm=='train_consensus':
        pairs=list(combinations(range(len(rows)),2));rng=np.random.default_rng(73)
        if len(pairs)>256:pairs=[pairs[i] for i in rng.choice(len(pairs),256,replace=False)]
        hypotheses=[]
        for a,b in pairs:
            try:
                point=linear_point(p[[a,b]],pixels[[a,b]])
                if max_parallax(point,[rows[a],rows[b]])>=1:hypotheses.append(point)
            except ValueError:continue
        if not hypotheses:raise ValueError('No nondegenerate triangulation pair')
        uv,z=project(hypotheses,p);error=np.linalg.norm(uv-pixels[None],axis=-1);inlier=(error<=3)&(z>0)
        counts=inlier.sum(1);median=np.median(np.where(inlier,error,1e6),axis=1)
        best=np.lexsort((median,-counts))[0];selected=inlier[best]
        if selected.sum()<max(3,int(np.ceil(.3*len(rows)))):raise ValueError('Insufficient train consensus')
        point=refine(p[selected],pixels[selected],np.array(hypotheses[best]))
    else:raise ValueError('Unknown arm')
    uv,z=project(point,p);error=np.linalg.norm(uv[0]-pixels,axis=-1)
    if (z[0,selected]<=0).any() or max_parallax(point,[r for r,k in zip(rows,selected) if k])<1:raise ValueError('Invalid depth/parallax')
    return point,error,selected,max_parallax(point,[r for r,k in zip(rows,selected) if k])

def quantiles(values):
    values=np.asarray(values,float)
    return dict(count=len(values),median=float(np.median(values)) if len(values) else None,p90=float(np.percentile(values,90)) if len(values) else None)

def run(root):
    request=read(root/'request.json');inference=read(root/'inference.json')
    assert inference['request_sha256']==sha(root/'request.json')
    dest=root/'triangulation';dest.mkdir(exist_ok=False)
    save(dest/'request.json',dict(source_request_sha256=sha(root/'request.json'),inference_sha256=sha(root/'inference.json'),
        script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('study_multiview_face_prior.py')),
        model_depth_used=False,validation_used_for_fitting=False,gate=request['gate'],arms=request['triangulation']['arms']))
    all_summaries=[]
    for frame in request['frames']:
        spec=read(root/frame/'input.json');rows={r['camera']['physical_camera']:r['camera'] for r in spec['inputs']}
        detected={r['camera']:r for r in inference['records'] if r['frame']==frame and r['detected']==1}
        val=[n for n in rows if any(n.startswith(prefix) for prefix in request['validation_prefixes'])]
        fit=[n for n in rows if n not in val];xy={n:portrait_to_native(r['portrait_xy'])[:468] for n,r in detected.items()}
        crop=request['crop'];valid={n:(np.array(detected[n]['portrait_xy'])[:468,0]>=crop[0])&(np.array(detected[n]['portrait_xy'])[:468,0]<crop[2])&
            (np.array(detected[n]['portrait_xy'])[:468,1]>=crop[1])&(np.array(detected[n]['portrait_xy'])[:468,1]<crop[3]) for n in detected}
        for arm in request['triangulation']['arms']:
            folder=dest/frame/arm;folder.mkdir(parents=True);entries=[];points=np.zeros((468,3));good=np.zeros(468,bool)
            for k in range(468):
                names=[n for n in fit if n in xy and valid[n][k]];vn=[n for n in val if n in xy and valid[n][k]]
                entry=dict(index=k,fit_camera_count=len(names),validation_camera_count=len(vn),passed=False)
                try:
                    point,errors,selected,angle=triangulate([rows[n] for n in names],[xy[n][k] for n in names],arm)
                    points[k]=point;good[k]=True
                    ve=np.array([])
                    if vn:
                        uv,z=project(point,projection_matrices([rows[n] for n in vn]));ve=np.linalg.norm(uv[0]-np.array([xy[n][k] for n in vn]),axis=-1)
                        assert (z>0).all()
                    sq=quantiles(errors[selected]);vq=quantiles(ve)
                    passed=len(vn)>=2 and selected.sum()>=3 and sq['median']<=2 and vq['median']<=2 and vq['p90']<=4
                    entry.update(status='triangulated',fit_cameras=names,selected_fit_cameras=[n for n,s in zip(names,selected) if s],
                        fit_errors=errors.tolist(),selected_fit_errors=errors[selected].tolist(),validation_cameras=vn,
                        validation_errors=ve.tolist(),parallax_degrees=angle,passed=bool(passed))
                except (ValueError,AssertionError) as e:entry.update(status='rejected',reason=str(e))
                entries.append(entry)
            np.savez_compressed(folder/'points.npz',points=points,triangulated=good,passed=np.array([e['passed'] for e in entries]))
            summaries={}
            for group,indices in request['groups'].items():
                selected=[entries[k] for k in indices];fe=[x for e in selected for x in e.get('selected_fit_errors',[])];ve=[x for e in selected for x in e.get('validation_errors',[])]
                summaries[group]=dict(landmarks=len(indices),triangulated=sum(e['status']=='triangulated' for e in selected),
                    passing=sum(e['passed'] for e in selected),passing_fraction=sum(e['passed'] for e in selected)/len(indices),fit=quantiles(fe),validation=quantiles(ve))
            gate=all(summaries[g]['passing_fraction']>=.8 for g in ['jaw','cheek'])
            summary=dict(frame=frame,arm=arm,groups=summaries,lower_face_gate_passed=gate,validation_cameras=val,
                detected_validation=[n for n in val if n in detected],detected_train_total=len(detected),fit_camera_count=len(fit),
                points_sha256=sha(folder/'points.npz'),geometry_changed=False)
            save(folder/'result.json',dict(summary=summary,landmarks=entries));all_summaries.append(summary)
            for name in [n for n in val if n in detected]+[n for n in fit if n in detected and n.startswith(('G004_A','M004_A'))]:
                item=next(r for r in spec['inputs'] if r['camera']['physical_camera']==name);assert sha(item['path'])==item['sha256']
                image=Image.open(item['path']).convert('RGB');draw=ImageDraw.Draw(image)
                uv,_=project(points[good],projection_matrices([rows[name]]));pred=np.stack((uv[:,0,1],1919-uv[:,0,0]),axis=-1)-np.array(crop[:2])
                observed=np.array(detected[name]['portrait_xy'])[:468][good]-np.array(crop[:2]);indices=np.flatnonzero(good)
                lower=set(request['groups']['jaw']+request['groups']['cheek'])
                for k,a,b in zip(indices,pred,observed):
                    if k not in lower:continue
                    draw.line([tuple(a),tuple(b)],fill='yellow',width=1)
                    draw.ellipse((a[0]-2,a[1]-2,a[0]+2,a[1]+2),fill='cyan')
                    draw.ellipse((b[0]-1,b[1]-1,b[0]+1,b[1]+1),fill='red')
                box=detected[name]['native_review_box'];image.crop(tuple(v-crop[i%2] for i,v in enumerate(box))).save(folder/(name+'_reprojection.png'))
            print(frame,arm,{g:{'fit':x['fit']['median'],'val':x['validation']['median'],'p90':x['validation']['p90'],'pass':x['passing_fraction']} for g,x in summaries.items()},flush=True)
    save(dest/'result.json',dict(request_sha256=sha(dest/'request.json'),summaries=all_summaries,
        geometry_completion_allowed=all(r['lower_face_gate_passed'] for r in all_summaries if r['arm']=='train_consensus'),
        heldout_used=False,geometry_changed=False,visual_review_pending=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=OUT);run(p.parse_args().root)
