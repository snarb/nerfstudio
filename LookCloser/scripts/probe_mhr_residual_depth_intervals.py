"""Posthoc all-mask ray interval diagnostic; never an input to fitting."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from fit_mhr_silhouette_conformance import prepare,sample_sdf
from joint_temporal_texture import project
from bake_joint_temporal_mesh import camera_depth
from admit_mhr_local_patch_depth import Scene2
from run_local_mhr_completion import read,save,sha,require
from diagnose_local_mhr_residual_rays import landscape_pixel

SOURCE=Path('/mnt/data/dec5_mhr_production_patch_001193')
PRIOR=Path('/mnt/data/dec5_mhr_silhouette_convergence/fit.npz')
LIMIT=.003
STEP=.00001
VERIFY_STEP=.000001
REFINE=.00000001


def segments(accepted):
    padded=np.r_[False,np.asarray(accepted,bool),False].astype(int)
    return list(zip(np.flatnonzero(np.diff(padded)==1),np.flatnonzero(np.diff(padded)==-1)-1))


def greedy_veto_cover(outside):
    """Deterministic camera cover of sampled rejected positions, not minimum core."""
    uncovered=np.ones(outside.shape[1],bool);chosen=[]
    while uncovered.any():
        counts=(outside&uncovered).sum(1);i=int(counts.argmax())
        if counts[i]==0:break
        chosen.append(i);uncovered &= ~outside[i]
    return chosen,bool(not uncovered.any())


class MaskProbe:
    def __init__(self,rows,masks,names):
        self.rows=rows;self.masks=[masks[names.index(r['physical_camera'])].astype(bool) for r in rows]
        self.sdfs=[(distance_transform_edt(~m)-distance_transform_edt(m)).astype(np.float32) for m in self.masks]
    def evaluate(self,points):
        uv,z=project(np.asarray(points).reshape(-1,3),self.rows)
        available=(z>0)&(uv[:,:,0]>2)&(uv[:,:,0]<1917)&(uv[:,:,1]>2)&(uv[:,:,1]<1077)
        binary=np.zeros_like(available);sdf=np.full(available.shape,np.nan,np.float32)
        for i,(mask,field) in enumerate(zip(self.masks,self.sdfs)):
            ids=np.flatnonzero(available[i]);p=uv[i,ids];xy=np.rint(p).astype(int)
            binary[i,ids]=~mask[xy[:,1],xy[:,0]];sdf[i,ids]=sample_sdf(field,p)[0]
        return available,binary,sdf
    def accepted(self,point,mode):
        av,b,s=self.evaluate(np.asarray(point).reshape(1,3))
        outside=b if mode=='binary' else (s>0)
        return bool(av.sum()>=2 and not outside.any())


def refine_edge(probe,origin,direction,lo,hi,mode,left_value):
    while hi-lo>REFINE:
        mid=(lo+hi)/2;value=probe.accepted(origin+mid*direction,mode)
        if value==left_value:lo=mid
        else:hi=mid
    return [float(lo),float(hi)]


def main():
    import open3d as o3d
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    require(not out.exists(),'New diagnostic root required')
    rq=read(SOURCE/'residual_hole/result.json');e=np.load(SOURCE/'residual_hole/evidence.npz')
    require(sha(SOURCE/'residual_hole/evidence.npz')==rq['evidence_sha256'],'Changed query evidence')
    inputs={str(SOURCE/'residual_hole/result.json'):sha(SOURCE/'residual_hole/result.json'),str(SOURCE/'residual_hole/evidence.npz'):rq['evidence_sha256']}
    for path,h in rq['input_hashes'].items():require(sha(path)==h,'Changed residual input');inputs[path]=h
    _,rows,masks,names,binding,_=prepare()
    ab=read(SOURCE/'admission/request.json')['inputs']
    for key in ['mask_sha256','override_sha256','mask_names_sha256','depth_receipt']:require(binding[key]==ab[key],'Wrong masks/calibration binding')
    for key in ['mask_path','override_path']:
        path=Path(binding[key]);inputs[str(path)]=sha(path)
    inputs[str(Path(binding['mask_path']).parent/'cameras.json')]=binding['mask_names_sha256']
    inputs[str(Path(binding['override_path']).parent/'request.json')]=binding['independent_mask_override_request_sha256']
    base=SOURCE/'admission/rgb/F004_E/baseline/frames/001193'
    render=read(base/'result.json');camera=render['camera'];rgb=np.array(Image.open(base/'frame.png'))
    for name,h in read(base/'complete.json')['hashes'].items():require(sha(base/name)==h,'Changed base render');inputs[str(base/name)]=h
    fit=np.load(PRIOR);scene=Scene2(fit['vertices'],fit['triangles']);d,ids,bary= camera_depth(scene,camera)
    pose=np.asarray(camera['transform_matrix']);ext=np.linalg.inv(pose@np.diag([1.,-1.,-1.,1.])).astype(np.float32)
    k=np.array([[camera['fl_x'],0,camera['cx']],[0,camera['fl_y'],camera['cy']],[0,0,1]],np.float32)
    rays=scene.create_rays_pinhole(o3d.core.Tensor(k),o3d.core.Tensor(ext),camera['w'],camera['h']).numpy()
    target=e['portrait_xy'];x,y=landscape_pixel(target[:,0],target[:,1]);np.testing.assert_array_equal(d[y,x],e['prior_depth']);np.testing.assert_array_equal(ids[y,x],e['prior_face'])
    uv=bary[y,x];bary_points=(fit['vertices'][fit['triangles'][ids[y,x]]]*np.c_[1-uv.sum(1),uv][...,None]).sum(1)
    np.testing.assert_array_equal(bary_points,e['prior_points'])
    saved=np.rot90(np.load(base/'target_depth.npz')['depth']);pd=np.rot90(d)
    controls=[];seeds=[[682,1144],[689,1144],[696,1144],[682,1154],[689,1154],[696,1154]]
    gy,gx=np.mgrid[1135:1176,670:721];good=(saved[gy,gx]>0)&np.isfinite(pd[gy,gx])&(rgb[gy,gx].max(2)>0)
    pool=np.c_[gx[good],gy[good]]
    for seed in seeds:
        order=np.argsort(((pool-np.array(seed))**2).sum(1),kind='stable')
        chosen=next(q.tolist() for q in pool[order] if q.tolist() not in controls);controls.append(chosen)
    xy=np.concatenate([target,np.asarray(controls)]);xx,yy=landscape_pixel(xy[:,0],xy[:,1]);ray=rays[yy,xx];t0=d[yy,xx]
    center=ray[:,:3].astype(np.float64)+t0[:,None].astype(np.float64)*ray[:,3:].astype(np.float64)
    bary_ray_delta=np.linalg.norm(center[:30]-e['prior_points'],axis=1)
    probe=MaskProbe(rows,masks,names);out.mkdir();save(out/'request.json',dict(frame='001193',view='F004_E',range=[-LIMIT,LIMIT],step=STEP,
        verification_step=VERIFY_STEP,boundary_refinement_bracket=REFINE,parameter='offset of actual calibrated non-unit t_hit; positive goes farther along ray',
        camera=camera,train_cameras=[r['physical_camera'] for r in rows],mask_binding=binding,mask_dilation=0,sdf_tolerance_pixels=0,
        availability='z>0; 2<u<1917; 2<v<1077; out-of-domain unknown; >=2 available masks',
        binary='np.rint of common frozen guard projection',sdf='bilinear EDT(background)-EDT(foreground), <=0 on same projected coordinates',
        controls_selected_without_mask_feasibility=True,control_seeds=seeds,control_pixels=controls,input_hashes=inputs,
        prior_depth_triangle_barycentric_replay_exact=True,max_barycentric_vs_ray_point_difference=float(bary_ray_delta.max()),
        script_sha256=sha(__file__),helpers={n:sha(Path(__file__).with_name(n)) for n in ['fit_mhr_silhouette_conformance.py','joint_temporal_texture.py','admit_mhr_local_patch_depth.py','bake_joint_temporal_mesh.py']},
        target_scan_is_fit_input=False,geometry_or_masks_modified=False,production_accepted=False))
    all_scans={};records=[]
    for label,step in [('coarse',STEP),('verification',VERIFY_STEP)]:
        offsets=np.linspace(-LIMIT,LIMIT,round(2*LIMIT/step)+1)
        points=center[:,None,:]+offsets[None,:,None]*ray[:,None,3:]
        av,b,s=probe.evaluate(points);shape=(len(rows),len(xy),len(offsets));av=av.reshape(shape);b=b.reshape(shape);s=s.reshape(shape)
        all_scans[label]=(offsets,av,b,s)
        np.savez_compressed(out/(label+'.npz'),offsets=offsets,available=av,binary_outside=b,signed_distance=s,
            portrait_xy=xy,rays=ray,prior_depth=t0,prior_triangle_ids=ids[yy,xx],prior_points=center)
        print(label,'samples',len(offsets),'binary feasible rays',int(((~b.any(0))&(av.sum(0)>=2)).any(1).sum()),flush=True)
    offsets,av,b,s=all_scans['verification'];coff,cav,cb,cs=all_scans['coarse']
    for qi,pixel in enumerate(xy):
        row=dict(pixel=pixel.tolist(),cohort='residual' if qi<30 else 'existing_hit_control',prior_depth=float(t0[qi]),
            prior_triangle_id=int(ids[yy[qi],xx[qi]]),original_depth=float(saved[pixel[1],pixel[0]]),
            euclidean_per_t_scale=float(np.linalg.norm(ray[qi,3:])),modes={})
        for mode,outside,coarse in [('binary',b[:,qi],cb[:,qi]),('bilinear_sdf',s[:,qi]>0,cs[:,qi]>0)]:
            good=~outside.any(0)&(av[:,qi].sum(0)>=2);segments_here=segments(good);intervals=[]
            for lo,hi in segments_here:
                left=[float(offsets[lo])]*2 if lo==0 else refine_edge(probe,center[qi],ray[qi,3:],offsets[lo-1],offsets[lo],mode,False)
                right=[float(offsets[hi])]*2 if hi==len(offsets)-1 else refine_edge(probe,center[qi],ray[qi,3:],offsets[hi],offsets[hi+1],mode,True)
                intervals.append(dict(left_bracket=left,right_bracket=right,left_truncated=bool(lo==0),right_truncated=bool(hi==len(offsets)-1)))
            cover,complete=greedy_veto_cover(outside);zero=len(offsets)//2
            persistent=np.flatnonzero(outside.all(1));fraction=outside.mean(1)
            row['modes'][mode]=dict(feasible=bool(good.any()),intervals=intervals,feasible_samples=int(good.sum()),
                coarse_feasible=bool((~coarse.any(0)&(cav[:,qi].sum(0)>=2)).any()),zero_feasible=bool(good[zero]),
                zero_veto_cameras=[rows[i]['physical_camera'] for i in np.flatnonzero(outside[:,zero])],
                persistent_veto_cameras=[rows[i]['physical_camera'] for i in persistent],
                greedy_sample_cover=[rows[i]['physical_camera'] for i in cover] if not good.any() else [],
                greedy_cover_complete=complete if not good.any() else None,
                veto_sample_fractions={rows[i]['physical_camera']:float(fraction[i]) for i in np.flatnonzero(fraction)},
                available_count_min=int(av[:,qi].sum(0).min()),minimum_veto_count=int(outside.sum(0).min()))
        # At the measured original control hit, not merely at the prior center.
        if qi>=30:
            point=ray[qi,:3]+row['original_depth']*ray[qi,3:];aa,bb,ss=probe.evaluate(point)
            row['control_original_hit']=dict(offset=row['original_depth']-row['prior_depth'],binary_veto=[rows[i]['physical_camera'] for i in np.flatnonzero(bb[:,0])],sdf_veto=[rows[i]['physical_camera'] for i in np.flatnonzero(ss[:,0]>0)])
        records.append(row)
    overview=Image.fromarray(rgb);draw=ImageDraw.Draw(overview)
    for qi,(x,y) in enumerate(xy):draw.point((x,y),fill='magenta' if qi<30 else 'cyan')
    overview.crop((650,1105,740,1200)).resize((540,570),Image.Resampling.NEAREST).save(out/'native_ray_locations_6x.png')
    overview.crop((500,1030,830,1270)).save(out/'native_underchin_context.png')
    # Simple categorical ray/depth evidence, not an interpolated numeric chart.
    for label,scan in all_scans.items():
        off,aa,bb,ss=scan;canvas=Image.new('RGB',(640,60+len(xy)*12));draw=ImageDraw.Draw(canvas)
        draw.text((5,3),'magenta: residual; cyan: control | binary left / SDF right | green feasible',fill='white')
        draw.text((5,18),'offset -0.003                    +0.003    -0.003                    +0.003',fill='white')
        for qi in range(len(xy)):
            for j,outside in enumerate([bb,ss>0]):
                good=~outside[:,qi].any(0)&(aa[:,qi].sum(0)>=2);indices=np.linspace(0,len(off)-1,300).round().astype(int)
                bar=np.where(good[indices,None],np.array([30,190,80]),np.array([160,35,35])).astype(np.uint8)
                canvas.paste(Image.fromarray(np.tile(bar[None],(9,1,1))),(30+j*305,50+qi*12))
            draw.text((1,50+qi*12),str(qi+1),fill='magenta' if qi<30 else 'cyan')
        canvas.save(out/(label+'_intervals.png'))
    save(out/'result.json',dict(request_sha256=sha(out/'request.json'),records=records,
        note='Finite scan with 1e-6 verification and 1e-8 transition brackets; intervals narrower than verification step are not globally excluded.',
        mask_feasibility_is_not_anatomical_or_measured_depth_truth=True,geometry_changed=False,
        outputs={f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file()}))
    print('done',[(c,{m:sum(r['modes'][m]['feasible'] for r in records if r['cohort']==c) for m in ['binary','bilinear_sdf']}) for c in ['residual','existing_hit_control']],flush=True)


if __name__=='__main__':main()
