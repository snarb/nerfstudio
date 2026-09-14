"""Isolated train-only DA3 confidence pilot; no production/default mutations.

Subcommands stage fixed-profile train RGB and infer cached pose-conditioned DA3.
All depth is camera-z in the original TSDF's normalized coordinate convention.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
from PIL import Image
from joint_temporal_texture import cameras, read, atomic_json, sha, ROOT, display, exr

DEFAULT = Path('/mnt/data/dec5_confidence_depth_prior')
MODEL = Path('/mnt/data/lookcloser_dec5_5a3_surface_repair/model_cache/hub/models--depth-anything--DA3-LARGE-1.1/snapshots/0e109ae307c5982f319a67cf6f9f99ccdc0ec97c')
REGIONS = {
    '001083':dict(camera='E004_B005_1210I7', hair=[(130,480),(126,435),(147,372),(191,325),(249,298),(335,295),(417,323),(452,362),(449,408),(407,452),(381,413),(330,411),(293,445),(254,481),(238,530),(208,559),(156,546),(130,508)],
        face_skin=[(248,503),(282,500),(307,516),(329,537),(345,559),(326,578),(312,599),(333,619),(309,634),(281,617),(251,585),(237,549)]),
    '001123':dict(camera='G004_A005_121071', hair=[(177,424),(185,373),(209,321),(251,277),(304,254),(362,258),(420,285),(464,325),(472,368),(445,411),(421,431),(416,386),(375,375),(328,384),(291,415),(274,423),(250,406),(237,435),(233,468),(210,484),(188,465)],
        face_skin=[(271,454),(300,456),(322,476),(339,493),(341,518),(316,534),(327,555),(333,567),(313,571),(289,550),(270,523),(256,491)])}
EVAL_REGIONS={
    '001083':dict(hair=[[(157,436),(172,373),(210,328),(258,300),(314,288),(370,288),(418,306),(445,339),(457,376),(446,416),(421,448),(413,447),(407,408),(392,390),(372,386),(348,390),(326,408),(298,435),(270,455),(250,455),(230,445),(219,465),(226,496),(220,531),(204,546),(185,533),(169,501)]],
        face_skin=[[(254,483),(277,482),(299,498),(321,510),(337,522),(341,538),(324,553),(320,570),(336,590),(321,603),(296,585),(273,565),(253,536),(247,512)],
                   [(311,436),(327,417),(345,404),(366,410),(382,429),(388,442),(365,439),(344,438)]]),
    '001123':dict(hair=[[(168,436),(185,374),(217,337),(258,311),(307,299),(359,300),(410,310),(451,335),(475,371),(479,410),(463,448),(441,477),(433,465),(428,432),(407,414),(383,404),(354,405),(326,423),(300,449),(278,460),(260,446),(248,458),(246,484),(251,507),(238,545),(214,547),(193,528),(179,495)]],
        face_skin=[[(283,505),(307,507),(327,525),(345,539),(354,547),(350,558),(342,574),(351,601),(340,610),(317,592),(295,572),(281,546)],
                   [(333,446),(348,430),(364,423),(380,430),(395,442),(402,454),(382,450),(359,448)]])}

def region_masks(frame):
    masks={}
    for key,poly in REGIONS[frame].items():
        if key=='camera':continue
        portrait=np.zeros((1920,1080),np.uint8)
        cv2.fillPoly(portrait,[np.array(poly,dtype=np.int32)*2],1)
        masks[key]=np.rot90(portrait,-1).astype(bool)
    return masks

def unproject(row,x,y,z,offset=0.):
    pose=np.asarray(row['transform_matrix'])
    q=np.column_stack(((x+offset-row['cx'])*z/row['fl_x'],-(y+offset-row['cy'])*z/row['fl_y'],-z))
    return q@pose[:3,:3].T+pose[:3,3]

def project_integer(row,points):
    pose=np.asarray(row['transform_matrix']);q=(points-pose[:3,3])@pose[:3,:3];z=-q[:,2]
    return np.column_stack((row['fl_x']*q[:,0]/z+row['cx'],-row['fl_y']*q[:,1]/z+row['cy'])).astype(np.float32),z

def raycast_integer(scene,row):
    import open3d as o3d
    yy,xx=np.indices((1080,1920));pose=np.asarray(row['transform_matrix']);center=pose[:3,3]
    points=unproject(row,xx.ravel(),yy.ravel(),np.ones(xx.size))
    rays=np.column_stack((np.broadcast_to(center,points.shape),points-center)).astype(np.float32)
    result=scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy().reshape(1080,1920)
    return np.where(np.isfinite(result),result,0)

def prepare_geometry(root,frames):
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    for frame in frames:
        out=root/frame;spec=read(out/'input.json');mesh=o3d.io.read_triangle_mesh(spec['mesh'])
        scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
        depths=[]
        for row in spec['frames']:depths.append(raycast_integer(scene,row))
        np.savez_compressed(out/'original_mesh_depth.npz',depth=np.stack(depths))
        masks=region_masks(frame);np.savez_compressed(out/'regions.npz',**masks)
        atomic_json(out/'regions.json',dict(coordinates='540x960 portrait thumbnail, scaled2 to native',
            manual_train_rgb_only=True,source_sha256=next(r['source_sha256'] for r in spec['frames'] if r['physical_camera']==REGIONS[frame]['camera']),regions=REGIONS[frame]))
        print(f'geometry frame={frame} raycast16 complete',flush=True)

def robust_fit(design,values):
    weight=np.ones(len(values));coef=np.linalg.lstsq(design,values,rcond=None)[0]
    for _ in range(5):
        residual=values-design@coef
        sigma=max(float(np.median(np.abs(residual-np.median(residual)))*1.4826),1e-6)
        weight=np.minimum(1,1.5*sigma/np.maximum(np.abs(residual),1e-12))
        coef=np.linalg.lstsq(design*np.sqrt(weight[:,None]),values*np.sqrt(weight),rcond=None)[0]
    return coef,float(np.sqrt(np.average((values-design@coef)**2,weights=weight)))

def support(points,reference,rows,depths,*,tolerance=.001,reprojection=1.5):
    """Distinct real train depth maps; exclude query view and require roundtrip.

    This is independent of learned depth, not statistically independent cameras.
    A frustum hit with no compatible observed depth contributes zero evidence.
    """
    count=np.zeros(len(points),np.uint8);free=np.zeros(len(points),np.uint8)
    refuv,_=project_integer(reference,points)
    for row,d in zip(rows,depths):
        if row['physical_camera']==reference['physical_camera']:continue
        uv,z=project_integer(row,points);xy=np.rint(uv).astype(np.int32)
        valid=(z>0)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
        ids=np.flatnonzero(valid)
        if not len(ids):continue
        observed=d[xy[ids,1],xy[ids,0]];available=np.isfinite(observed)&(observed>0)
        free[ids]+=available&(observed>z[ids]+tolerance*3)
        compatible=available&(np.abs(observed-z[ids])<=tolerance)
        chosen=ids[compatible]
        if not len(chosen):continue
        actual=unproject(row,xy[chosen,0],xy[chosen,1],observed[compatible])
        back,_=project_integer(reference,actual)
        roundtrip=np.linalg.norm(back-refuv[chosen],axis=1)<=reprojection
        # A near-duplicate camera cannot establish useful parallax support.
        refpose=np.asarray(reference['transform_matrix']);pose=np.asarray(row['transform_matrix'])
        a=points[chosen]-refpose[:3,3];b=points[chosen]-pose[:3,3]
        cosine=np.sum(a*b,1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
        count[chosen]+=roundtrip&(cosine<np.cos(np.deg2rad(1.)))
    return count,free

def load_real(root,frame):
    from import_colmap_mvs_depth_dataset import read_colmap_dense_array
    from render_patchmatch_camera_path import normalize_frame
    spec=read(root/frame/'real_depth_input.json');source=read(spec['transforms'])
    rows,_,metadata=cameras(frame);mapping={r['physical_camera']:r for r in source['frames']}
    scale=read(metadata)['dataparser_scale'];depths=[];hashes={}
    for row in rows:
        f=mapping[row['physical_camera']]
        normalized=normalize_frame(f,source,read(metadata))
        for key in ['transform_matrix','fl_x','fl_y','cx','cy','w','h']:
            if not np.allclose(normalized[key],row[key],rtol=0,atol=1e-6):raise ValueError(f'Real-depth camera mismatch {key}')
        path=Path(spec['dense'])/'stereo'/'depth_maps'/(f['file_path']+'.geometric.bin')
        d=read_colmap_dense_array(path)
        if d.shape==(1080,1920,1):d=d[...,0]
        d=d*scale
        if d.shape!=(1080,1920):raise ValueError('Real depths must be native1920')
        depths.append(d);hashes[row['physical_camera']]=sha(path)
    return rows,depths,dict(spec,depth_sha256=hashes,scale=scale)

def analyze(root,frames):
    from scipy import ndimage
    import open3d as o3d
    for frame in frames:
        started=time.monotonic();out=root/frame;spec=read(out/'input.json');rows=spec['frames']
        priors=np.load(out/'prior.npz')['depth'];portraitpriors=np.load(out/'prior_portrait.npz')['depth'];meshdepth=np.load(out/'original_mesh_depth.npz')['depth']
        refindex=next(i for i,r in enumerate(rows) if r['physical_camera']==REGIONS[frame]['camera']);ref=rows[refindex]
        realrows,realdepths,realreceipt=load_real(root,frame);realindex=next(i for i,r in enumerate(realrows) if r['physical_camera']==ref['physical_camera'])
        pm=realdepths[realindex];md=meshdepth[refindex];masks=region_masks(frame)
        domain=ndimage.binary_dilation(np.logical_or.reduce(list(masks.values())),iterations=18)
        yy,xx=np.nonzero(domain&(pm>0));pts=unproject(ref,xx,yy,pm[yy,xx]);counts,free=support(pts,ref,realrows,realdepths)
        countsimage=np.zeros(pm.shape,np.uint8);countsimage[yy,xx]=counts
        trusted=(countsimage>=3)&(md>0)&(np.abs(pm-md)<.001)
        distance_to_trusted=ndimage.distance_transform_edt(~trusted)
        # Fit robust metric scale/shift only at measured PatchMatch anchors that
        # agree with at least three other observed depth maps and original mesh.
        aligned=[];alignedportrait=[];alignments=[]
        for row,prior,portraitprior,depth in zip(rows,priors,portraitpriors,meshdepth):
            ridx=next(i for i,r in enumerate(realrows) if r['physical_camera']==row['physical_camera']);rd=realdepths[ridx]
            valid=(depth>0)&(prior>0)&np.isfinite(prior)&(rd>0)&(np.abs(depth-rd)<.001)
            ys,xs=np.nonzero(valid);take=np.arange(0,len(xs),max(1,len(xs)//3000));ys=ys[take];xs=xs[take]
            obs,_=support(unproject(row,xs,ys,rd[ys,xs]),row,realrows,realdepths)
            ys=ys[obs>=3];xs=xs[obs>=3]
            if len(xs)<30:raise ValueError('Insufficient independent alignment anchors')
            design=np.column_stack((prior[ys,xs],np.ones(len(xs))))
            coef,rmse=robust_fit(design,rd[ys,xs]);aligned.append(prior*coef[0]+coef[1])
            pcoef,prmse=robust_fit(np.column_stack((portraitprior[ys,xs],np.ones(len(xs)))),rd[ys,xs])
            alignedportrait.append(portraitprior*pcoef[0]+pcoef[1])
            alignments.append(dict(camera=row['physical_camera'],scale_shift=coef.tolist(),observed_anchor_rmse=rmse,
                portrait_scale_shift=pcoef.tolist(),portrait_anchor_rmse=prmse,independent_anchor_count=len(xs),minimum_other_measured_views=3))
        aligned=np.stack(aligned);prior=aligned[refindex]
        alignedportrait=np.stack(alignedportrait);portraitprior=alignedportrait[refindex]
        np.savez_compressed(out/'aligned_prior.npz',depth=aligned,portrait_depth=alignedportrait)
        # Hand-traced train-only hair region bounds the tested surface. The skin
        # region is a preservation control. Masks are fallible semantic priors.
        candidate=(md==0)&np.logical_or.reduce(list(masks.values()))
        labels,n=ndimage.label(candidate);variants={k:np.zeros(pm.shape,np.float32) for k in ['plane_boundary','da3_boundary','da3_multiview','da3_portrait_multiview']};records=[]
        for number,slices in enumerate(ndimage.find_objects(labels),1):
            if slices is None:continue
            hole=labels==number;area=int(hole.sum())
            if area<3 or area>6000:continue
            ring=ndimage.binary_dilation(hole,iterations=12)&trusted
            y,x=np.nonzero(ring)
            if len(x)<30:
                records.append(dict(component=number,area=area,status='insufficient_independent_boundary',anchors=len(x)));continue
            center=np.array([x.mean(),y.mean()]);design=np.column_stack(((x-center[0])/50,(y-center[1])/50,np.ones(len(x))))
            hy,hx=np.nonzero(hole);target=np.column_stack(((hx-center[0])/50,(hy-center[1])/50,np.ones(len(hx))))
            plane,plane_rmse=robust_fit(design,pm[y,x]);resid,resid_rmse=robust_fit(design,pm[y,x]-prior[y,x])
            presid,presid_rmse=robust_fit(design,pm[y,x]-portraitprior[y,x])
            values={'plane_boundary':target@plane,'da3_boundary':prior[hy,hx]+target@resid,'da3_portrait_multiview':portraitprior[hy,hx]+target@presid}
            rec=dict(component=number,area=area,anchors=len(x),plane_rmse=plane_rmse,da3_residual_rmse=resid_rmse,variants={})
            for mode,z in values.items():
                rmse={'plane_boundary':plane_rmse,'da3_boundary':resid_rmse,'da3_portrait_multiview':presid_rmse}[mode]
                # Same gates across both times; no extrapolated depth layer.
                near=distance_to_trusted[hy,hx]<=12
                eligible=(rmse<=.001)&near&(z>0)&(z>=np.quantile(pm[y,x],.02)-.002)&(z<=np.quantile(pm[y,x],.98)+.002)
                p=unproject(ref,hx,hy,z);observed,contradictions=support(p,ref,realrows,realdepths)
                eligible&=contradictions<2
                if mode=='da3_portrait_multiview':
                    learned,_=support(p,ref,rows,alignedportrait,tolerance=.003,reprojection=3.)
                    eligible&=(learned>=3)&(observed>=1)
                variants[mode][hy[eligible],hx[eligible]]=z[eligible]
                rec['variants'][mode]=dict(accepted=int(eligible.sum()),real_support_ge2=int((eligible&(observed>=2)).sum()),
                    free_space_rejected=int((contradictions>=2).sum()))
                if mode=='da3_boundary':
                    learned,_=support(p,ref,rows,aligned,tolerance=.003,reprojection=3.)
                    strict=eligible&(learned>=3)&(observed>=1)
                    variants['da3_multiview'][hy[strict],hx[strict]]=z[strict]
                    rec['variants']['da3_multiview']=dict(accepted=int(strict.sum()),learned_support_ge3=int((eligible&(learned>=3)).sum()))
            records.append(rec)
        baseline=o3d.io.read_triangle_mesh(spec['mesh']);v=np.asarray(baseline.vertices);t=np.asarray(baseline.triangles)
        summary=[]
        for name,addeddepth in variants.items():
            accepted=addeddepth>0;domainmesh=ndimage.binary_dilation(accepted)&((md>0)|accepted)
            y,x=np.nonzero(domainmesh);z=np.where(accepted,addeddepth,md)[y,x]
            vertices=unproject(ref,x,y,z);index=np.full(md.shape,-1,np.int32);index[y,x]=np.arange(len(x))+len(v)
            aa=index[:-1,:-1];bb=index[:-1,1:];cc=index[1:,:-1];dd=index[1:,1:];tri=[]
            for a,b,c,h in [(aa,bb,cc,accepted[:-1,:-1]|accepted[:-1,1:]|accepted[1:,:-1]),(bb,dd,cc,accepted[:-1,1:]|accepted[1:,1:]|accepted[1:,:-1])]:
                ok=(a>=0)&(b>=0)&(c>=0)&h;tri.append(np.column_stack((a[ok],b[ok],c[ok])))
            triangles=np.concatenate(tri);vv=np.concatenate((v,vertices))
            if len(triangles):triangles=triangles[np.ptp(vv[triangles],axis=1).max(1)<.002]
            tt=np.concatenate((t,triangles));mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),o3d.utility.Vector3iVector(tt));mesh.compute_vertex_normals()
            dest=out/name;dest.mkdir(exist_ok=True);o3d.io.write_triangle_mesh(str(dest/'mesh.ply'),mesh)
            np.savez_compressed(dest/'evidence.npz',proposed_depth=addeddepth,accepted=accepted)
            if not np.array_equal(vv[:len(v)],v) or not np.array_equal(tt[:len(t)],t):raise RuntimeError('Original geometry changed')
            summary.append(dict(variant=name,added_triangles=len(triangles),added_pixels=int(accepted.sum()),
                face_skin_added=int((accepted&masks['face_skin']).sum()),hair_added=int((accepted&masks['hair']).sum()),
                original_vertices_and_triangles_preserved_exactly=True,mesh_sha256=sha(dest/'mesh.ply')))
        np.savez_compressed(out/'independent_support.npz',counts=countsimage,trusted=trusted,patchmatch_reference=pm)
        # Small pseudo-holes test local shape on trusted real depth, separately
        # in skin and hair. Withheld PM is a consistency target, not ground truth.
        pseudo=[];rng=np.random.default_rng(20260914)
        for region,mask in masks.items():
            eligible=ndimage.binary_erosion(mask&trusted,iterations=14)
            ys,xs=np.nonzero(eligible);chosen=[]
            for index in rng.permutation(len(xs)):
                x,y=xs[index],ys[index]
                if any((x-a)**2+(y-b)**2<40**2 for a,b in chosen):continue
                chosen.append((x,y))
                hole=np.zeros(pm.shape,bool);hole[y-6:y+6,x-6:x+6]=True
                ring=ndimage.binary_dilation(hole,iterations=8)&~hole&trusted
                by,bx=np.nonzero(ring);hy,hx=np.nonzero(hole)
                a=np.column_stack(((bx-x)/20,(by-y)/20,np.ones(len(bx))));b=np.column_stack(((hx-x)/20,(hy-y)/20,np.ones(len(hx))))
                pc,_=robust_fit(a,pm[by,bx]);dc,_=robust_fit(a,pm[by,bx]-prior[by,bx])
                dpc,_=robust_fit(a,pm[by,bx]-portraitprior[by,bx]);pv=b@pc;dv=prior[hy,hx]+b@dc;dpv=portraitprior[hy,hx]+b@dpc
                pseudo.append(dict(region=region,center_xy=[int(x),int(y)],pixels=len(hx),
                    plane_mae=float(np.mean(np.abs(pv-pm[hy,hx]))),da3_mae=float(np.mean(np.abs(dv-pm[hy,hx]))),
                    da3_portrait_mae=float(np.mean(np.abs(dpv-pm[hy,hx])))))
                if len(chosen)>=8:break
        atomic_json(out/'analysis.json',dict(frame=frame,elapsed_seconds=time.monotonic()-started,real_depth=realreceipt,alignments=alignments,
            regions={k:dict(pixels=int(m.sum()),baseline_missing=int((m&(md==0)).sum()),trusted=int((m&trusted).sum()),
                da3_trusted_mae=float(np.mean(np.abs(prior-pm)[m&trusted])) if (m&trusted).any() else None,
                portrait_trusted_mae=float(np.mean(np.abs(portraitprior-pm)[m&trusted])) if (m&trusted).any() else None,
                measured_support_median=float(np.median(countsimage[m&(pm>0)])) if (m&(pm>0)).any() else None) for k,m in masks.items()},
            candidates=records,variants=summary,pseudo_holdout=pseudo,pseudo_holdout_is_not_ground_truth=True,
            learned_geometry_is_inferred_not_measured=True,heldout_used=False))
        print(f'analyzed frame={frame} '+str(summary),flush=True)

def render(root,frames,variants=None):
    from copy import deepcopy
    import torch
    from render_smooth_temporal_mesh_video import render_one
    from render_patchmatch_camera_path import normalize_frame
    from joint_temporal_texture import SOURCE,CALIBRATION
    torch.set_num_threads(2)
    base=read('/mnt/data/dec5_expanded_head_dynamic_150_v3/request.json')
    source={Path(r['source_dataset']).name:r for r in base['source_rows']}
    for frame in frames:
        spec=read(root/frame/'input.json');record=deepcopy(next(r for r in base['inventory'] if r['frame_id']==frame))
        calibration=read(CALIBRATION);held=deepcopy(next(r for r in calibration['frames'] if r['physical_camera']=='F004_B005_1210O9'))
        held=normalize_frame(held,calibration,read(spec['metadata']))
        for variant in (variants or ['baseline','plane_boundary','da3_boundary','da3_multiview','da3_portrait_multiview']):
            mesh=Path(spec['mesh']) if variant=='baseline' else root/frame/variant/'mesh.ply'
            out=root/frame/variant/'render_eval';out.mkdir(parents=True,exist_ok=True)
            (out/'frames').mkdir(exist_ok=True)
            r=deepcopy(record);r['mesh']=str(mesh);r['mesh_sha256']=sha(mesh);r['camera']=held
            request=dict(comparison='confidence prior matched original hard-source renderer',variant=variant,inventory=[r],
                profiles_sha256=spec['profiles_sha256'],exposure_sha256=spec['exposure_sha256'],uses_heldout_rgb=False,
                source_rows=[source[frame]],renderer_sha256=sha(Path(__file__).with_name('render_smooth_temporal_mesh_video.py')))
            if (out/'request.json').exists() and read(out/'request.json')!=request:raise ValueError('Render input changed')
            atomic_json(out/'request.json',request)
            if variant!='baseline' and sha(mesh)==spec['mesh_sha256']:
                baseline=root/frame/'baseline'/'render_eval'/'frames'/frame
                atomic_json(out/'identical_geometry_reuse.json',dict(mesh_sha256=sha(mesh),baseline=str(baseline),
                    baseline_complete_sha256=sha(baseline/'complete.json'),reason='Byte-identical original mesh and frozen renderer/camera/profiles'))
                print(f'reused identical geometry {frame} {variant}',flush=True);continue
            render_one(out,r,source[frame])
            print(f'rendered {frame} {variant}',flush=True)

def render_baseline(root,frames):return render(root,frames,variants=['baseline'])

def preview(root,frames):
    from PIL import ImageDraw
    for frame in frames:
        out=root/frame;spec=read(out/'input.json');rows=spec['frames'];masks=region_masks(frame)
        index=next(i for i,r in enumerate(rows) if r['physical_camera']==REGIONS[frame]['camera'])
        md=np.load(out/'original_mesh_depth.npz')['depth'][index]
        prior=np.load(out/'prior.npz')['depth'][index]
        rgb=np.array(Image.open(rows[index]['file_path']));domain=np.logical_or.reduce(list(masks.values()))
        depths=[md,prior];labels=['Train RGB','Original TSDF depth','Raw DA3 depth']
        if (out/'aligned_prior.npz').exists():depths.append(np.load(out/'aligned_prior.npz')['depth'][index]);labels.append('Aligned DA3 depth')
        lo,hi=np.quantile(md[(md>0)&domain],[.01,.99]);images=[rgb]
        for d in depths:
            color=cv2.applyColorMap(np.rint(np.clip((d-lo)/(hi-lo),0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)[...,::-1]
            color[(d<=0)|~domain]=0;images.append(color)
        parts=[]
        for a,label in zip(images,labels):
            im=Image.fromarray(a).transpose(Image.Transpose.ROTATE_90).crop((200,450,970,1350))
            parts.append(im)
        panel=Image.new('RGB',(770*len(parts),940));draw=ImageDraw.Draw(panel)
        for i,(part,label) in enumerate(zip(parts,labels)):
            panel.paste(part,(770*i,40));draw.text((770*i+8,10),label,fill='white')
        panel.save(out/'depth_comparison_native.png');panel.thumbnail((1540,470));panel.save(out/'depth_comparison_preview.png')
        atomic_json(out/'raw_prior_diagnostics.json',dict(regions={name:dict(raw_bias_median=float(np.median((prior-md)[mask&(md>0)])),
            raw_abs_error_median=float(np.median(np.abs(prior-md)[mask&(md>0)]))) for name,mask in masks.items()},
            reference_is_original_mesh_not_ground_truth=True))

def inspect_orientation(root,frames):
    from PIL import ImageDraw
    for frame in frames:
        out=root/frame;spec=read(out/'input.json');index=next(i for i,r in enumerate(spec['frames']) if r['physical_camera']==REGIONS[frame]['camera'])
        md=np.load(out/'original_mesh_depth.npz')['depth'][index];masks=region_masks(frame);domain=np.logical_or.reduce(list(masks.values()))
        rgb=np.array(Image.open(spec['frames'][index]['file_path']));parts=[rgb];labels=['Train RGB'];stats={}
        lo,hi=np.quantile(md[(md>0)&domain],[.01,.99])
        for mode in ['mesh','landscape','portrait']:
            if mode=='mesh':depth=md
            else:
                depth=np.load(out/('prior.npz' if mode=='landscape' else 'prior_portrait.npz'))['depth'][index]
                valid=(md>0)&(depth>0)&domain;y,x=np.nonzero(valid);take=np.arange(0,len(x),16);y=y[take];x=x[take]
                coef,rmse=robust_fit(np.column_stack((depth[y,x],np.ones(len(x)))),md[y,x]);depth=depth*coef[0]+coef[1]
                stats[mode]=dict(mesh_only_scale_shift=coef.tolist(),mesh_only_rmse=rmse,
                    regions={k:dict(median_absolute_mesh_error=float(np.median(np.abs(depth-md)[m&(md>0)])),
                        p90_absolute_mesh_error=float(np.quantile(np.abs(depth-md)[m&(md>0)],.9))) for k,m in masks.items()})
            color=cv2.applyColorMap(np.rint(np.clip((depth-lo)/(hi-lo),0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)[...,::-1]
            color[(depth<=0)|~domain]=0;parts.append(color);labels.append(mode+' depth (mesh-aligned diagnostic)')
        panel=Image.new('RGB',(770*len(parts),940));draw=ImageDraw.Draw(panel)
        for i,(part,label) in enumerate(zip(parts,labels)):
            im=Image.fromarray(part).transpose(Image.Transpose.ROTATE_90).crop((200,450,970,1350));panel.paste(im,(770*i,40));draw.text((770*i+8,10),label,fill='white')
        panel.save(out/'orientation_native.png');panel.thumbnail((1540,470));panel.save(out/'orientation_preview.png')
        atomic_json(out/'orientation_diagnostic.json',dict(statistics=stats,mesh_is_not_independent_measurement=True,used_for_geometry=False))

def setup_evaluation(root,frames):
    from joint_temporal_texture import SOURCE
    request=dict(frames=frames,variants=['baseline','plane_boundary','da3_boundary','da3_multiview','da3_portrait_multiview'],
        confidence=dict(measured_depth_tolerance=.001,return_reprojection_pixels=1.5,minimum_parallax_degrees=1.,
            minimum_boundary_other_views=3,minimum_strict_candidate_other_views=1,minimum_learned_other_views=3,
            learned_depth_tolerance=.003,learned_reprojection_pixels=3.,free_space_tolerance=.003,max_free_space_votes=1),
        boundary=dict(radius_pixels=12,minimum_anchors=30,maximum_component_area=6000,maximum_rmse=.001,range_slack=.002,maximum_triangle_extent=.002),
        heldout_camera='F004_B005_1210O9',heldout_use='evaluation only; no input/inference/selection',
        protocol_frozen_before_reading_eval_rgb=True,script_sha256_at_protocol_freeze=sha(__file__))
    path=root/'experiment_request.json'
    if path.exists() and read(path)!=request:raise ValueError('Protocol already frozen')
    atomic_json(path,request)
    exposure=read(ROOT/'exposure.json')['fixed_exposure_gain']
    for frame in frames:
        spec=read(SOURCE/frame/'transforms.json');held=next(r for r in spec['frames'] if r['physical_camera']=='F004_B005_1210O9')
        p=SOURCE/frame/held['file_path'];rgb=np.rint(255*display(exr(p),exposure)).clip(0,255).astype(np.uint8)
        out=root/frame/'evaluation';out.mkdir(exist_ok=True);im=Image.fromarray(rgb).transpose(Image.Transpose.ROTATE_90);im.save(out/'gt_native.png')
        im.thumbnail((540,960));im.save(out/'gt_preview.png');atomic_json(out/'gt.json',dict(source=str(p),source_sha256=sha(p),
            fixed_exposure=exposure,exposure_sha256=sha(ROOT/'exposure.json'),evaluation_only=True))

def evaluation_masks(root,frames):
    from PIL import ImageDraw
    for frame in frames:
        out=root/frame/'evaluation';masks={};im=Image.open(out/'gt_native.png');draw=ImageDraw.Draw(im)
        for name,polys in EVAL_REGIONS[frame].items():
            mask=np.zeros((1920,1080),np.uint8);cv2.fillPoly(mask,[np.array(poly,np.int32)*2 for poly in polys],1);masks[name]=mask.astype(bool)
            for poly in polys:draw.line([(x*2,y*2) for x,y in poly+[poly[0]]],fill='red' if name=='hair' else 'cyan',width=2)
        np.savez_compressed(out/'masks.npz',**masks);im.save(out/'mask_review_native.png');im.thumbnail((540,960));im.save(out/'mask_review.png')
        atomic_json(out/'masks.json',dict(polygons=EVAL_REGIONS[frame],coordinate_system='540x960 portrait scaled2 to native',
            manual_gt_only=True,prediction_inspected_when_tracing=False,evaluation_only=True,gt_sha256=sha(out/'gt_native.png'),masks_sha256=sha(out/'masks.npz')))

def score(root,frames):
    import torch
    from PIL import ImageDraw
    from score_colmap_patchmatch_tsdf_face import masked_display_metrics
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    torch.set_num_threads(2);model=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval();summary=[]
    for frame in frames:
        out=root/frame/'evaluation';gt=np.array(Image.open(out/'gt_native.png'));masks=np.load(out/'masks.npz');variants=read(root/'experiment_request.json')['variants']
        predictions=[gt];names=['GT']
        for variant in variants:
            p=root/frame/variant/'render_eval'/'frames'/frame/'frame.png'
            reuse=root/frame/variant/'render_eval'/'identical_geometry_reuse.json'
            if reuse.exists():
                receipt=read(reuse);baseline=Path(receipt['baseline'])
                if sha(baseline/'complete.json')!=receipt['baseline_complete_sha256']:raise ValueError('Changed reused baseline')
                p=baseline/'frame.png'
            if not p.exists():continue
            pred=np.array(Image.open(p));predictions.append(pred);names.append(variant)
            for region in ['face_skin','hair']:
                with torch.inference_mode():
                    result=masked_display_metrics(torch.tensor(pred.transpose(2,0,1)/255,dtype=torch.float32,device='cuda'),
                        torch.tensor(gt.transpose(2,0,1)/255,dtype=torch.float32,device='cuda'),torch.tensor(masks[region],device='cuda'),model)
                result={k.removeprefix('face_'):v for k,v in result.items()}
                summary.append(dict(frame=frame,variant=variant,region=region,**result,prediction_sha256=sha(p),gt_sha256=sha(out/'gt_native.png')))
        for region in ['face_skin','hair']:
            y,x=np.nonzero(masks[region]);x0,x1=max(0,x.min()-12),min(1080,x.max()+13);y0,y1=max(0,y.min()-12),min(1920,y.max()+13)
            width,height=x1-x0,y1-y0;panel=Image.new('RGB',(width*len(predictions),height+32));draw=ImageDraw.Draw(panel)
            for i,(pred,name) in enumerate(zip(predictions,names)):
                panel.paste(Image.fromarray(pred[y0:y1,x0:x1]),(width*i,32));draw.text((width*i+4,8),name,fill='white')
            panel.save(out/(region+'_comparison_native.png'));panel.thumbnail((1600,650));panel.save(out/(region+'_comparison_preview.png'))
    atomic_json(root/'metrics.json',dict(rows=summary,protocol='Exact masked RGB PSNR; tight region bbox zero outside mask for SSIM and AlexNet LPIPS',
        eval_rgb_used_for_prediction=False,eval_roi_used_for_geometry=False))
    print(f'scored rows={len(summary)}',flush=True)

def moving_view(root,frames):
    import open3d as o3d
    from PIL import ImageDraw
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    base=read('/mnt/data/dec5_expanded_head_dynamic_150_v3/request.json')
    for frame in frames:
        out=root/frame/'moving_view';out.mkdir(exist_ok=True);spec=read(root/frame/'input.json')
        record=next(r for r in base['inventory'] if r['frame_id']==frame);row=record['camera'];variants=read(root/'experiment_request.json')['variants'];images=[];stats=[]
        first_depth=None
        for name in variants:
            path=Path(spec['mesh']) if name=='baseline' else root/frame/name/'mesh.ply'
            mesh=o3d.io.read_triangle_mesh(str(path));v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles);mesh.compute_triangle_normals();normals=np.asarray(mesh.triangle_normals)
            d,ids,bary=camera_depth(scene_for(v,t),row);hit=np.isfinite(d);rgb=np.zeros((1080,1920,3),np.uint8)
            light=np.array(row['transform_matrix'])[:3,2];shade=.2+.8*np.abs(normals[ids[hit]]@light);rgb[hit]=(shade[:,None]*255).clip(0,255).astype(np.uint8)
            if first_depth is None:first_depth=np.where(hit,d,0)
            newly_visible=hit&(first_depth==0);infront=hit&(first_depth>0)&(d<first_depth-.001)
            im=Image.fromarray(rgb).transpose(Image.Transpose.ROTATE_90);im.save(out/(name+'_clay_native.png'));images.append(im)
            np.savez_compressed(out/(name+'_depth.npz'),depth=np.where(hit,d,0),newly_visible=newly_visible)
            stats.append(dict(variant=name,mesh_sha256=sha(path),newly_visible_pixels=int(newly_visible.sum()),
                pixels_occluding_old_surface_by_more_than_001=int(infront.sum()),camera_pose=row,
                ray_miss_is_not_evidence_of_missing_anatomy=True))
        # A fixed view-independent crop is only for reading the native comparison.
        panel=Image.new('RGB',(700*len(images),832));draw=ImageDraw.Draw(panel)
        for i,(im,name) in enumerate(zip(images,variants)):
            panel.paste(im.crop((300,450,1000,1250)),(700*i,32));draw.text((700*i+8,8),name,fill='white')
        panel.save(out/'clay_comparison_native.png');panel.thumbnail((1750,500));panel.save(out/'clay_comparison_preview.png')
        atomic_json(out/'comparison.json',dict(camera_source='/mnt/data/dec5_expanded_head_dynamic_150_v3/request.json',
            rows=stats,uses_target_rgb=False,baseline_is_original_unrepaired_tsdf=True))

def failure_patches(root,frames):
    from scipy import ndimage
    from PIL import ImageDraw
    for frame in frames:
        spec=read(root/frame/'input.json');analysis=read(root/frame/'analysis.json');index=next(i for i,r in enumerate(spec['frames']) if r['physical_camera']==REGIONS[frame]['camera'])
        md=np.load(root/frame/'original_mesh_depth.npz')['depth'][index];evidence=np.load(root/frame/'independent_support.npz');pm=evidence['patchmatch_reference'];trusted=evidence['trusted']
        priors=np.load(root/frame/'aligned_prior.npz');prior=priors['depth'][index];portrait=priors['portrait_depth'][index]
        rgb=np.array(Image.open(spec['frames'][index]['file_path']));candidate=(md==0)&np.logical_or.reduce(list(region_masks(frame).values()));labels,_=ndimage.label(candidate)
        selected=sorted([c for c in analysis['candidates'] if 'variants' in c],key=lambda c:c['area'],reverse=True)[:3]
        out=root/frame/'failure_patches';out.mkdir(exist_ok=True)
        for rec in selected:
            hole=labels==rec['component'];ring=ndimage.binary_dilation(hole,iterations=12)&trusted
            colored=rgb.copy();colored[hole]=[255,0,0];colored[ring]=[0,255,0]
            anchor=float(np.median(pm[ring]));parts=[rgb,colored];names=['Train RGB','Miss + anchors']
            for depth,name in [(pm,'PatchMatch'),(prior,'DA3'),(portrait,'DA3 upright')]:
                color=cv2.applyColorMap(np.rint(np.clip((depth-anchor+.005)/.01,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)[...,::-1];color[depth<=0]=0;parts.append(color);names.append(name)
            y,x=np.nonzero(np.rot90(hole));box=(max(0,x.min()-24),max(0,y.min()-24),min(1080,x.max()+25),min(1920,y.max()+25));width,height=box[2]-box[0],box[3]-box[1]
            panel=Image.new('RGB',(width*len(parts),height+48));draw=ImageDraw.Draw(panel)
            for i,(part,name) in enumerate(zip(parts,names)):
                panel.paste(Image.fromarray(np.rot90(part)).crop(box),(i*width,48));draw.text((i*width+4,5),name,fill='white')
            panel.save(out/(f'component_{rec["component"]}_native.png'))
        atomic_json(out/'selected.json',dict(components=selected,depth_color_range_normalized_units='local boundary median +/-0.005',
            original_mesh_is_candidate_locator_only=True,green_anchors_require_at_least_3_other_observed_depths=True))

def audit(root,frames):
    import open3d as o3d
    from joint_temporal_texture import HELD_CAMERAS
    records=[];metrics=read(root/'metrics.json')['rows']
    for frame in frames:
        spec=read(root/frame/'input.json');analysis=read(root/frame/'analysis.json')
        assert sha(spec['mesh'])==spec['mesh_sha256']
        assert len(spec['frames'])==16 and not ({r['physical_camera'] for r in spec['frames']}&HELD_CAMERAS)
        for row in spec['frames']:assert sha(row['source_file_path'])==row['source_sha256']
        assert len(analysis['real_depth']['depth_sha256'])==62
        old=o3d.io.read_triangle_mesh(spec['mesh']);v=np.asarray(old.vertices);t=np.asarray(old.triangles)
        mask=np.load(root/frame/'evaluation/masks.npz')['face_skin'];base=np.array(Image.open(root/frame/'baseline/render_eval/frames'/frame/'frame.png'))
        for row in analysis['variants']:
            name=row['variant'];p=root/frame/name/'mesh.ply';assert sha(p)==row['mesh_sha256'];mesh=o3d.io.read_triangle_mesh(str(p))
            assert np.array_equal(np.asarray(mesh.vertices)[:len(v)],v) and np.array_equal(np.asarray(mesh.triangles)[:len(t)],t)
            assert len(np.asarray(mesh.triangles))==len(t)+row['added_triangles']
            pred=root/frame/name/'render_eval/frames'/frame/'frame.png';reuse=root/frame/name/'render_eval/identical_geometry_reuse.json'
            if reuse.exists():
                assert sha(p)==spec['mesh_sha256'];receipt=read(reuse);pred=Path(receipt['baseline'])/'frame.png'
                assert sha(Path(receipt['baseline'])/'complete.json')==receipt['baseline_complete_sha256']
            image=np.array(Image.open(pred));changed=int(((image!=base).any(-1)&mask).sum())
            for region in ['face_skin','hair']:
                metric=next(r for r in metrics if r['frame']==frame and r['variant']==name and r['region']==region)
                assert metric['prediction_sha256']==sha(pred)
                assert all(np.isfinite(metric[k]) for k in ['psnr','ssim','lpips'])
            records.append(dict(frame=frame,variant=name,original_geometry_preserved_exactly=True,changed_skin_pixels=changed,
                mesh_sha256=sha(p),prediction_sha256=sha(pred),reused_identical_geometry=reuse.exists()))
    assert len(metrics)==len(frames)*10
    atomic_json(root/'audit.json',dict(rows=records,metrics_rows=len(metrics),script_sha256=sha(__file__),
        source_rgb_unchanged=True,heldout_excluded_from_inference=True,passed=True,visual_acceptance_is_separate=True))
    print(f'audit passed frames={len(frames)} metrics={len(metrics)}',flush=True)

def stage(root, frames):
    profiles=read(ROOT/'camera_profiles.json'); gains=dict(zip(profiles['physical_cameras'],profiles['rgb_gain']))
    exposure=read(ROOT/'exposure.json')['fixed_exposure_gain']
    # Frozen physical-camera membership across time, including both diagnostic views.
    prefixes={'D004_A005','D004_B005','E004_A005','E004_B005','E004_C005',
              'F004_A005','F004_C005','G004_A005','G004_B005','G004_C005',
              'H004_A005','H004_B005','H004_C005','I004_A005','I004_B005','I004_C005'}
    for frame in frames:
        out=root/frame; out.mkdir(parents=True,exist_ok=True)
        rows,mesh,meta=cameras(frame); rows=[r for r in rows if r['physical_camera'][:9] in prefixes]
        if len(rows)!=16: raise ValueError(f'Expected frozen16, got {len(rows)}')
        staged=[]
        for r in rows:
            path=out/'rgb'/(r['physical_camera']+'.png');path.parent.mkdir(exist_ok=True)
            rgb=np.rint(255*display(exr(r['file_path'])*np.array(gains[r['physical_camera']]),exposure)).clip(0,255).astype(np.uint8)
            Image.fromarray(rgb).save(path)
            staged.append(dict(r,source_file_path=r['file_path'],file_path=str(path),source_sha256=sha(r['file_path'])))
        atomic_json(out/'input.json',dict(frame=frame,frames=staged,mesh=str(mesh),mesh_sha256=sha(mesh),metadata=str(meta),
            profiles_sha256=sha(ROOT/'camera_profiles.json'),exposure_sha256=sha(ROOT/'exposure.json'),
            heldout_rgb_used=False,script_sha256=sha(__file__)))
        print(f'staged frame={frame} cameras={len(rows)}',flush=True)

def infer(root, frames, portrait=False):
    from depth_anything_3.api import DepthAnything3
    from build_da3_pose_depth_dataset import nerfstudio_camera_to_da3
    import torch
    torch.cuda.set_per_process_memory_fraction(.35)
    model=DepthAnything3.from_pretrained(str(MODEL)).to('cuda')
    for frame in frames:
        out=root/frame; spec=read(out/'input.json'); rows=spec['frames']
        filename='prior_portrait.npz' if portrait else 'prior.npz'
        if (out/filename).exists(): raise ValueError('Prior already exists')
        camera=[nerfstudio_camera_to_da3(r,{}) for r in rows]
        paths=[r['file_path'] for r in rows]
        if portrait:
            paths=[];rot=np.eye(4,dtype=np.float32);rot[:3,:3]=[[0,1,0],[-1,0,0],[0,0,1]]
            for i,row in enumerate(rows):
                p=out/'rgb_portrait'/(row['physical_camera']+'.png');p.parent.mkdir(exist_ok=True)
                Image.open(row['file_path']).transpose(Image.Transpose.ROTATE_90).save(p);paths.append(str(p))
                ext,k=camera[i];knew=np.array([[k[1,1],0,k[1,2]],[0,k[0,0],1919-k[0,2]],[0,0,1]],np.float32)
                camera[i]=(rot@ext,knew)
        torch.cuda.synchronize();started=time.monotonic()
        result=model.inference(paths,extrinsics=np.stack([c[0] for c in camera]),
            intrinsics=np.stack([c[1] for c in camera]),align_to_input_ext_scale=True,process_res=1008,
            process_res_method='upper_bound_resize',export_dir=None,use_ray_pose=False,ref_view_strategy='saddle_balanced')
        torch.cuda.synchronize();seconds=time.monotonic()-started
        size=(1080,1920) if portrait else (1920,1080)
        def resize(d):
            d=cv2.resize(d,size,interpolation=cv2.INTER_LINEAR)
            return np.rot90(d,-1).copy() if portrait else d
        depth=np.stack([resize(d) for d in result.depth]);conf=np.stack([resize(d) for d in result.conf])
        np.savez_compressed(out/filename,depth=depth,confidence=conf)
        atomic_json(out/('inference_portrait.json' if portrait else 'inference.json'),dict(model=str(MODEL),forward_seconds=seconds,shape=list(depth.shape),
            input_sha256=sha(out/'input.json'),prior_sha256=sha(out/filename),portrait_rotation=portrait,
            cuda_peak_bytes=torch.cuda.max_memory_allocated(),learned_depth_is_independent_measurement=False))
        print(f'inferred frame={frame} seconds={seconds:.2f}',flush=True)

def infer_portrait(root,frames):return infer(root,frames,portrait=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['stage','infer','infer_portrait','prepare_geometry','analyze','render','render_baseline','preview','inspect_orientation','setup_evaluation','evaluation_masks','score','moving_view','failure_patches','audit'])
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--frames',nargs='+',default=['001083','001123'])
    a=p.parse_args();cv2.setNumThreads(2);globals()[a.command](a.output,a.frames)
