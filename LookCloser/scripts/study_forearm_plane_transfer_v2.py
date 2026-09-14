"""Bounded three-time follow-up: FOV-unknown semantics and dual-ray clipping.

No source/default writes, no new learned inference. V1 artifacts remain immutable.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import numpy as np
import cv2
import study_forearm_plane_transfer as v1
import study_forearm_confidence_prior as pilot
from joint_temporal_texture import read, atomic_json, sha

OUT=Path('/mnt/data/dec5_forearm_plane_transfer_v2')
FRAMES=['001029','001033','001037']
V1=Path('/mnt/data/dec5_forearm_plane_transfer')
BASE={f:V1/f for f in FRAMES}
BASE['001033']=V1/'algorithm_replay_001033/001033'
POLYGONS={**v1.POLYGONS,'001033':pilot.POLYGONS}

def configure():
    v1.OUT=OUT;v1.FRAMES=FRAMES;v1.POLYGONS=POLYGONS;v1.CONTROLS=OUT/'controls'

def eligible_points(z,distance,skinvotes,disagree,trusted_free):
    return np.isfinite(z)&(z>0)&(distance<=100)&(skinvotes>=2)&(disagree==0)&(trusted_free==0)

def semantic_domain(row,points,inside):
    """V2 uses all in-frame pixels; the separate v3 runner overrides only this."""
    return inside

def freeze(protocol_overrides=None):
    if OUT.exists():raise ValueError('Use a fresh v2 root; never overwrite a frozen study')
    OUT.mkdir(parents=True);(OUT/'controls').mkdir()
    provenance={}
    for frame in FRAMES:
        src=BASE[frame];dest=OUT/frame;dest.mkdir();spec=read(src/'input.json')
        control=Path('/mnt/data/dec5_forearm_depth_control_001033') if frame=='001033' else Path('/mnt/data/dec5_forearm_temporal_transfer/controls')/frame
        assert (control/'complete.json').exists()
        (OUT/'controls'/frame).symlink_to(control,target_is_directory=True)
        for name in ['input.json','diagnostic.npz','analysis.json']:
            target=dest/('v1_analysis.json' if name=='analysis.json' else name)
            shutil.copyfile(src/name,target)
        (dest/'rgb').mkdir()
        for image in (src/'rgb').glob('*.png'):shutil.copyfile(image,dest/'rgb'/image.name)
        shutil.copyfile(src/'skin_masks_native.png',dest/'skin_masks_native.png')
        provenance[frame]={str(src/n):sha(src/n) for n in ['input.json','diagnostic.npz','analysis.json','plane/mesh.ply','plane_clipped/mesh.ply']}
        assert sha(spec['mesh'])==spec['mesh_sha256']
    protocol=read(V1/'protocol.json');protocol.update(frames=FRAMES,skin_polygons=POLYGONS,
        rule='V2 only: out-of-image unknown, >=2 in-frame skin votes, every in-frame disagreement veto; dual-lattice clipping',
        minimum_reviewed_skin_views=2,out_of_image_is_unknown=True,in_frame_semantic_disagreement_veto=True,
        clip_ray_offsets=[0.,.5],unchanged_plane_coefficients_and_trusted_anchors=True,
        skin_polygons_reused_without_adjustment=True,v1_provenance=provenance,
        original_protocol_sha256=sha(V1/'protocol.json'),script_sha256_at_freeze=sha(__file__),
        rendered_color_scope='Matched raw local ablation; no production source-mask or temporal-label wrappers',
        native_clay_review_required_before_rgb=True)
    if protocol_overrides:protocol.update(protocol_overrides)
    atomic_json(OUT/'protocol.json',protocol)
    print(f'Frozen v{protocol.get("study_version",2)} across all three times; no held-out or neural input',flush=True)

def analyze(frame):
    import open3d as o3d
    from scipy import ndimage
    out=OUT/frame
    if (out/'analysis.json').exists():raise ValueError('Keep existing v2 analysis')
    prior=read(out/'v1_analysis.json');spec=read(out/'input.json');rows,depths,hashes=v1.load_real(frame)
    assert {str(Path(p).resolve()):h for p,h in hashes.items()}=={str(Path(p).resolve()):h for p,h in prior['source_depth_sha256'].items()}
    data=np.load(out/'diagnostic.npz');skin=v1.masks(frame);byname={r['physical_camera']:i for i,r in enumerate(rows)}
    ref=rows[byname[v1.NAMES[0]]];md=data[v1.NAMES[0]+'_mesh'];trusted=data[v1.NAMES[0]+'_trusted']
    hy,hx=np.nonzero(skin[v1.NAMES[0]]&(md==0));coef=np.array(prior['plane_inverse_coefficients'])
    z=1/(np.column_stack((hx/100,hy/100,np.ones(len(hx))))@coef);points=v1.unproject(ref,hx,hy,z)
    observed,free=v1.support(points,ref,rows,depths)
    skinvotes=np.zeros(len(z),np.uint8);disagree=np.zeros(len(z),np.uint8);unknown=np.zeros(len(z),np.uint8);trusted_free=np.zeros(len(z),np.uint8)
    for name in v1.NAMES:
        uv,cz=v1.project_integer(rows[byname[name]],points);xy=np.rint(uv).astype(int)
        inside=(cz>0)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080);ids=np.flatnonzero(inside);qx,qy=xy[ids].T
        available=semantic_domain(rows[byname[name]],points,inside);semantic_ids=np.flatnonzero(available)
        sx,sy=xy[semantic_ids].T;is_skin=skin[name][sy,sx]
        skinvotes[semantic_ids]+=is_skin;disagree[semantic_ids]+=~is_skin;unknown+=~available
        trusted_free[ids]+=data[name+'_trusted'][qy,qx]&(depths[byname[name]][qy,qx]>cz[ids]+.003)
    distance=ndimage.distance_transform_edt(~trusted)[hy,hx]
    eligible=eligible_points(z,distance,skinvotes,disagree,trusted_free)
    added=np.zeros(md.shape,np.float32);added[hy[eligible],hx[eligible]]=z[eligible];accepted=added>0
    old=o3d.io.read_triangle_mesh(spec['mesh']);v=np.asarray(old.vertices);t=np.asarray(old.triangles)
    domain=ndimage.binary_dilation(accepted)&((md>0)|accepted);y,x=np.nonzero(domain)
    newv=v1.unproject(ref,x,y,np.where(accepted,added,md)[y,x]);index=np.full(md.shape,-1,np.int32);index[y,x]=np.arange(len(x))+len(v)
    a=index[:-1,:-1];b=index[:-1,1:];c=index[1:,:-1];d=index[1:,1:];tri=[]
    for aa,bb,cc,h in [(a,b,c,accepted[:-1,:-1]|accepted[:-1,1:]|accepted[1:,:-1]),(b,d,c,accepted[:-1,1:]|accepted[1:,1:]|accepted[1:,:-1])]:
        ok=(aa>=0)&(bb>=0)&(cc>=0)&h;tri.append(np.column_stack((aa[ok],bb[ok],cc[ok])))
    triangles=np.concatenate(tri);vv=np.concatenate((v,newv));triangles=triangles[np.ptp(vv[triangles],axis=1).max(1)<.002]
    tt=np.concatenate((t,triangles));dest=out/'plane';dest.mkdir()
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),o3d.utility.Vector3iVector(tt));mesh.compute_vertex_normals();o3d.io.write_triangle_mesh(str(dest/'mesh.ply'),mesh)
    assert np.array_equal(vv[:len(v)],v) and np.array_equal(tt[:len(t)],t)
    np.savez_compressed(dest/'evidence.npz',depth=added,accepted=accepted,all_candidate_xy=np.column_stack((hx,hy)),observed=observed,
        raw_free_space=free,trusted_free_space=trusted_free,skin_views=skinvotes,in_frame_disagreements=disagree,unknown_views=unknown)
    record=dict(prior,source_depth_sha256=hashes,accepted_pixels=int(eligible.sum()),added_triangles=len(triangles),
        accepted_zero_measured_votes=int((eligible&(observed==0)).sum()),accepted_ge2_measured_votes=int((eligible&(observed>=2)).sum()),
        accepted_two_skin_one_unknown=int((eligible&(skinvotes==2)&(unknown==1)).sum()),
        rejected_in_frame_disagreement=int((disagree>0).sum()),rejected_fewer_two_skin_views=int((skinvotes<2).sum()),
        rejected_trusted_free_space=int((trusted_free>0).sum()),mesh_sha256=sha(dest/'mesh.ply'),protocol_sha256=sha(OUT/'protocol.json'),
        source_v1_analysis_sha256=sha(out/'v1_analysis.json'),study_version=read(OUT/'protocol.json').get('study_version',2))
    record.pop('rejected_skin_limit',None);record.pop('elapsed_seconds',None)
    atomic_json(out/'analysis.json',record)
    print(frame,{k:record[k] for k in ['accepted_pixels','added_triangles','accepted_two_skin_one_unknown','accepted_zero_measured_votes']},flush=True)

def lattice_depth(scene,camera,offset):
    from bake_joint_temporal_mesh import camera_depth
    if offset==.5:return camera_depth(scene,camera)
    assert offset==0.
    import open3d as o3d
    yy,xx=np.indices((1080,1920));center=np.asarray(camera['transform_matrix'])[:3,3]
    points=v1.unproject(camera,xx.ravel(),yy.ravel(),np.ones(xx.size))
    rays=np.column_stack((np.broadcast_to(center,points.shape),points-center)).astype(np.float32)
    hit=scene.cast_rays(o3d.core.Tensor(rays))
    return hit['t_hit'].numpy().reshape(1080,1920),hit['primitive_ids'].numpy().reshape(1080,1920),None

def clip_plane(frame):
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    out=OUT/frame;spec=read(out/'input.json');rows,depths,_=v1.load_real(frame)
    old=o3d.io.read_triangle_mesh(spec['mesh']);ov=np.asarray(old.vertices);ot=np.asarray(old.triangles);sceneold=scene_for(ov,ot)
    mesh=o3d.io.read_triangle_mesh(str(out/'plane/mesh.ply'));v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles).copy()
    selected={'moving':spec['moving_camera']};selected.update({r['physical_camera']:r for r in rows if r['physical_camera'] in v1.NAMES})
    olddepth={(name,offset):lattice_depth(sceneold,camera,offset)[0] for name,camera in selected.items() for offset in [0.,.5]}
    # Verify the integer calibration conversion against the independently implemented old helper.
    for name,camera in selected.items():
        ref=v1.raycast_integer(sceneold,camera);actual=np.where(np.isfinite(olddepth[name,0.]),olddepth[name,0.],0)
        assert np.array_equal(actual>0,ref>0) and np.allclose(actual,ref,rtol=0,atol=1e-6)
    rounds=[]
    if (out/'plane_clipped/mesh.ply').exists():raise ValueError('Keep existing clipped v2 candidate')
    for iteration in range(4):
        scene=scene_for(v,t);remove=set();checks=[]
        for name,camera in selected.items():
            for offset in [0.,.5]:
                d,ids,_=lattice_depth(scene,camera,offset);od=olddepth[name,offset]
                front=np.isfinite(od)&np.isfinite(d)&(d<od-.001);y,x=np.nonzero(front)
                votes,_=v1.support(v1.unproject(camera,x,y,od[y,x],offset=offset),camera,rows,depths)
                implicated=ids[y[votes>=3],x[votes>=3]];assert (implicated>=len(ot)).all();remove.update(implicated.tolist())
                checks.append(dict(camera=name,ray_offset=offset,trusted_old_occluded=int((votes>=3).sum())))
        rounds.append(dict(iteration=iteration,removed_triangles=len(remove),checks=checks))
        if not remove:break
        keep=np.ones(len(t),bool);keep[list(remove)]=False;t=t[keep]
    assert np.array_equal(t[:len(ot)],ot) and np.array_equal(v[:len(ov)],ov)
    dest=out/'plane_clipped';dest.mkdir();result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t))
    result.compute_vertex_normals();o3d.io.write_triangle_mesh(str(dest/'mesh.ply'),result)
    atomic_json(dest/'result.json',dict(rounds=rounds,parent_plane_sha256=sha(out/'plane/mesh.ply'),protocol_sha256=sha(OUT/'protocol.json'),
        retained_added_triangles=len(t)-len(ot),removed_total=len(np.asarray(mesh.triangles))-len(t),mesh_sha256=sha(dest/'mesh.ply'),
        original_geometry_preserved=True,heldout_used=False,guard_zero_in_last_pass=all(r['trusted_old_occluded']==0 for r in rounds[-1]['checks'])))
    print(frame,rounds,flush=True)

def evaluation_inputs(frame):
    """Reuse the previously frozen GT-only scoring region, never as geometry input."""
    v1.evaluation_inputs(frame)
    if frame=='001033':
        points=read('/mnt/data/dec5_forearm_confidence_prior/metrics.json')['heldout_forearm_polygon']
        v1.evaluation_mask(frame,np.array(points).ravel().tolist())
    else:
        region=read(BASE[frame]/'evaluation/regions.json')
        v1.evaluation_mask(frame,np.array(region['points']).ravel().tolist(),region.get('not_visible',False))

def seam_diagnosis(frame):
    """Read-only reason codes at the visually identified 001037 moving seam."""
    from PIL import Image,ImageDraw
    assert frame=='001037'
    out=OUT/frame;spec=read(out/'input.json');analysis=read(out/'analysis.json');maps=np.load(out/'diagnostic.npz')
    ref=next(r for r in spec['frames'] if r['physical_camera']==v1.NAMES[0]);camera=spec['moving_camera']
    coef=np.array(analysis['plane_inverse_coefficients']);pose=np.asarray(ref['transform_matrix']);cp=np.asarray(camera['transform_matrix'])
    normal=pose[:3,:3]@np.array([coef[0]*ref['fl_x']/100,-coef[1]*ref['fl_y']/100,-(coef[0]*ref['cx']/100+coef[1]*ref['cy']/100+coef[2])])
    # Entire native crop enables checking the localization, then exact seam box is summarized.
    py,px=np.indices((1920,1080));x=1919-py.ravel();y=px.ravel()
    direction=v1.unproject(camera,x,y,np.ones(len(x)),offset=.5)-cp[:3,3]
    z=(1-normal@(cp[:3,3]-pose[:3,3]))/(direction@normal);points=cp[:3,3]+z[:,None]*direction
    uv,cz=v1.project_integer(ref,points);q=np.rint(uv).astype(int);inside=(cz>0)&(q[:,0]>=0)&(q[:,0]<1920)&(q[:,1]>=0)&(q[:,1]<1080)
    codes=np.zeros(len(x),np.uint8);ids=np.flatnonzero(inside);qx,qy=q[ids].T
    skin=v1.masks(frame)[v1.NAMES[0]];md=maps[v1.NAMES[0]+'_mesh'];ev=np.load(out/'plane/evidence.npz')
    codes[ids]=1;codes[ids[skin[qy,qx]]]=2;codes[ids[skin[qy,qx]&(md[qy,qx]>0)]]=3
    candidates=ev['all_candidate_xy'];a,b=candidates.T
    reason=np.zeros(md.shape,np.uint8);reason[b,a]=4
    reason[b[ev['in_frame_disagreements']>0],a[ev['in_frame_disagreements']>0]]=5
    reason[b[ev['skin_views']<2],a[ev['skin_views']<2]]=6
    reason[b[ev['trusted_free_space']>0],a[ev['trusted_free_space']>0]]=7
    reason[ev['accepted']]=8
    rr=reason[qy,qx];codes[ids[rr>0]]=rr[rr>0]
    codes=codes.reshape(1920,1080)
    baseline=np.rot90(np.load(out/'review/moving/baseline_depth.npz')['depth'])
    plane=np.rot90(np.load(out/'review/moving/plane_depth.npz')['depth'])
    # Fixed native read-only localization from the already viewed clay image.
    box=(280,1862,395,1883);roi=np.zeros(codes.shape,bool);roi[box[1]:box[3],box[0]:box[2]]=True
    missing=roi&(baseline==0)&(plane==0)
    labels={0:'outside reference image',1:'outside reference skin',2:'unclassified skin',3:'original reference hit excluded',
        4:'other candidate rejection',5:'in-frame skin disagreement',6:'fewer than two skin supports',7:'trusted free-space veto',8:'accepted reference point but missing moving triangle coverage'}
    palette=np.array([[0,0,0],[80,80,80],[255,255,255],[255,0,0],[128,0,255],[255,128,0],[255,255,0],[255,0,255],[0,255,255]],np.uint8)
    image=palette[codes];image[~((baseline==0)&(plane==0))]//=4
    im=Image.fromarray(image).crop((150,1780,560,1920));ImageDraw.Draw(im).rectangle((130,82,245,103),outline='white')
    im.save(out/'seam_reason_native.png')
    central=np.zeros(codes.shape,bool);central[1862:1883,310:370]=True;central&=missing
    per_view=[];selected_ids=np.flatnonzero(central.ravel());allskin=v1.masks(frame)
    for row in spec['frames']:
        uv,zr=v1.project_integer(row,points[selected_ids]);qq=np.rint(uv).astype(int)
        inside=(zr>0)&(qq[:,0]>=0)&(qq[:,0]<1920)&(qq[:,1]>=0)&(qq[:,1]<1080);ii=np.flatnonzero(inside)
        disagreement=np.zeros(len(qq),bool);disagreement[ii]=~allskin[row['physical_camera']][qq[ii,1],qq[ii,0]]
        native=np.column_stack((qq[disagreement,1],1919-qq[disagreement,0]))
        per_view.append(dict(camera=row['physical_camera'],in_frame_disagreement_pixels=int(disagreement.sum()),
            disagreement_native_min=native.min(0).tolist() if len(native) else None,
            disagreement_native_max=native.max(0).tolist() if len(native) else None,out_of_image=int((~inside).sum())))
    originalids=np.flatnonzero((codes.ravel()==3)&missing.ravel());oq=q[originalids]
    original_depth=md[oq[:,1],oq[:,0]];plane_ref=cz[originalids]
    atomic_json(out/'seam_diagnosis.json',dict(frame=frame,moving_native_box=box,missing_pixels=int(missing.sum()),
        counts={labels[k]:int((missing&(codes==k)).sum()) for k in labels},
        central_strip_box=[310,1862,370,1883],central_missing=int(central.sum()),
        central_counts={labels[k]:int((central&(codes==k)).sum()) for k in labels},central_train_projection=per_view,
        original_hit_camera_z_minus_plane_quantiles=np.quantile(original_depth-plane_ref,[0,.5,1]).tolist() if len(oq) else None,
        original_reference_pixels=np.unique(oq,axis=0).tolist(),diagnostic_only_no_geometry_or_rule_changed=True,
        plane_mesh_sha256=sha(out/'plane/mesh.ply'),rule_is_same_as_v1=True))
    print(read(out/'seam_diagnosis.json'),flush=True)

def summarize():
    from PIL import Image,ImageDraw
    results=[];metrics=[];audits=[];panel=Image.new('RGB',(1720,1455));draw=ImageDraw.Draw(panel);version=read(OUT/'protocol.json').get('study_version',2)
    for j,frame in enumerate(FRAMES):
        out=OUT/frame;analysis=read(out/'analysis.json');clip=read(out/'plane_clipped/result.json')
        geometry=read(out/'geometry_review.json')['rows'];semantic=read(out/'semantic_review.json')['rows']
        assert clip['guard_zero_in_last_pass']
        assert all(r['occluded_old_pixels_with_ge3_measured_votes']==0 for r in geometry if r['variant']=='plane_clipped')
        assert all(r['integer_front_old_pixels_with_ge3_measured_votes']==0 for r in semantic)
        results.append(dict(frame=frame,analysis=analysis,clipping=clip,geometry=geometry,semantic=semantic))
        metrics.extend(read(out/'metrics.json')['rows']);audits.append(read(out/'audit.json'))
        src=BASE[frame] if frame!='001033' else pilot.OUT/frame
        parts=[out/'rgb'/(v1.NAMES[1]+'.png'),out/'baseline/render_train_H_A/frames'/frame/'frame.png',
            src/'plane_clipped/render_train_H_A/frames'/frame/'frame.png',out/'plane_clipped/render_train_H_A/frames'/frame/'frame.png']
        for i,(path,label) in enumerate(zip(parts,['Train reference','Original TSDF','V1 clipped plane',f'V{version} clipped plane'])):
            im=Image.open(path)
            if i==0:im=Image.fromarray(np.rot90(np.array(im)))
            panel.paste(im.crop((0,1500,430,1920)),(i*430,j*485+30));draw.text((i*430+4,j*485+8),frame+' '+label,fill='white')
    panel.save(OUT/'three_time_H_A_native.png')
    atomic_json(OUT/'summary.json',dict(results=results,metrics=metrics,audits=audits,protocol_sha256=sha(OUT/'protocol.json'),
        original_geometry_preserved=True,production_defaults_changed=False,raw_ablation_not_production_color=True,
        sparse_times_not_full_temporal_validation=True,inferred_not_measured=True,manual_review_is_separate=True))
    print(f'Three v{version} times, 18 renders / 18 metric region records audited',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','analyze','clip_plane','geometry_review','semantic_review','render','evaluation_inputs','score','audit','summarize','seam_diagnosis'])
    parser.add_argument('--frame',choices=FRAMES);parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();OUT=args.output
    cv2.setNumThreads(2);configure()
    if args.command in ['freeze','summarize']:globals()[args.command]()
    elif args.command in ['analyze','clip_plane','evaluation_inputs','seam_diagnosis']:globals()[args.command](args.frame)
    else:getattr(v1,args.command)(args.frame)
