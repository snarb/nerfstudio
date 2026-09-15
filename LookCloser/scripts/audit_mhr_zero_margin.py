"""Exact offset-only replay and fixed-cohort comparison; targets posthoc only."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageDraw
import study_mhr_zero_margin as study
from study_multiview_face_prior import read, save, sha
from triangulate_face_prior import quantiles


def audit():
    proof = study.configure()
    q = read(study.FINAL/'protocol.json')
    assert json.dumps(proof, sort_keys=True) == json.dumps(q['zero_margin_adapter'], sort_keys=True)
    assert q['recipe']['boundary_tolerance_pixels'] == 0
    old = read(study.CONTROL/'protocol.json')['recipe']
    assert {k:v for k,v in q['recipe'].items() if k!='boundary_tolerance_pixels'} == {k:v for k,v in old.items() if k!='boundary_tolerance_pixels'}
    factory = study.continuation.make_optimizer
    def factory_at_replay_root():
        factory.__globals__['ROOT'] = study.continuation.ROOT
        return factory()
    study.continuation.make_optimizer = factory_at_replay_root
    import audit_mhr_silhouette_convergence as frozen
    frozen.main()
    save(study.ROOT/'audit_wrapper.json', dict(status='passed', wrapper_sha256=sha(__file__),
        offset_only_proof=proof, frozen_auditor_path=str(Path(frozen.__file__).resolve()),
        frozen_auditor_sha256=sha(frozen.__file__), audit_sha256=sha(study.FINAL/'audit.json'),
        frozen_audit_cohort_tolerance_pixels=2, production_modified=False))


def compare():
    import open3d as o3d
    from bake_joint_temporal_mesh import camera_depth
    from study_confidence_depth_prior import unproject
    from study_multiview_face_prior import CROP
    fit=study.fit
    study.configure()
    a=np.load(study.FINAL/'fit.npz'); old=np.load(study.CONTROL/'fit.npz')
    np.testing.assert_array_equal(a['baseline'],old['baseline'])
    np.testing.assert_array_equal(a['triangles'],old['triangles'])
    _,rows,masks,names,evidence,validation=fit.prepare()
    sdfs=[]
    for row in rows:
        m=masks[names.index(row['physical_camera'])].astype(bool)
        sdfs.append((distance_transform_edt(~m)-distance_transform_edt(m)).astype(np.float32))
    variants=[('base',a['baseline']),('margin2',old['vertices']),('margin0',a['vertices'])]
    grids=[]
    for label,v in variants:
        grid=np.full((62,int(a['active'].sum())),np.nan)
        for ci,(ids,values,_) in enumerate(fit.silhouette_samples(v[a['active']],rows,sdfs)):grid[ci,ids]=values
        grids.append(grid)
    common=np.logical_and.reduce([np.isfinite(g) for g in grids]);cohorts={}
    for split,selected in [('fit',~validation),('reserved',validation)]:
        take=common&selected[:,None]
        for tolerance in [0,2]:
            cohorts[f'{split}_margin{tolerance}']=dict(samples=int(take.sum()),arms={
                label:dict(outside=int((g[take]>tolerance).sum()),mean_excess=float(np.maximum(g[take]-tolerance,0).mean()),
                           excess=quantiles(np.maximum(g[take]-tolerance,0))) for (label,_),g in zip(variants,grids)})
    dest=study.ROOT/'comparison';dest.mkdir(exist_ok=False)
    np.savez_compressed(dest/'cohorts.npz',base=grids[0],margin2=grids[1],margin0=grids[2],common=common,validation=validation)
    # Saved production residual coordinates are only queried after fitting.
    source=Path('/mnt/data/dec5_mhr_production_patch_001193')
    residual=source/'residual_hole/evidence.npz'
    xy=np.load(residual)['portrait_xy']
    camfile=source/'admission/rgb/F004_E/baseline/frames/001193/result.json'
    camera=read(camfile)['camera'];native=np.c_[1919-xy[:,1],xy[:,0]]
    center=np.asarray(camera['transform_matrix'])[:3,3]
    direction=unproject(camera,native[:,0],native[:,1],np.ones(len(native)),offset=.5)-center
    rays=o3d.core.Tensor(np.c_[np.broadcast_to(center,direction.shape),direction].astype(np.float32))
    residual_stats={};point_evidence={};tri=a['triangles'];files=[]
    def panel(items,path):
        w,h=items[0][1].size;im=Image.new('RGB',(len(items)*w,h+24));draw=ImageDraw.Draw(im)
        for i,(label,p) in enumerate(items):im.paste(p,(i*w,24));draw.text((i*w+3,5),label,fill='white')
        im.save(path);files.append(dict(path=str(path),sha256=sha(path)))
    clays=[]
    for label,v in variants[1:]:
        scene=fit.Scene2(v,tri);hit=scene.cast_rays(rays);d=hit['t_hit'].numpy();ok=np.isfinite(d)
        points=center+direction[ok]*d[ok,None];ids=hit['primitive_ids'].numpy()[ok]
        samples=np.concatenate([points[:,None,:],v[tri[ids]]],axis=1).reshape(-1,3)
        sdf=np.full((62,len(samples)),np.nan)
        for ci,(ix,val,_) in enumerate(fit.silhouette_samples(samples,rows,sdfs)):sdf[ci,ix]=val
        reshaped=sdf.reshape(62,-1,4);outside=(reshaped>0).any(2).sum(0)
        topo=np.load((study.CONTROL if label=='margin2' else study.FINAL)/'review_v2/silhouette_topology.npz')
        unsafe=np.union1d(topo['reversed_triangles'],np.unique(topo['strict_pairs']))
        residual_stats[label]=dict(hits=int(ok.sum()),total_rays=len(ok),first_hit_zero_sdf_pass=int(((reshaped[:,:,0]>0).sum(0)==0).sum()),
            hit_and_three_parent_vertices_zero_sdf_pass=int((outside==0).sum()),outside_cameras=quantiles(outside),
            unsafe_parent_hits=int(np.isin(ids,unsafe).sum()),sdf_pixels=quantiles(sdf[np.isfinite(sdf)]))
        point_evidence.update({label+'_depth':d,label+'_points':points,label+'_faces':ids,label+'_sdf':reshaped})
        depth,faces,_=camera_depth(scene,camera);mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(tri));mesh.compute_triangle_normals()
        valid=np.isfinite(depth);col=np.full((*depth.shape,3),20,np.uint8)
        col[valid]=(60+170*abs(np.asarray(mesh.triangle_normals)[faces[valid]]@np.array([.3,.4,.866])))[:,None]
        clays.append((label,Image.fromarray(np.rot90(col)).crop((635,1100,755,1210))))
    panel(clays,dest/'residual_prior_clay.png')
    np.savez_compressed(dest/'residual.npz',portrait_xy=xy,**point_evidence)
    # Compare already rendered native silhouette columns without rerendering.
    for name in ['C004_E005_1210OC_clay.png','G004_B005_1210FG_clay.png','requested_hole_prior_only.png']:
        candidates=list((study.FINAL/'review_v2').glob(name))
        if not candidates and name.startswith('C004_E'):candidates=list((study.FINAL/'review_v2').glob('C004_E*_clay.png'))
        if not candidates:continue
        name=candidates[0].name;items=[]
        for label,root in [('margin2',study.CONTROL),('margin0',study.FINAL)]:
            im=Image.open(root/'review_v2'/name).convert('RGB');count=3 if name.startswith('requested') else 4
            items.append((label,im.crop((im.width*(count-1)//count,24,im.width,im.height))))
        panel(items,dest/name)
    topology={label:read(root/'review_v2/result.json')['geometry']['silhouette'] for label,root in [('margin2',study.CONTROL),('margin0',study.FINAL)]}
    primary={}
    for label,root in [('margin2',study.CONTROL),('margin0',study.FINAL)]:
        p=np.load(root/'locality/silhouette.npz')
        primary[label]=dict(hits=len(p['points']),strict_zero_sdf_pass=int(((p['sdf']>0).sum(0)==0).sum()),
            rim_distance=quantiles(np.linalg.norm(p['rim_closest']-p['rim_points'],axis=1)),rim_signed_offset=quantiles(p['rim_signed_offset']))
    bindings=[residual,camfile,study.CONTROL/'fit.npz',study.FINAL/'fit.npz',study.FINAL/'protocol.json']
    save(dest/'result.json',dict(script_sha256=sha(__file__),fixed_cohorts=cohorts,residual_posthoc=residual_stats,
        primary_posthoc=primary,topology=topology,displacement_from_margin2=quantiles(np.linalg.norm(a['vertices']-old['vertices'],axis=1)),
        input_hashes={str(p):sha(p) for p in bindings},files=files,target_used_in_fit=False,production_modified=False))
    print(json.dumps(dict(cohorts=cohorts,residual=residual_stats,topology=topology,primary=primary)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit','compare']);args=parser.parse_args()
    audit() if args.stage=='audit' else compare()
