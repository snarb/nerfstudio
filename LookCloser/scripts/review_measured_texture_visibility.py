"""Matched review and actual-source replay for measured-depth RGB admission."""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from study_measured_texture_visibility import ROOT,FRAME,DEPTH_ROOT
from study_confidence_depth_prior import load_real,project_integer
from review_measured_free_surface import VIEWS,baseline
from review_subface_free_space import BOXES,HEADS
from review_jaw_repair_transfer import verified_image,panel
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth


def independent_far(depth,uv,z):
    xy=np.rint(uv).astype(int);x,y=xy[:,0],xy[:,1]
    inside=np.isfinite(uv).all(1)&np.isfinite(z)&(z>0)&(x>=2)&(x<1918)&(y>=2)&(y<1078)
    ids=np.flatnonzero(inside);result=np.zeros(len(z),bool)
    taps=np.array([depth[y[ids]+dy,x[ids]+dx] for dy in range(-2,3) for dx in range(-2,3)])
    positive=np.isfinite(taps)&(taps>0);ordered=np.sort(np.where(positive,taps,np.inf),axis=0)
    with np.errstate(invalid='ignore'):
        stable=np.isfinite(ordered[19])&(ordered[4]>0)&((ordered[19]-ordered[4])<=.005*ordered[4])
    result[ids]=stable&((positive&(taps>z[ids]+np.maximum(.005,.01*z[ids]))).sum(0)>=20)
    return result


def review(view):
    old=baseline(FRAME,view);new=ROOT/FRAME/view
    a,ar=verified_image(old,FRAME);b,br=verified_image(new,FRAME)
    aq,bq=read(old/'request.json'),read(new/'request.json')
    for key in ['mesh_sha256','camera','source_cameras','fixed_exposure']:assert ar[key]==br[key]
    for key in ['recipe','profiles_sha256','exposure_sha256','calibration_sha256']:assert aq[key]==bq[key]
    guard=read(new/'guard_audit.json')
    assert guard['request_sha256']==sha(new/'request.json')
    assert guard['frame_complete_sha256']==sha(new/'frames'/FRAME/'complete.json')
    ad=np.load(old/'frames'/FRAME/'target_depth.npz')['depth'];bd=np.load(new/'frames'/FRAME/'target_depth.npz')['depth']
    np.testing.assert_array_equal(ad,bd)
    row=bq['inventory'][0];mesh=o3d.io.read_triangle_mesh(row['mesh'])
    vertices=np.asarray(mesh.vertices,np.float32);triangles=np.asarray(mesh.triangles,np.uint32)
    d,ids,bary=camera_depth(scene_for(vertices,triangles),row['camera']);hit=np.isfinite(d)
    np.testing.assert_array_equal(np.where(hit,d,0),bd)
    pixels=np.flatnonzero(hit);face=ids[hit];weights=np.column_stack((1-bary[hit].sum(1),bary[hit]))
    points=(vertices[triangles[face]]*weights[:,:,None]).sum(1)
    sources=[np.array(Image.open(root/'frames'/FRAME/'source_ids.png')).ravel()[pixels] for root in [old,new]]
    rows,depths,receipt=load_real(DEPTH_ROOT,FRAME)
    assert receipt==bq['measured_texture_visibility']['depth_receipt']
    assert [r['physical_camera'] for r in rows]==br['source_cameras']
    bad=np.zeros((2,len(pixels)),bool)
    for ci,(camera,depth) in enumerate(zip(rows,depths)):
        for arm,chosen in enumerate(sources):
            selected=np.flatnonzero(chosen==ci)
            uv,z=project_integer(camera,points[selected]);bad[arm,selected]=independent_far(depth,uv,z)
    assert not bad[1].any(),'Chosen RGB source contradicts measured guard'
    from diagnose_lipstick_fin_depth import POLYGON
    polygon=Image.new('L',(1080,1920));ImageDraw.Draw(polygon).polygon(POLYGON,fill=1)
    diagnostic=np.rot90(np.asarray(polygon,bool),-1).ravel()[pixels] if view=='K004_B005_1210DS' else np.zeros(len(pixels),bool)
    dest=new/'review';dest.mkdir(exist_ok=False)
    images=[a,b];names=['original self-mesh visibility','+ measured farther-layer guard']
    bindings={}
    if view!='moving':
        gt=DEPTH_ROOT/FRAME/'review'/view/'train_gt.png'
        images=[np.array(Image.open(gt)),*images];names=['actual train GT',*names];bindings[str(gt)]=sha(gt)
    panel(dest/'lipstick_native.png',images,names,BOXES[view])
    panel(dest/'head_native.png',images,names,HEADS[view])
    newblack=(a.max(2)>0)&(b.max(2)==0)
    for name,box in [('lipstick',BOXES[view]),('head',HEADS[view])]:
        overlays=[]
        for im in [a,b]:
            marked=im.copy();marked[newblack]=[255,0,255];overlays.append(marked)
        panel(dest/(name+'_new_black.png'),overlays,['original / magenta = newly black','candidate / magenta = newly black'],box)
    for root in [old,new]:
        complete=read(root/'frames'/FRAME/'complete.json')
        bindings[str(root/'request.json')]=sha(root/'request.json')
        bindings[str(root/'frames'/FRAME/'complete.json')]=sha(root/'frames'/FRAME/'complete.json')
        bindings.update({str(root/'frames'/FRAME/n):h for n,h in complete['hashes'].items()})
    np.savez_compressed(dest/'source_replay.npz',target_pixels=pixels,actual_sources=np.stack(sources),
        contradicts_measured_far=bad,diagnostic_polygon=diagnostic)
    result=dict(view=view,bindings=bindings,mesh_and_target_depth_exact=True,
        actual_colored_sources_checked=[int((s<62).sum()) for s in sources],
        actual_sources_contradicting_stable_far=[int(r.sum()) for r in bad],
        diagnostic_pixels=int(diagnostic.sum()),diagnostic_contradicting_sources=[int((r&diagnostic).sum()) for r in bad],
        diagnostic_source_255=[int(((s==255)&diagnostic).sum()) for s in sources],
        changed_rgb=int(np.any(a!=b,2).sum()),new_black_rgb=int(newblack.sum()),
        newly_colored_rgb=int(((a.max(2)==0)&(b.max(2)>0)).sum()),
        guard_audit_sha256=sha(new/'guard_audit.json'),independent_far_footprint_arithmetic=True,
        query_world_points_recreated_from_same_float32_barycentrics=True,
        hashes={p.name:sha(p) for p in dest.iterdir() if p.is_file()},script_sha256=sha(__file__),
        visual_status='pending',production_promoted=False,counts_not_quality_metrics=True)
    atomic_json(dest/'result.json',result)
    print({k:v for k,v in result.items() if k not in ['bindings','hashes']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--view',choices=VIEWS,required=True)
    review(p.parse_args().view)
