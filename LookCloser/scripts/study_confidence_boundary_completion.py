"""Opt-in measured-pin assembly control on three already diagnosed forearms."""
from pathlib import Path
from copy import deepcopy
import argparse,time
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from study_confidence_depth_prior import unproject
from confidence_boundary_completion import solve_depth,grid_faces
from curve_forearm_delta import semantic_faces
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
import study_forearm_plane_transfer_v3 as prior

PARENT=Path('/mnt/data/dec5_phase30_early_texture_dynamic_150')
BASE=Path('/mnt/data/dec5_forearm_color_qualified_curved')
OUT=Path('/mnt/data/dec5_forearm_measured_boundary')

def prepare(output,frame,annotation_domain=False):
    prior.configure();v1=prior.v2.v1;root=prior.OUT/frame;out=output/frame;out.mkdir(parents=True,exist_ok=True)
    source=next(r for r in read(PARENT/'request.json')['inventory'] if r['frame_id']==frame)
    ev=np.load(root/'plane/evidence.npz');data=np.load(root/'diagnostic.npz');accepted=ev['accepted']
    md=data[v1.NAMES[0]+'_mesh'];trusted=data[v1.NAMES[0]+'_trusted'];domain=binary_dilation(accepted)&((md>0)|accepted)
    pins=domain&~accepted&trusted
    rows,depths,hashes=v1.load_real(frame);ref=next(r for r in rows if r['physical_camera']==v1.NAMES[0])
    anchors=Path('/mnt/data/dec5_forearm_multiview_anchors')/frame;fit_result=read(anchors/'result.json')
    if sha(anchors/'anchors.npz')!=fit_result['anchors_sha256']:raise ValueError('Changed observed anchors')
    fit=next(r for r in fit_result['fit'] if r['model']=='quadratic')
    request=dict(frame=frame,source_mesh=source['mesh'],source_mesh_sha256=sha(source['mesh']),
        parent_request_sha256=sha(PARENT/'request.json'),source_depth_sha256=hashes,
        prior_evidence_sha256=sha(root/'plane/evidence.npz'),diagnostic_sha256=sha(root/'diagnostic.npz'),
        observed_guard=dict(kind='depth_and_color_witnesses',min_color_witnesses=3,chroma_mean_abs_limit=.04),
        fit_result_sha256=sha(anchors/'result.json'),parameters=dict(residual_regularization=.05,max_displacement=.012,
        measured_pins_only=True,semantic_masks_unchanged=True,quadric_not_measured_anatomy=True),
        scripts={n:sha(Path(__file__).with_name(n)) for n in [Path(__file__).name,'confidence_boundary_completion.py',
            'curve_forearm_delta.py','photometric_forearm_depth_guard.py','diagnose_forearm_color_witnesses.py']})
    if annotation_domain:
        request['parameters']['known_annotation_margin']=3
        request['scripts']['annotation_mask_domain.py']=sha(Path(__file__).with_name('annotation_mask_domain.py'))
    if request['source_mesh_sha256']!=source['mesh_sha256']:raise ValueError('Changed production geometry')
    if (out/'request.json').exists() and read(out/'request.json')!=request:raise ValueError('Changed frozen request')
    atomic_json(out/'request.json',request)
    if (out/'geometry_result.json').exists():
        r=read(out/'geometry_result.json');assert r['request_sha256']==sha(out/'request.json')
        for p,h in r['hashes'].items():assert sha(out/p)==h
        return
    y,x=np.nonzero(domain);q=(np.column_stack([x,y])-fit['reference_center'])/100
    design=np.column_stack([q,np.ones(len(q)),q[:,0]**2,q[:,0]*q[:,1],q[:,1]**2])
    inverse=design@np.array(fit['all_camera_coefficients']);model=np.zeros(md.shape);model[y,x]=1/inverse
    solved,stats=solve_depth(domain,model,md,pins)
    raw=np.where(accepted,ev['depth'],md);delta=solved[domain]-raw[domain]
    if np.max(np.abs(delta))>.012:raise ValueError('Completion exceeds frozen displacement bound')
    original=o3d.io.read_triangle_mesh(source['mesh']);ov,ot=np.asarray(original.vertices),np.asarray(original.triangles)
    nv=unproject(ref,x,y,solved[y,x]);vertices=np.concatenate([ov,nv]);index=np.full(md.shape,-1,int);index[y,x]=np.arange(len(x))+len(ov)
    faces=grid_faces(domain,accepted,index);initial_count=len(faces)
    select=semantic_faces
    if annotation_domain:
        from annotation_mask_domain import semantic_faces as select
    faces,semantic=select(vertices,faces,rows,v1.masks(frame),axis_extent=True)
    triangles=np.concatenate([ot,faces]);stats.update(semantic,unfiltered_grid_triangles=initial_count,
        all_boundary_pixels=int((domain&~accepted).sum()),depth_change_quantiles=np.quantile(delta,[0,.5,1]).tolist())
    def save(name,tt):
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(tt));mesh.compute_vertex_normals()
        if not o3d.io.write_triangle_mesh(str(out/name),mesh):raise IOError('Mesh write failed')
    save('transferred.ply',triangles)
    from photometric_forearm_depth_guard import make_guard
    veto,calls,provenance=make_guard(frame,rows,depths);rounds=[]
    for iteration in range(8):
        scene=scene_for(vertices,triangles);remove=set();checks=[]
        for ci,(camera,depth) in enumerate(zip(rows,depths)):
            for offset in [0,.5]:
                ids,count,raw_count=veto(scene,camera,depth,rows,depths,len(ot),len(triangles),offset)
                remove.update(ids.tolist());checks.append(dict(camera=camera['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw_count))
            if (ci+1)%10==0:
                atomic_json(out/'progress.json',dict(stage='color_qualified_depth_guard',iteration=iteration,cameras=ci+1,flagged=len(remove),unix_time=time.time()))
                print(frame,iteration,ci+1,len(remove),flush=True)
        rounds.append(dict(iteration=iteration,removed_triangles=len(remove),checks=checks))
        if not remove:break
        keep=np.ones(len(triangles),bool);keep[list(remove)]=False
        if not keep[:len(ot)].all():raise ValueError('Original geometry removal attempted')
        triangles=triangles[keep]
    save('guarded.ply',triangles);passed=not rounds[-1]['removed_triangles']
    saved=o3d.io.read_triangle_mesh(str(out/'guarded.ply'))
    assert np.array_equal(np.asarray(saved.vertices)[:len(ov)],ov) and np.array_equal(np.asarray(saved.triangles)[:len(ot)],ot)
    panel=Image.new('RGB',(1290,495));draw=ImageDraw.Draw(panel)
    for i,(label,path) in enumerate([('previous',BASE/frame/'guarded.ply'),('measured pins / raw',out/'transferred.ply'),('measured pins / guarded',out/'guarded.ply')]):
        mesh=o3d.io.read_triangle_mesh(str(path));mesh.compute_triangle_normals();scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
        d,ids,_=camera_depth(scene,source['camera']);valid=np.isfinite(d);im=np.zeros((*ids.shape,3),np.uint8)
        light=np.abs(np.asarray(mesh.triangle_normals)@np.array([.3,.4,.866]));im[valid]=(60+170*light[ids[valid],None]).astype(np.uint8)
        im[valid&(ids>=len(ot))]=[240,60,50];panel.paste(Image.fromarray(np.rot90(im)).crop((0,1450,430,1920)),(i*430,25));draw.text((i*430+3,4),label,fill='white')
    panel.save(out/'moving_forearm_clay_native.png')
    np.savez_compressed(out/'evidence.npz',domain=domain,accepted=accepted,measured_pins=pins,depth=solved)
    atomic_json(out/'geometry_result.json',dict(request_sha256=sha(out/'request.json'),stats=stats,rounds=rounds,
        observed_guard_passed=passed,final_added_triangles=len(triangles)-len(ot),depth_hashes=hashes,
        color_guard_calls=calls,color_guard_provenance=provenance,original_mesh_prefix_exact=True,
        hashes={n:sha(out/n) for n in ['transferred.ply','guarded.ply','evidence.npz','moving_forearm_clay_native.png']},
        visual_status='pending',production_accepted=False))
    print(frame,'finished',stats,passed,len(triangles)-len(ot),flush=True)

def render(output,frame):
    import render_smooth_temporal_mesh_video as renderer
    from study_early_texture_prior import install
    install();renderer.torch.set_num_threads(2);prior.configure()
    result=read(output/frame/'geometry_result.json');assert result['observed_guard_passed']
    for p,h in result['hashes'].items():assert sha(output/frame/p)==h
    parent=renderer.verify_request(PARENT);rows,_,_=prior.v2.v1.cameras(frame)
    for view in ['moving','H004_A005_1210M6']:
        dest=output/'rgb'/frame/view/'guarded';request=deepcopy(parent)
        request['inventory']=[r for r in request['inventory'] if r['frame_id']==frame];row=request['inventory'][0]
        row.update(mesh=str(output/frame/'guarded.ply'),mesh_sha256=sha(output/frame/'guarded.ply'))
        if view!='moving':row['camera']=next(r for r in rows if r['physical_camera']==view)
        request.update(partial_diagnostic_only=True,full_video_candidate=False,geometry_result_sha256=sha(output/frame/'geometry_result.json'))
        request['script_hashes'][Path(__file__).name]=sha(__file__);dest.mkdir(parents=True,exist_ok=True);(dest/'frames').mkdir(exist_ok=True)
        if (dest/'request.json').exists() and read(dest/'request.json')!=request:raise ValueError('Changed render request')
        atomic_json(dest/'request.json',request);renderer.render(dest,[frame])
        baseline=output/'rgb'/frame/view/'baseline';src=Path('/mnt/data/dec5_forearm_early_texture_prior')/'rgb'/frame/view/'baseline'
        if baseline.exists():assert baseline.is_symlink() and baseline.resolve()==src.resolve()
        else:baseline.symlink_to(src,target_is_directory=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render']);p.add_argument('--frame',required=True,choices=['001029','001033','001037'])
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--annotation-domain',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare(a.output,a.frame,a.annotation_domain)
    else:render(a.output,a.frame)
