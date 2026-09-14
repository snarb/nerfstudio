"""Compare three train-reference hole domains on one unchanged world quadric.

Positive-only annotations are an explicit diagnostic upper bound, not an
accepted semantic policy. This program edits no mesh, mask or prediction.
"""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from study_confidence_depth_prior import unproject,project_integer,raycast_integer
from forearm_quadric_rays import world_quadric,world_plane,intersect_near_plane
from ordered_forearm_admission import point_votes
from diffusion_mesh_repair import scene_for
import study_forearm_plane_transfer_v3 as prior

BASE=Path('/mnt/data/dec5_forearm_admission_quadric_bounded')
OUT=Path('/mnt/data/dec5_forearm_reference_coverage_probe')


def run(output,frame):
    prior.configure();v1=prior.v2.v1;root=prior.OUT/frame
    rows,depths,hashes=v1.load_real(frame);analysis=read(root/'analysis.json')
    if hashes!=analysis['source_depth_sha256']:raise ValueError('Changed observed maps')
    source=BASE/frame;geometry=read(source/'geometry_result.json')
    if sha(source/'guarded.ply')!=geometry['hashes']['guarded.ply']:raise ValueError('Changed starting mesh')
    fitpath=Path('/mnt/data/dec5_forearm_multiview_anchors')/frame/'result.json'
    fit=next(r for r in read(fitpath)['fit'] if r['model']=='quadratic')
    reference=next(r for r in rows if r['physical_camera']==v1.NAMES[0])
    quadric=world_quadric(reference,fit);plane=world_plane(reference,analysis['plane_inverse_coefficients'])
    folder=output/frame;folder.mkdir(parents=True,exist_ok=True)
    request=dict(frame=frame,mesh_sha256=sha(source/'guarded.ply'),mesh=str(source/'guarded.ply'),
        fit_result_sha256=sha(fitpath),diagnostic_sha256=sha(root/'diagnostic.npz'),depth_sha256=hashes,
        scripts={n:sha(Path(__file__).with_name(n)) for n in [Path(__file__).name,'forearm_quadric_rays.py','ordered_forearm_admission.py']},
        train_references=v1.NAMES,shared_shape_not_three_refits=True,max_original_reference_depth_change=.01,
        max_distance_from_trusted_pixel=100,positive_only_is_diagnostic_not_acceptance=True,mesh_changed=False,heldout_used=False)
    if (folder/'request.json').exists() and read(folder/'request.json')!=request:raise ValueError('Frozen probe mismatch')
    atomic_json(folder/'request.json',request)
    mesh=o3d.io.read_triangle_mesh(str(source/'guarded.ply'));scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
    masks=v1.masks(frame);data=np.load(root/'diagnostic.npz');results=[]
    for name in v1.NAMES:
        camera=next(r for r in rows if r['physical_camera']==name);old=raycast_integer(scene,camera)
        y,x=np.nonzero(masks[name]&(old==0));center=np.asarray(camera['transform_matrix'])[:3,3]
        directions=unproject(camera,x,y,np.ones(len(x)))-center
        t,plane_t=intersect_near_plane(center,directions,quadric,plane)
        good=np.isfinite(t)&(t>0);x,y,t=x[good],y[good],t[good];points=center+t[:,None]*directions[good]
        uv,z=project_integer(reference,points);inverse=np.column_stack([uv/100,np.ones(len(uv))])@analysis['plane_inverse_coefficients']
        bounded=(inverse>0)&(np.abs(z-1/np.maximum(inverse,1e-12))<=.01)
        votes,negative,free=point_votes(points,rows,v1.NAMES,masks,data,depths,prior.v2.semantic_domain)
        distance=distance_transform_edt(~data[name+'_trusted'])[y,x]
        positive=(votes>=2)&(free==0)&bounded&(distance<=100);strict=positive&(negative==0)
        rgb=np.array(Image.open(root/'rgb'/(name+'.png')));overlay=rgb.copy()
        overlay[y[positive&~strict],x[positive&~strict]]=[255,60,60]
        overlay[y[strict],x[strict]]=[0,255,100]
        panel=Image.new('RGB',(860,444));draw=ImageDraw.Draw(panel)
        for i,im in enumerate([rgb,overlay]):panel.paste(Image.fromarray(np.rot90(im)).crop((0,1500,430,1920)),(i*430,24))
        draw.text((2,3),name+' GT / green=strict; red=positive-only',fill='white')
        panel.save(folder/(name+'.png'))
        np.savez_compressed(folder/(name+'.npz'),xy=np.column_stack([x,y]),points=points,depth=t,
            strict=strict,positive_only=positive,support=votes,negative=negative,free=free,bounded=bounded,distance=distance)
        results.append(dict(camera=name,current_skin_misses=int((masks[name]&(old==0)).sum()),
            valid_quadric_intersections=len(points),strict_candidates=int(strict.sum()),positive_only_candidates=int(positive.sum()),
            old_annotation_disagreement_only=int((positive&~strict).sum()),
            panel_sha256=sha(folder/(name+'.png')),arrays_sha256=sha(folder/(name+'.npz'))))
    atomic_json(folder/'result.json',dict(request_sha256=sha(folder/'request.json'),records=results,
        geometry_changed=False,visual_status='requires_actual_review',point_counts_not_recovered_render_pixels=True))
    print(frame,results,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=['001029','001033','001037'],required=True)
    p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();run(a.output,a.frame)
