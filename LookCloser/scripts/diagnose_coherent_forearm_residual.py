"""Attribute remaining fixed-train-ROI misses before/after native carving."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from study_confidence_depth_prior import unproject,project_integer
from forearm_quadric_rays import world_quadric,world_plane,intersect_near_plane
from ordered_forearm_admission import point_votes
from scipy.ndimage import distance_transform_edt
import study_forearm_plane_transfer_v3 as prior


def run(root,frame):
    prior.configure();v1=prior.v2.v1;rows,depths,hashes=v1.load_real(frame)
    folder=root/frame;request=read(folder/'request.json');result=read(folder/'geometry_result.json')
    if hashes!=request['source_depth_sha256'] or result['request_sha256']!=sha(folder/'request.json'):raise ValueError('Changed geometry inputs')
    camera=request['reference_camera'];name=camera['physical_camera'];masks=v1.masks(frame);mask=masks[name]
    maps={}
    for label,path in [('previous',Path(request['source_mesh'])),('raw',folder/'transferred.ply'),('guarded',folder/'guarded.ply')]:
        mesh=o3d.io.read_triangle_mesh(str(path));d,_,_=camera_depth(scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles)),camera);maps[label]=d
    predroot=root/'rgb'/frame/name/'guarded';render=predroot/'frames'/frame
    receipt=read(render/'complete.json')
    if receipt['request_sha256']!=sha(predroot/'request.json'):raise ValueError('Changed render request')
    for n,h in receipt['hashes'].items():
        if sha(render/n)!=h:raise ValueError('Changed rendered output')
    # The production wrapper also masks a target if it is a physical TRAIN
    # camera. Never mislabel that target-mask clipping as missing mesh geometry.
    spec=read(predroot/'request.json')['inventory'][0]['source_masks'];maskroot=Path(spec['root'])
    for n,k in [('complete.json','complete_sha256'),('masks.npz','masks_sha256'),('cameras.json','cameras_sha256')]:
        if sha(maskroot/n)!=spec[k]:raise ValueError('Changed production source masks')
    names=read(maskroot/'cameras.json');targetmask=np.load(maskroot/'masks.npz')['masks'][names.index(name)]>0
    saved=np.load(render/'target_depth.npz')['depth'];geometry_valid=np.isfinite(maps['guarded']);valid=geometry_valid&targetmask
    if not np.array_equal(valid,saved>0) or not np.allclose(maps['guarded'][valid],saved[valid],atol=1e-7,rtol=0):raise ValueError('Ray convention differs from actual renderer')
    missing=mask&~valid;target_mask_only=missing&geometry_valid&~targetmask
    geometry_missing=missing&~geometry_valid
    preexisting=geometry_missing&~np.isfinite(maps['raw']);carved=geometry_missing&np.isfinite(maps['raw'])
    y,x=np.nonzero(preexisting);center=np.asarray(camera['transform_matrix'])[:3,3]
    directions=unproject(camera,x+.5,y+.5,np.ones(len(x)))-center
    reference=request['fit_reference_camera'];analysis=read(prior.OUT/frame/'analysis.json')
    fitpath=Path('/mnt/data/dec5_forearm_multiview_anchors')/frame/'result.json'
    if sha(fitpath)!=request['fit_sha256']:raise ValueError('Changed model fit')
    fit=next(r for r in read(fitpath)['fit'] if r['model']=='quadratic')
    z,_=intersect_near_plane(center,directions,world_quadric(reference,fit),world_plane(reference,analysis['plane_inverse_coefficients']))
    finite=np.isfinite(z)&(z>0);codes=np.zeros(mask.shape,np.uint8);codes[y,x]=1;codes[carved]=2;codes[target_mask_only]=9
    x,y,z=x[finite],y[finite],z[finite];points=unproject(camera,x+.5,y+.5,z)
    uv,rz=project_integer(reference,points);inverse=np.column_stack([uv/100,np.ones(len(uv))])@analysis['plane_inverse_coefficients']
    bounded=(inverse>0)&(np.abs(rz-1/np.maximum(inverse,1e-12))<=.01)
    data=np.load(prior.OUT/frame/'diagnostic.npz')
    votes,negative,free=point_votes(points,rows,v1.NAMES,masks,data,depths,prior.v2.semantic_domain)
    distance=distance_transform_edt(~data[name+'_trusted'])[y,x]
    codes[y,x]=3
    # Mutually exclusive ordered attribution; also retain overlapping gate counts.
    passed=np.ones(len(x),bool);labels={1:'no_quadric_intersection',2:'native_guard_removed',3:'grid_or_final_semantic_boundary',
        4:'model_displacement_bound',5:'fewer_than_two_positive_annotations',6:'available_annotation_disagreement',7:'initial_trusted_free_veto',8:'trusted_distance_bound',9:'physical_train_target_foreground_mask_only'}
    failures=[~bounded,votes<2,negative>0,free>0,distance>100]
    for code,bad in zip(range(4,9),failures):
        chosen=passed&bad;codes[y[chosen],x[chosen]]=code;passed&=~bad
    records={labels[k]:int((codes==k).sum()) for k in labels}
    if sum(records.values())!=missing.sum():raise ValueError('Attribution does not partition missing skin')
    gt=np.array(Image.open(prior.OUT/frame/'rgb'/(name+'.png')));overlay=gt.copy()
    colors={1:(255,255,255),2:(255,40,40),3:(255,255,0),4:(255,0,255),5:(0,120,255),6:(0,255,120),7:(255,120,0),8:(150,80,255),9:(0,255,255)}
    for k,c in colors.items():overlay[codes==k]=c
    panel=Image.new('RGB',(860,630));draw=ImageDraw.Draw(panel)
    for i,im in enumerate([gt,overlay]):panel.paste(Image.fromarray(np.rot90(im)).crop((0,1500,430,1920)),(i*430,25))
    draw.text((4,3),'Train GT / remaining depth-hole attribution',fill='white')
    for i,(k,label) in enumerate(labels.items()):draw.text((4,455+i*18),label+': '+str(records[label]),fill=colors[k])
    panel.save(folder/'residual_attribution.png');np.savez_compressed(folder/'residual_attribution.npz',codes=codes)
    atomic_json(folder/'residual_attribution.json',dict(frame=frame,script_sha256=sha(__file__),geometry_result_sha256=sha(folder/'geometry_result.json'),
        saved_render_depth_verified=True,integer_vs_half_pixel_boundary_counts_not_identical=True,
        fixed_train_roi=True,heldout_used=False,raw_geometry_missing_pixels_by_variant={k:int((mask&~np.isfinite(d)).sum()) for k,d in maps.items()},
        rendered_train_target_missing=int(missing.sum()),train_target_mask_policy_replayed=True,source_masks_spec=spec,
        exclusive_attribution=records,overlapping_gate_failure_counts={labels[k]:int(bad.sum()) for k,bad in zip(range(4,9),failures)},
        panel_sha256=sha(folder/'residual_attribution.png'),arrays_sha256=sha(folder/'residual_attribution.npz')))
    print(frame,records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',required=True)
    a=p.parse_args();run(a.root,a.frame)
