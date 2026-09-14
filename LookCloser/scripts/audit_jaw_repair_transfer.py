"""Independent final ray/topology audit of the bounded jaw transfer controls."""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from study_jaw_repair_transfer import OUT,PARENT,mask_votes
from study_jaw_boundary_notches import propose,SETTINGS
from study_confidence_depth_prior import load_real
from guard_jaw_measured_depth import measured_pixel_veto,initial_admission
from diffusion_mesh_repair import scene_for


def topology(mesh):
    t=np.asarray(mesh.triangles)
    _,counts=np.unique(np.sort(t[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    labels,_,_=mesh.cluster_connected_triangles()
    return dict(components=len(np.unique(labels)),nonmanifold_edges=int((counts>2).sum()))


def run(output,frame):
    folder=output/frame;request=read(folder/'request.json');result=read(folder/'result.json')
    if result['request_sha256']!=sha(folder/'request.json'):raise ValueError('Changed request')
    for name,h in request['scripts'].items():
        if sha(Path(__file__).with_name(name))!=h:raise ValueError('Changed producer dependency')
    for name,h in result['hashes'].items():
        if sha(folder/name)!=h:raise ValueError('Changed geometry output')
    if sha(request['source_mesh'])!=request['source_mesh_sha256']:raise ValueError('Changed old mesh')
    old=o3d.io.read_triangle_mesh(request['source_mesh']);new=o3d.io.read_triangle_mesh(str(folder/'mesh.ply'))
    v,t=np.asarray(old.vertices),np.asarray(old.triangles);nt=np.asarray(new.triangles)
    if not np.array_equal(v,np.asarray(new.vertices)) or not np.array_equal(t,nt[:len(t)]):raise ValueError('Changed prefix')
    raw,_=propose(v,t,SETTINGS);a=np.load(folder/'evidence.npz')
    if not np.array_equal(raw[len(t):],a['proposals']):raise ValueError('Changed raw proposal recipe')
    if not np.array_equal(nt[len(t):],a['proposals'][a['retained_proposal_ids']]):raise ValueError('Invalid retained mapping')
    initial=initial_admission(a['votes'],a['free'],a['mask_support'],a['mask_outside'])
    if not initial[a['retained_proposal_ids']].all():raise ValueError('Retained unqualified samples')
    before,after=topology(old),topology(new)
    if after['components']>before['components'] or after['nonmanifold_edges']>before['nonmanifold_edges']:
        raise ValueError('New island or nonmanifold edge')
    rows,depths,receipt=load_real(Path(request['depth_root']),frame)
    if receipt!=request['depth_receipt']:raise ValueError('Changed observed depth')
    source=next(r for r in read(PARENT/'request.json')['inventory'] if r['frame_id']==frame)
    maskroot=Path(source['source_masks']['root']);masks=np.load(maskroot/'masks.npz')['masks'];names=read(maskroot/'cameras.json')
    if sha(maskroot/'masks.npz')!=request['source_mask_sha256']:raise ValueError('Changed original masks')
    if 'mask_override' in request:
        from build_measured_foreground_override import add_certified_seeds
        override=Path(request['mask_override']['root'])/frame;mr=read(override/'result.json')
        if sha(override/'result.json')!=request['mask_override']['result_sha256']:raise ValueError('Changed override result')
        if mr['request_sha256']!=sha(override/'request.json'):raise ValueError('Changed override request')
        for name,h in mr['hashes'].items():
            if sha(override/name)!=h:raise ValueError('Changed override evidence')
        seeds=np.load(override/'evidence.npz')['seeds'];index=names.index(mr['camera'])
        expected=add_certified_seeds(masks[index],seeds);actual=np.load(override/'mask.npy')
        if not np.array_equal(expected,actual):raise ValueError('Changed mask expansion rule')
        masks=masks.copy();masks[index]=actual
    ms,mo=mask_votes(v,a['proposals'],rows,masks,names)
    if not np.array_equal(ms,a['mask_support']) or not np.array_equal(mo,a['mask_outside']):raise ValueError('Semantic gate replay mismatch')
    scene=scene_for(v,nt);checks=[]
    for camera,depth in zip(rows,depths):
        for offset in [0,.5]:
            ids,count,raw_count=measured_pixel_veto(scene,camera,depth,rows,depths,len(t),len(nt),offset)
            if len(ids) or count:raise ValueError('Final measured ray contradiction')
            checks.append(dict(camera=camera['physical_camera'],offset=offset,qualified_veto_pixels=count))
    atomic_json(folder/'independent_audit.json',dict(result_sha256=sha(folder/'result.json'),
        original_prefix_exact=True,proposal_replay_exact=True,retained_mapping_exact=True,semantic_replay_exact=True,
        topology_before=before,topology_after=after,checks=checks,script_sha256=sha(__file__),production_accepted=False))
    print(frame,'124 fresh rays pass; topology',before,'->',after,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--frame',required=True);a=p.parse_args();run(a.output,a.frame)
