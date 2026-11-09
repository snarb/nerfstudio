#!/usr/bin/env python3
"""Export auditable observed-train-ray free-interval candidates for six AABBs.

No model is loaded. No carving, masks, occupancy admission or training supervision
is applied. Coordinates are prepared_world; t is Euclidean distance along a unit
ray. Surface margin=max(.01, .05*camera-to-witness distance), identical for all
boxes. Opaque visible SfM tracks suggest free line segments, not whole voxels.
"""
from pathlib import Path
import argparse,json
import numpy as np
from probe_marshawn_occupancy import file_sha


def project(points, pose, focal, principal):
    local=(np.asarray(points)-np.asarray(pose)[:3,3])@np.asarray(pose)[:3,:3]
    depth=-local[:,2]
    return np.column_stack([focal[0]*local[:,0]/depth+principal[0],-focal[1]*local[:,1]/depth+principal[1]]),depth


def clip_intervals(origins,directions,witness_depth,bounds,near=.01,min_margin=.01,relative_margin=.05):
    origins,directions=np.asarray(origins,float),np.asarray(directions,float)
    distances=np.asarray(witness_depth,float);box=np.asarray(bounds,float)
    if origins.shape!=directions.shape or origins.ndim!=2 or origins.shape[1]!=3 or distances.shape!=(len(origins),):raise ValueError('Invalid ray arrays')
    if box.shape!=(2,3) or not np.isfinite(box).all() or (box[1]<=box[0]).any():raise ValueError('Invalid AABB')
    if not np.isfinite([near,min_margin,relative_margin]).all() or near<0 or min_margin<=0 or not 0<relative_margin<1:raise ValueError('Invalid physical margin policy')
    valid=np.isfinite(origins).all(1)&np.isfinite(directions).all(1)&np.isfinite(distances)&(distances>0)&np.isclose(np.linalg.norm(directions,axis=1),1,rtol=0,atol=1e-7)
    entry=np.full(len(origins),near);exit=np.full(len(origins),np.inf)
    for axis in range(3):
        parallel=directions[:,axis]==0
        valid&=~parallel|((origins[:,axis]>=box[0,axis])&(origins[:,axis]<box[1,axis]))
        d=np.where(parallel,1,directions[:,axis]);a=(box[0,axis]-origins[:,axis])/d;b=(box[1,axis]-origins[:,axis])/d
        entry=np.maximum(entry,np.where(parallel,-np.inf,np.minimum(a,b)))
        exit=np.minimum(exit,np.where(parallel,np.inf,np.maximum(a,b)))
    margin=np.maximum(min_margin,relative_margin*distances)
    end=np.minimum(exit,distances-margin);valid&=np.isfinite(entry)&np.isfinite(end)&(end>entry)
    return entry,end,margin,valid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','observations','witnesses','tracks','boxes','seed-manifest','rig','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();metadata=json.loads((a.data/'transforms.json').read_text());preparation=json.loads((a.data/'preparation.json').read_text());boxes=json.loads(a.boxes.read_text());seed=json.loads(a.seed_manifest.read_text());tracks=json.loads(a.tracks.read_text())
    obs=np.load(a.observations);witness=np.load(a.witnesses);xyz=witness['xyz'];names=obs['camera_names'].tolist()
    if boxes.get('coordinate_system')!='prepared_world' or not boxes.get('approved'):raise ValueError('Approved prepared_world boxes required')
    binding={'transforms_sha256':file_sha(a.data/'transforms.json'),'normalization_sha256':file_sha(a.data/'preparation.json'),'points_sha256':file_sha(a.witnesses),'tracks_sha256':file_sha(a.tracks),'observations_sha256':file_sha(a.observations),'boxes_sha256':file_sha(a.boxes)}
    if any(seed.get(k)!=v for k,v in binding.items()):raise ValueError('Seed provenance mismatch')
    if boxes.get('normalization_sha256')!=binding['normalization_sha256']:raise ValueError('Bounds normalization mismatch')
    calibration={name:file_sha(a.rig/name) for name in ['cameras.txt','images.txt']}
    if calibration!=preparation['rig_sha256'] or calibration!=seed['calibration_sha256']:raise ValueError('Calibration files changed')
    if names!=witness['cameras'].tolist() or tracks['points']!=len(xyz):raise ValueError('Witness camera/track domain mismatch')
    point_idx=obs['point_indices'];camera_idx=obs['camera_indices'];uv=obs['uv']
    if any(len(set(c))<3 for c in tracks['camera_ids']):raise ValueError('Each trusted point must have at least three observed cameras')
    if point_idx.min()<0 or point_idx.max()>=len(xyz) or camera_idx.min()<0 or camera_idx.max()>=len(names):raise ValueError('Invalid observation indices')
    actual={(i,c) for i,cameras in enumerate(tracks['camera_ids']) for c in cameras};provided=list(zip(point_idx.tolist(),camera_idx.tolist()))
    if len(set(provided))!=len(provided) or set(provided)!=actual:raise ValueError('Observations must match actual unique track-camera graph exactly')
    train=set(metadata['train_filenames']);evaluation=set(metadata['val_filenames'])|set(metadata['test_filenames'])
    frames={f['physical_camera']:f for f in metadata['frames'] if f['file_path'] in train}
    if set(names)!=set(frames) or train&evaluation:raise ValueError('Train camera scope mismatch or eval overlap')
    origins=np.empty((len(point_idx),3));projection_error=np.empty(len(point_idx));depths=np.empty(len(point_idx));image_hashes={}
    for camera,name in enumerate(names):
        frame=frames[name]
        if any(abs(frame.get(k,0))>1e-12 for k in ['k1','k2','k3','k4','p1','p2']):raise ValueError('Zero-distortion prepared calibration required')
        pose=np.array(frame['transform_matrix']);ids=np.flatnonzero(camera_idx==camera);origins[ids]=pose[:3,3]
        predicted,z=project(xyz[point_idx[ids]],pose,[frame['fl_x'],frame['fl_y']],[frame['cx'],frame['cy']]);projection_error[ids]=np.linalg.norm(predicted-uv[ids],axis=1);depths[ids]=z
        if (z<=0).any() or ((predicted<0)|(predicted>=np.array([frame['w'],frame['h']]))).any():raise ValueError('Invalid visible train observation projection')
        image_hashes[name]={'file':frame['file_path'],'sha256':file_sha(a.data/frame['file_path'])}
    if not np.isfinite(projection_error).all() or projection_error.max()>1e-4:raise ValueError('Observation projection/normalization mismatch')
    offset=xyz[point_idx]-origins;distance=np.linalg.norm(offset,axis=1);direction=offset/distance[:,None]
    a.output.mkdir(parents=True,exist_ok=False);variants=[]
    for scope in ['scene','subject']:
        for size in ['tight','medium','loose']:
            box=np.array(boxes['scopes'][scope][size]);start,end,margin,valid=clip_intervals(origins,direction,distance,box)
            accepted=np.flatnonzero(valid);name=scope+'_'+size;path=a.output/(name+'.npz');inside=((xyz[point_idx]>=box[0])&(xyz[point_idx]<box[1])).all(1)
            np.savez_compressed(path,origins=origins[valid],directions=direction[valid],t_start=start[valid],t_end=end[valid],witness_depth=distance[valid],surface_margin=margin[valid],point_idx=point_idx[valid],camera_idx=camera_idx[valid],observation_idx=accepted,uv=uv[valid],witness_xyz=xyz[point_idx[valid]],aabb=box,camera_names=np.array(names))
            variants.append(dict(name=name,file=path.name,sha256=file_sha(path),aabb=box.tolist(),accepted=int(valid.sum()),excluded_invalid_or_empty=int((~valid).sum()),unique_points=len(np.unique(point_idx[valid])),cameras=len(np.unique(camera_idx[valid])),accepted_witnesses_outside_box=int((valid&~inside).sum()),interval_length_median=float(np.median(end[valid]-start[valid])),interval_length_total=float((end[valid]-start[valid]).sum())))
    report=dict(schema_version=1,kind='observed_train_sfm_free_ray_interval_candidates',coordinate_system='prepared_world',t_units='Euclidean prepared-world distance; directions are unit length',policy=dict(near=.01,minimum_surface_margin=.01,relative_surface_margin=.05,same_physical_policy_all_boxes=True),bindings=binding,calibration_sha256=calibration,seed_manifest_sha256=file_sha(a.seed_manifest),images=image_hashes,script_sha256=file_sha(__file__),source_observations=len(point_idx),source_points=len(xyz),source_cameras=len(names),projection_error_max_pixels=float(projection_error.max()),variants=variants,uses_eval_rgb=False,uses_masks=False,applied_to_training=False,applied_to_occupancy=False,limitations=['Free line-of-sight candidates from actual train SfM track-camera graph, not independent geometry ground truth.','No whole-voxel carving or mask supervision; a ray does not establish an entire cell is empty.','Witnesses outside a particular box can still supply valid clipped free intervals through that box.','Transparency, moving-scene synchronization, occlusion or incorrect feature tracks can violate presumed free intervals.','Reprojection here verifies data consistency with fitted calibration; it does not independently establish solve accuracy.'])
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'variants':variants,'projection_error_max_pixels':report['projection_error_max_pixels']}),flush=True)


if __name__=='__main__':main()
