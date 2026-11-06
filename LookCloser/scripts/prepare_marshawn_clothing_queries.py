"""Freeze103 original TRAIN clothing diagnostic rays before A/B/C quality gates."""
import argparse,copy,json,hashlib
from pathlib import Path
import numpy as np
from audit_marshawn_tuning_checkpoint import ray_for_pixel

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(campaign,output):
    campaign,output=Path(campaign),Path(output);original=campaign/'independent_occupancy/queries/shirt_queries_with_depth.json';review=campaign/'expanded_clothing_geometry_v1/visibility_review.json';manifest=campaign/'expanded_clothing_geometry_v1/manifest.json';data=campaign/'data'
    request=copy.deepcopy(json.loads(original.read_text()));labels=json.loads(review.read_text());meta=json.loads(manifest.read_text())
    if sha(review)!=meta['bindings']['visibility_review_sha256']:raise ValueError('Sealed clothing source review changed')
    rows={Path(r['file_path']).stem.split('_')[-1]:r for r in json.loads((data/'transforms.json').read_text())['frames']}
    names=request['camera_order'];reserved={63,64,71,72}
    for p in request['pixels']:
        p['diagnostic_group']='old_local_reserved'if p['witness_id']in reserved else'duplicate_excluded'if p['witness_id']==66 else'old_prefix_supervised'
        p['actor']='black';p['region']='original_upper_back_lettering'
    for point in labels['points']:
        for source in point['reviewed_observations']:
            camera=source['camera'];x,y=np.floor(source['detected_uv']).astype(int)
            p=dict(camera=names.index(camera),camera_name=camera,x=int(x),y=int(y),witness_id=point['point'],
                   witness_xyz=point['xyz_floor'],actor=point['actor'],region=point['region'],
                   semantic='manually_reviewed_opaque_clothing_original_TRAIN_feature',
                   diagnostic_group='new_whole_patch_reserved'if point['status']=='reserved_whole_patch'else'new_prefix_supervised',
                   detected_uv=source['detected_uv'],projected_uv=source['projected_uv'])
            request['pixels'].append(p);request['rays'].append(ray_for_pixel(rows[camera],int(x),int(y)))
    keys=[(p['camera_name'],p['x'],p['y'])for p in request['pixels']]
    # Historical duplicate66 intentionally retains its predeclared audit identity;
    # new detections must not add duplicate source pixels.
    new=keys[28:]
    if len(set(new))!=len(new)or set(new)&set(keys[:28]):raise ValueError('New diagnostic source-pixel collision')
    groups={}
    for i,(p,ray)in enumerate(zip(request['pixels'],request['rays'])):
        groups.setdefault(p['diagnostic_group'],[]).append(i)
        origin,direction,xyz=map(np.asarray,[ray['origin'],ray['direction'],p['witness_xyz']]);depth=float((xyz-origin)@direction)
        p['query_witness_depth']=depth;p['point_to_pixel_ray_distance']=float(np.linalg.norm(xyz-origin-depth*direction))
        if depth<=0 or p['point_to_pixel_ray_distance']>.005:raise ValueError('Source detector ray does not qualify witness proximity')
    counts={k:len(v)for k,v in groups.items()}
    assert counts==dict(old_prefix_supervised=18,old_local_reserved=8,duplicate_excluded=2,new_prefix_supervised=66,new_whole_patch_reserved=9)
    request.update(kind='PREDECLARED103_TRAIN_CLOTHING_DIAGNOSTIC',query_groups=groups,group_counts=counts,
        original28_unchanged_pixel_rays=True,source_manifest_sha256=sha(manifest),source_review_sha256=sha(review),
        original_queries_sha256=sha(original),transforms_sha256=sha(data/'transforms.json'),builder_sha256=sha(Path(__file__)),
        source='Original28shirt rays plus75 manually reviewed actual detected-feature pixel centers; no heldout camera supervision',
        limitation='Source point/ray residuals explicitly retained. All observations TRAIN and correlated SfM; reserves excluded only from auxiliary objectives, not RGB. No whole-scene or perfect-grid certificate.')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(request,indent=2)+'\n');print(json.dumps(dict(output=str(output),groups=counts,max_perpendicular_error=max(p['point_to_pixel_ray_distance']for p in request['pixels']))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.campaign,a.output)
