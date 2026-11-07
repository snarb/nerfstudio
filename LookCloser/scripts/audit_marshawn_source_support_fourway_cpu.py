"""CPU-only four-way SOURCE support feasibility; never native occupancy mutation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import maximum_filter
from marshawn_source_point_admission import SourcePointAdmission
from marshawn_source_margin_admission import GeometricSourcePointAdmission
from marshawn_semantic_cell_admission import WholeCellAdmission
from audit_marshawn_source_margin_cpu import integrate

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def ray_union(origin,direction,lo,hi,bounds):
    origin=np.asarray(origin,dtype=np.float64);direction=np.asarray(direction,dtype=np.float64)
    if np.any(abs(direction)<1e-14):
        nonzero=abs(direction)>=1e-14;near=np.full_like(lo,-np.inf);far=np.full_like(hi,np.inf);valid=np.ones(len(lo),bool)
        for axis in range(3):
            if nonzero[axis]:a=(lo[:,axis]-origin[axis])/direction[axis];b=(hi[:,axis]-origin[axis])/direction[axis];near[:,axis]=np.minimum(a,b);far[:,axis]=np.maximum(a,b)
            else:valid&=(origin[axis]>=lo[:,axis])&(origin[axis]<=hi[:,axis])
    else:
        a=(lo-origin)/direction;b=(hi-origin)/direction;near=np.minimum(a,b);far=np.maximum(a,b);valid=np.ones(len(lo),bool)
    # The source UNKNOWN outside policy is not positive support. All geometric
    # intervals below are clipped to the unchanged accepted field AABB.
    t0,t1=0.,np.inf
    for axis in range(3):
        if abs(direction[axis])<1e-14:
            if not bounds[0,axis]<=origin[axis]<=bounds[1,axis]:return []
        else:
            a=(bounds[0,axis]-origin[axis])/direction[axis];b=(bounds[1,axis]-origin[axis])/direction[axis]
            t0=max(t0,min(a,b));t1=min(t1,max(a,b))
    start=np.maximum(near.max(1),t0);end=np.minimum(far.min(1),t1);valid&=end>start+1e-12
    starts=start[valid];ends=end[valid];order=np.argsort(starts);merged=[]
    for i in order:
        x,y=float(starts[i]),float(ends[i])
        if merged and x<=merged[-1][1]+1e-12:merged[-1][1]=max(merged[-1][1],y)
        else:merged.append([x,y])
    return merged

def main(c,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);assert not torch.cuda.is_initialized();started=time.monotonic();base=c/'native_semantic_grid_20000';pointbase=c/'independent_occupancy/quarter_distortion_sourcepoint_counterfactual_v1';inputs={};variants={};sources={};contracts={};bounds=None
    with np.load(c/'semantic_margin_continuation_candidate_v1/anchor/grid.npz')as z:target_bounds=z['cascade_aabbs'][0].reshape(2,3)
    for width in ['W32','W64']:
        folder=base/'occlusion_v2'/width;manifest=read(folder/'support_manifest.json');receipt=read(folder/'grid_receipt.json');sources[width]=manifest
        for name in ['support_manifest.json','grid_receipt.json','semantic_grid.npz','final_train_ray_receipt.json','final_train_ray_misses.npz']:inputs[str(folder/name)]=sha(folder/name)
        assert receipt['grid_sha256']==sha(folder/'semantic_grid.npz')and receipt['no_eval_masks_or_RGB_read']and len(receipt['train_cameras'])==63
        with np.load(folder/'semantic_grid.npz')as z:
            if bounds is None:bounds=z['aabb'].copy();transform=z['world_to_box'].copy()
            else:assert np.array_equal(bounds,z['aabb'])and np.array_equal(transform,z['world_to_box'])
            assert np.array_equal(z['semantic_gate_pad1'],maximum_filter(z['semantic_gate0'],size=3,mode='constant',cval=0))
            for key in ['semantic_gate0','semantic_gate_pad1']:
                mask=z[key].copy();name=width+'_'+key;pitch=(bounds[1]-bounds[0])/np.array(mask.shape);indices=np.argwhere(mask);low=bounds[0]+indices*pitch-.002-2e-7;high=bounds[0]+(indices+1)*pitch+.002+2e-7;lookup=GeometricSourcePointAdmission(torch.from_numpy(mask),bounds,margin=.002,epsilon=2e-7,base_class=SourcePointAdmission);mapped=WholeCellAdmission(mask,bounds).map_target_grid((128,128,128),target_bounds)
                variants[name]=dict(width=width,key=key,mask=mask,lookup=lookup,low=low,high=high,source_cells=int(mask.sum()),whole_LC_cover_cells=int(mapped.sum()),mapped_sha256=hashlib.sha256(mapped.tobytes()).hexdigest())
        contracts[width]={k:manifest[k]for k in ['source_corrections_sha256','mat_manifest_sha256','bag_qualification_sha256','source_transforms_sha256','clear_vote_disabled_cameras']}
    assert contracts['W32']==contracts['W64'];assert np.array_equal(bounds,np.array(read(c/'data/ready.json')['aabb']))
    classification=[];rows64={r['camera']:r for r in sources['W64']['records']}
    raw_manifest=c.parent/'two_actor_isolation/confidence_native_hair_v2/manifest.json'
    raw_rows={r['camera']:r for r in read(raw_manifest)['rows']};inputs[str(raw_manifest)]=sha(raw_manifest)
    margin_manifest=base/'margin_W32/support_manifest.json';margin_rows={r['camera']:r for r in read(margin_manifest)['records']};inputs[str(margin_manifest)]=sha(margin_manifest)
    for r in sources['W32']['records']:
        other=rows64[r['camera']];assert r['source_semantic_sha256']==other['source_semantic_sha256']and r['mat_mask_sha256']==other['mat_mask_sha256']
        semantic=c/'data/semantics'/(r['camera']+'.npz');assert sha(semantic)==r['source_semantic_sha256'];inputs[str(semantic)]=r['source_semantic_sha256']
        raw=raw_rows[r['camera']];assert raw['split']=='TRAIN' and sha(raw['raw_archive'])==raw['raw_archive_sha256']==margin_rows[r['camera']]['raw_archive_sha256'];inputs[raw['raw_archive']]=raw['raw_archive_sha256']
        assert sha(r['mat_mask_path'])==r['mat_mask_sha256'];inputs[r['mat_mask_path']]=r['mat_mask_sha256']
        for row in [r,other]:assert sha(row['archive'])==row['archive_sha256'];inputs[row['archive']]=row['archive_sha256']
        with np.load(r['archive'])as a,np.load(other['archive'])as b:
            assert np.array_equal(a['valid'],b['valid'])and np.array_equal(a['unknown'],b['unknown'])and not(a['support']&~b['support']).any()
            classification.append(dict(camera=r['camera'],valid_byte_identical=True,UNKNOWN_byte_identical=True,positive_actor_source_semantic_sha256=r['source_semantic_sha256'],mat_source_sha256=r['mat_mask_sha256'],W32_support_subset_W64=True,support_removed_pixels=int((b['support']&~a['support']).sum())))
    queries={name:read(pointbase/'queries'/name)for name in ['clothing103.json','shoe6.json','DC12.json']}
    for name in queries:inputs[str(pointbase/'queries'/name)]=sha(pointbase/'queries'/name)
    positives=[]
    for name in ['clothing103.json','shoe6.json']:
        q=queries[name]
        for p,ray in zip(q['pixels'],q['rays']):positives.append((name,p,ray))
    xyz=np.array([p['witness_xyz']for _,p,_ in positives]);results={}
    frames={f['physical_camera']:f for f in read(c/'data/transforms.json')['frames']}
    for name,v in variants.items():
        retained=v['lookup'](torch.tensor(xyz,dtype=torch.float64)).numpy();records=[]
        for index,(source,p,ray)in enumerate(positives):
            depth=p['query_witness_depth'];o=np.array(ray['origin']);d=np.array(ray['direction']);point=o+depth*d;depth_admit=bool(v['lookup'](torch.tensor(point[None],dtype=torch.float64)).item());intervals=ray_union(o,d,v['low'],v['high'],bounds);containing=[x for x in intervals if x[0]-1e-12<=depth<=x[1]+1e-12];cover=sum(max(0.,min(y,depth+.02)-max(x,depth-.02))for x,y in intervals)/.04
            records.append(dict(index=index,source=source,camera=p['camera_name'],witness_id=p['witness_id'],actor=p.get('actor'),region=p.get('region'),true_point_retained=bool(retained[index]),pixel_depth_point_retained=depth_admit,depth_interval=containing[0]if containing else None,front_depth_margin=depth-containing[0][0]if containing else None,back_depth_margin=containing[0][1]-depth if containing else None,band_pm02_fraction=min(1.,cover),point_to_pixel_ray_distance=p['point_to_pixel_ray_distance']))
        historic=read(base/'occlusion_v2'/v['width']/'final_train_ray_receipt.json');misspath=base/'occlusion_v2'/v['width']/'final_train_ray_misses.npz';remaining=[];checked=0
        with np.load(misspath)as z:
            for frame in historic['rows']:
                cam=frame['camera'];coords=z[cam+'_'+v['key']];f=frames[cam];pose=np.array(f['transform_matrix']);o=pose[:3,3]
                for x,y in coords:
                    local=np.array([(x+.5-f['cx'])/f['fl_x'],-(y+.5-f['cy'])/f['fl_y'],-1.]);direction=pose[:3,:3]@local;direction/=np.linalg.norm(direction);intervals=ray_union(o,direction,v['low'],v['high'],bounds);checked+=1
                    if not intervals:remaining.append(dict(camera=cam,x=int(x),y=int(y)))
        results[name]=dict(source_cells=v['source_cells'],whole_LC_cover_cells=v['whole_LC_cover_cells'],mapped_sha256=v['mapped_sha256'],geometric_Linf_margin=.002,numerical_epsilon=2e-7,source_halo_distinct_from_native_R=True,positive_occurrences=len(records),unique_positive_positions=len(np.unique(xyz,axis=0)),retained_true_points=int(retained.sum()),retained_pixel_depth_points=sum(r['pixel_depth_point_retained']for r in records),minimum_front_depth_margin=min(r['front_depth_margin']for r in records if r['front_depth_margin']is not None),minimum_back_depth_margin=min(r['back_depth_margin']for r in records if r['back_depth_margin']is not None),minimum_band_pm02_fraction=min(r['band_pm02_fraction']for r in records),positive_records=records,historical_ray_misses_without_margin=checked,remaining_historical_ray_misses_with_margin=len(remaining),remaining_ray_miss_pixels=remaining,historical_positive_ray_count=historic['totals']['actor_rays'])
        write(out/(name+'_positives.json'),results[name]);print(name,'positive',int(retained.sum()),'remainingraymisses',len(remaining),flush=True)
    # All supplied positions are from already saved exact126316 A queries. No
    # network is loaded or queried; this is fixed-position integration only.
    tracebase=pointbase/'quarter_distortion_raw126316/native_A';mass=[]
    for label in ['clothing103','DC12']:
        q=queries[label+'.json']
        audit_path=tracebase/(label+'_trace/rays/audit.json');audit=read(audit_path);inputs[str(audit_path)]=sha(audit_path)
        assert audit['checkpoint_sha256']=='340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641' and audit['step']==126316 and audit['frozen_state_unchanged']
        assert audit['transforms_sha256']==contracts['W64']['source_transforms_sha256'] and len(audit['rows'])==len(q['pixels'])
        for i,p in enumerate(q['pixels']):
            assert audit['rows'][i]['pixel']==p
            for profile in ['native','uniform2x']:
                path=tracebase/(label+'_trace/rays')/f'pixel{i:04d}_{profile}.npz'
                if not path.exists():raise FileNotFoundError(path)
                inputs[str(path)]=sha(path)
                with np.load(path)as z:
                    xyz_t=torch.from_numpy(z['xyz']);weights=z['actual_weights'];mid=(z['t_start']+z['t_end'])/2;band=abs(mid-p['query_witness_depth'])<=.02 if 'query_witness_depth'in p else np.zeros(len(mid),bool)
                    for name,v in variants.items():
                        keep=v['lookup'](xyz_t).numpy();cw,rgb=integrate(z['t_start'],z['t_end'],z['sigma'],z['rgb'],keep)
                        mass.append(dict(variant=name,label=label,index=i,profile=profile,actor=p.get('actor'),witness_id=p.get('witness_id'),original_actual_mass=float(weights.sum()),removed_original_mass=float(weights[~keep].sum()),original_band_mass=float(weights[band].sum()),removed_original_band_mass=float(weights[band&~keep].sum()),counterfactual_opacity=float(cw.sum()),counterfactual_RGB=rgb.tolist()))
    write(out/'saved_sample_mass.json',dict(rows=mass,kind='FIXED_SAVED_A_POSITIONS_CPU_INTEGRAL_NOT_NATIVE_REPLAY',source_checkpoint_sha256='340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641'))
    summary=[]
    for name,v in results.items():
        item={k:value for k,value in v.items()if k not in ['positive_records','remaining_ray_miss_pixels']}
        item['variant']=name
        for label in ['clothing103','DC12']:
            for profile in ['native','uniform2x']:
                rows=[x for x in mass if x['variant']==name and x['label']==label and x['profile']==profile];item[label+'_'+profile]=dict(mean_original_removed_mass=float(np.mean([x['removed_original_mass']for x in rows])),total_removed_band_mass=float(sum(x['removed_original_band_mass']for x in rows)),total_original_band_mass=float(sum(x['original_band_mass']for x in rows)))
        summary.append(item)
    assert not torch.cuda.is_initialized();write(out/'report.json',dict(passed=True,CPU_only=True,no_model_loaded=True,no_grid_or_labels_changed=True,CUDA_initialized=False,seconds=time.monotonic()-started,source_width_units='native_RAW_pixels_not_previous_RGB_loss_width',shared_contracts=contracts,per_camera_classification=classification,table=summary,inputs_sha256=inputs,helper_sha256=sha(__file__),limitations=['Sparse109 observations are41 unique positive positions, not full hair/feet coverage.','5568 inherited material/occlusion ambiguity remains; not an optical alpha1 target.','Depth margins are exact along existing calibrated pixel rays, not global3D clearance or calibrated uncertainty.','Historical4.657M ray receipt is inherited; only its known misses are retested because geometric expansion is monotone.','Whole-cell cover counts are CPU diagnostics, not changes to current nativeR0 traversal or production support.','Saved-sample integration is not actual new field/native rendering, and exclusion does not certify physical free space.']))
    print(out,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.campaign,a.output)
