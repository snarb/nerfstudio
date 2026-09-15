"""Final seal of fresh per-time fitting, generic admission and matched CPU review."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from transfer_local_mhr_prior import settings,verify
from review_local_mhr_transfer import VIEWS,camera_crops,enclosed_misses
from run_local_mhr_completion import read,save,sha,require,check_seal,prefix_check


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--completion',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--tests',type=Path,required=True);a=p.parse_args();spec=read(a.spec);s=settings(spec)
    root=a.completion;target=root/'final_seal.json';require(not target.exists(),'Final seal already exists')
    verify(spec);checked={};check_seal(s['FINAL'],checked)
    config=read(root/'config.json');require(config['frame']==spec['frame'],'Wrong source time')
    for path,h in config['input_hashes'].items():require(sha(path)==h,'Changed configured input');checked[path]=h
    audit=read(root/'audit.json');require(audit['status']=='passed','Generic audit incomplete')
    require(audit['config_sha256']==sha(root/'config.json'),'Changed config')
    for name,h in audit['inventory'].items():require(sha(root/name)==h,'Changed replay artifact: '+name)
    direct=read(root/'admission/audit.json');require(sha(root/'admission/audit.json')==audit['admission_audit_sha256'],'Changed depth audit')
    require(direct['native_ray_checks_replayed']==248 and direct['source_geometric_depth_hashes']==62,'Incomplete native/depth checks')
    geometry=prefix_check(config)
    d=np.load(root/'candidates/silhouette100/domain_evidence.npz');e=np.load(root/'candidates/silhouette100/proposal_evidence.npz')
    require(not d['unsafe_parent'][e['parent_triangle_ids']].any(),'Unsafe parent admitted')
    head=read(s['ROOT']/'head_audit.json');require(head['status']=='passed' and head['exact_model_forward_replays']==3 and head['reserved_views']==8,'Head replay missing')
    require(read(s['ROOT']/'conformance_audit.json')['status']=='passed','Conformance replay missing')
    continuation=read(s['FINAL']/'audit.json');require(continuation['status']=='passed' and continuation['exact_all_iterates_replayed']==100,'Continuation replay missing')
    source,views=camera_crops(spec);parent=read(spec['production_request']);renders=[]
    from joint_temporal_texture import HELD_CAMERAS
    from review_mhr_production_patch_control import check_recipe
    check_recipe(parent)
    import open3d as o3d
    from admit_mhr_local_patch_depth import Scene2
    from bake_joint_temporal_mesh import camera_depth
    mesh=o3d.io.read_triangle_mesh(source['mesh'])
    scene=Scene2(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
    mask_root=Path(source['source_masks']['root'])
    masks=dict(zip(read(mask_root/'cameras.json'),np.load(mask_root/'masks.npz')['masks']))
    baseline_depth_replay=[];metric_rows=0
    for view in VIEWS:
        rgb_data={}
        for branch in ['baseline','strict','interpolated']:
            folder=root/'admission/rgb'/view/branch;request=read(folder/'request.json');frame=folder/'frames'/spec['frame']
            require(request['recipe']==parent['recipe'] and request['inventory'][0]['camera']==views[view]['camera'],'Changed policy/camera')
            require(request['inventory'][0]['source_masks']==source['source_masks'],'Texture masks changed')
            for key in ['profiles_sha256','exposure_sha256','source_quality_implementation_sha256']:
                require(request[key]==parent[key],'Changed shared source policy')
            expected=source['mesh_sha256'] if branch=='baseline' else geometry[branch]['mesh_sha256']
            require(request['inventory'][0]['mesh_sha256']==expected,'Unexpected render mesh')
            complete=read(frame/'complete.json');require(complete['request_sha256']==sha(folder/'request.json'),'Changed render request')
            for name,h in complete['hashes'].items():require(sha(frame/name)==h,'Changed render output')
            result=read(frame/'result.json');require(result['mesh_sha256']==expected and result['camera']==views[view]['camera'],'Render input mismatch')
            names=result['source_cameras'];require(len(names)==len(set(names))==62 and not set(names)&HELD_CAMERAS,'Render source leak')
            require(set(names)==set(config['training_cameras']),'Render cohort mismatch')
            require(Image.open(frame/'frame.png').size==(1080,1920),'Wrong native RGB shape')
            depth=np.load(frame/'target_depth.npz')['depth'];require(depth.shape==(1080,1920) and np.isfinite(depth).all() and (depth>=0).all(),'Invalid native depth')
            rgb_data[branch]=(np.array(Image.open(frame/'frame.png')),np.rot90(depth))
            if branch=='baseline':
                raw,_,_=camera_depth(scene,views[view]['camera'])
                filtered=raw.copy();name=views[view]['camera']['physical_camera']
                if name in masks:filtered[masks[name]==0]=np.inf
                np.testing.assert_array_equal(np.where(np.isfinite(filtered),filtered,0),depth)
                raw_portrait=np.rot90(np.where(np.isfinite(raw),raw,0))
                baseline_depth_replay.append(dict(view=view,exact_current_masked_depth=True,
                    hits_removed_by_existing_texture_mask=int((np.isfinite(raw)&~np.isfinite(filtered)).sum()),
                    enclosed_unmasked=int(enclosed_misses(raw_portrait,views[view]['crop']).sum()),
                    enclosed_current_policy=int(enclosed_misses(np.rot90(depth),views[view]['crop']).sum())))
            for n,h in request['helpers'].items():path=Path(__file__).with_name(n);require(sha(path)==h,'Renderer helper changed');checked[str(path)]=h
            renders.append(dict(view=view,branch=branch,receipt_sha256=sha(frame/'complete.json')))
        for kind in ['native_clay','native_rgb']:
            receipt=read(root/kind/(view+'.json'));require(receipt['completion_config_sha256']==sha(root/'config.json'),'Review config changed')
            require(receipt['crops']==views,'Review cameras/crops changed')
            for item in receipt['files']:require(sha(item['path'])==item['sha256'],'Review image changed')
        original,bd=rgb_data['baseline'];holes=enclosed_misses(bd,views[view]['crop'])
        for row in read(root/'native_rgb'/(view+'.json'))['statistics']:
            rgb,d=rgb_data[row['branch']];new=(d>0)&(bd<=0);common=(d>0)&(bd>0)
            delta=np.zeros(d.shape);delta[common]=d[common]-bd[common]
            expected=dict(branch=row['branch'],enclosed_original_misses=int(holes.sum()),
                enclosed_remaining=int((holes&(d<=0)).sum()),new_hits=int(new.sum()),
                new_hits_without_rgb=int((new&(rgb.max(2)==0)).sum()),
                newly_black=int(((original.max(2)>0)&(rgb.max(2)==0)).sum()),
                lost_hits=int(((bd>0)&(d<=0)).sum()),nearer_over003=int((delta<-.003).sum()),
                maximum_nearer=float(max(0,-delta.min())),farther_over003=int((delta>.003).sum()))
            require(row==expected,'RGB metric replay mismatch');metric_rows+=1
    # All source transformations remain pinned and independently reconstructible.
    adapters=list(Path(spec['inputs']).glob('*adapter.json'))+list(root.glob('*adapter*.json'))
    for path in adapters:
        q=read(path)
        if 'original_source' not in q:continue
        require(sha(q['frozen_path'])==q['frozen_sha256'],'Adapter source changed')
        code=q['original_source']
        for edit in q['replacements']:
            require(code.count(edit['before'])==edit['expected_count'],'Adapter count mismatch')
            code=code.replace(edit['before'],edit['after'])
        require(code==q['generated_source'],'Adapter reconstruction mismatch')
    visual=read(root/'visual_review.json');require(visual['reviewer']=='LLM' and not visual['production_promoted'],'Visual verdict missing')
    for path,h in visual['viewed_images'].items():require(sha(path)==h,'Reviewed image changed');checked[path]=h
    for path in [a.spec,a.report,a.tests,Path(__file__),Path(__file__).with_name('transfer_local_mhr_prior.py'),Path(__file__).with_name('review_local_mhr_transfer.py')]:checked[str(path.resolve())]=sha(path)
    save(target,dict(status='passed',frame=spec['frame'],checked_bindings=checked,
        inventory={str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*')) if f.is_file()},
        prior_seal_sha256=sha(s['FINAL']/'final_seal.json'),geometry=geometry,renders=renders,
        exact_baseline_depth_replays=baseline_depth_replay,exact_rgb_metric_rows_replayed=metric_rows,
        report_sha256=sha(a.report),tests_sha256=sha(a.tests),script_sha256=sha(__file__),
        production_modified=False,full_sequence_accepted=False,independent_heldout_metrics=False))
    print('final seal passed',len(checked),'bindings',len(renders),'renders',flush=True)


if __name__=='__main__':main()
