"""CPU-only class-evidence proposal; no optimizer, field, cache edits or GPU.

Compare exact current posteriors and an algebraic FIRST-Adam-update diagnostic
for one old camera packet, with versus without proposed extra class rows. This
is not a trained checkpoint or a prediction of a multi-step GPU trajectory.
"""
import argparse,json,os,sys,shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from marshawn_frozen_semantic_head import FrozenActorGrid,rendered_classes,conditional_class_objective,tensor_digest
from marshawn_semantic_residual_head import FrozenResidualActorGrid
from prepare_marshawn_frozen_semantic_cpu import sha,write


def grid_values(grid,xyz,bounds):
    unit=(xyz-bounds[0])/(bounds[1]-bounds[0]);inside=((unit>=0)&(unit<=1)).all(-1)
    coords=(2*unit-1)[:,[2,1,0]].reshape(1,1,1,-1,3)
    return F.grid_sample(grid,coords,align_corners=True,padding_mode='border').reshape(-1),inside


def raw_logits(head,xyz):
    if hasattr(head,'residual_logits'):
        a,inside=grid_values(head.coarse_logits,xyz,head.bounds);b,_=grid_values(head.residual_logits,xyz,head.bounds);return a+b,inside
    return grid_values(head.logits,xyz,head.bounds)


def run(campaign,output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);output.mkdir(parents=True,exist_ok=False)
    Froot=campaign/'frozen_LC_semantic_matched_refinement_plan_v1'
    capturep=campaign/'frozen_LC_standalone21_capture_plan_v1/capture/complete.json'
    mp=campaign/'native_actor_extra16_capture_plan_CPU_v1/manifest.json'
    capture=json.loads(capturep.read_text());manifest=json.loads(mp.read_text())
    assert sha(capturep)=='6144bdc542ae30108b8c987c92271cbe2a251a8049f0a12e8a5262af3ae5dfca'
    assert capture['passed']and all(capture['source_checks'].values())and capture['all_query_fit_weights']==0
    assert capture['pipeline_digest']=='c83bc7dbb1504063c35f0a2b14ae3fb329ff27ed2eb6b3661eb89b4918d46636'
    queryp=campaign/'frozen_LC_standalone21_capture_plan_v1/query_manifest.json'
    query_manifest=json.loads(queryp.read_text())
    assert sha(queryp)==capture['query_manifest_sha256']
    assert sha(mp)==query_manifest['extra16_canonical_manifest_sha256']=='9205c0953dab3bc7b3710e1b750dc95b32f26f540df26daa1cd42061b20c721a'
    intended={r['query_id']:r for r in manifest['rows']}
    assert len(intended)==16 and sum(r['proposed_future_weight']==1 for r in intended.values())==15
    assert sum(r['semantic_target']==0 for r in intended.values())==12
    code=output/'code';code.mkdir()
    shutil.copyfile(Path(__file__),code/Path(__file__).name)
    for module in ['marshawn_frozen_semantic_head','marshawn_semantic_residual_head','prepare_marshawn_frozen_semantic_cpu']:
        leaf=Path(sys.modules[module].__file__);shutil.copyfile(leaf,code/leaf.name)
    hashes={str(mp):sha(mp),str(capturep):sha(capturep),str(queryp):sha(queryp)}
    hashes.update({str(p):sha(p)for p in code.iterdir()})
    for r in intended.values():
        leaf=Path(r['source_review']);assert sha(leaf)==manifest['inputs_sha256'][str(leaf)]
        hashes[str(leaf)]=sha(leaf)
    remote='/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106'
    def local(path):return Path(path.replace(remote,str(campaign)))
    extra={};rows=[]
    for record in capture['records']:
        p=local(record['archive']);assert sha(p)==record['archive_sha256'];hashes[str(p)]=sha(p)
        with np.load(p,allow_pickle=False)as z:
            assert np.count_nonzero(z['label_weight'])==0
            indices=[i for i,q in enumerate(z['query_id'])if str(q)in intended]
            if not indices:continue
            chosen_samples=np.isin(z['ray_indices'],indices);remap={old:new for new,old in enumerate(indices)}
            q=dict(xyz=torch.from_numpy(z['xyz'][chosen_samples]),weight=torch.from_numpy(z['weight'][chosen_samples]),
                ray_indices=torch.tensor([remap[int(i)]for i in z['ray_indices'][chosen_samples]],dtype=torch.int64),
                sigma=torch.from_numpy(z['sigma'][chosen_samples]),starts=torch.from_numpy(z['starts'][chosen_samples]),ends=torch.from_numpy(z['ends'][chosen_samples]),
                target=torch.from_numpy(z['semantic_target'][indices]),confidence=torch.tensor([intended[str(z['query_id'][i])]['proposed_future_weight']for i in indices]),
                alpha=torch.from_numpy(z['native_alpha'][indices]),cap=torch.from_numpy(z['accepted_box_actor_cap'][indices]),
                query_ids=[str(z['query_id'][i])for i in indices],camera=record['camera'])
            extra[record['camera']]=q
            packet=output/(record['camera']+'_proposed_class_packet.npz')
            np.savez_compressed(packet,xyz=q['xyz'].numpy(),weight=q['weight'].numpy(),ray_indices=q['ray_indices'].numpy(),
                semantic_target=q['target'].numpy(),label_weight=np.zeros(len(indices),np.float32),proposed_label_weight=q['confidence'].numpy(),
                native_alpha=q['alpha'].numpy(),accepted_box_actor_cap=q['cap'].numpy(),query_id=np.array(q['query_ids']),
                world_to_box=np.eye(4),model_digest=np.array(capture['pipeline_digest']),native_packet_context=np.array('STANDALONE21_ACTUAL_CONTEXT_NOT_ORIGINAL128'))
            hashes[str(packet)]=sha(packet)
            for j,i in enumerate(indices):
                r=intended[str(z['query_id'][i])]
                assert z['semantic_target'][i]==r['semantic_target']and r['split']=='TRAIN'
                assert q['alpha'][j]>0 and q['cap'][j]/q['alpha'][j]>.95
                rows.append(dict(query_id=q['query_ids'][j],camera=record['camera'],pixel_xy=z['pixel_xy'][i].tolist(),target=float(q['target'][j]),
                    current_fit_weight=0,proposed_weight=float(q['confidence'][j]),native_alpha=float(q['alpha'][j]),box_cap_ratio=float(q['cap'][j]/q['alpha'][j]),
                    samples=int((z['ray_indices']==i).sum()),source_review=r['source_review'],category=r['category'],pure_class_only=True,
                    optical_alpha_target=None,whole_ray_empty_target=False,root_LC_qualification_pending=True))
    assert len(rows)==16 and len([r for r in rows if r['proposed_weight']==1])==15
    cachep=campaign/'frozen_LC_semantic_head_silver_alias_v1/capture_adapter_receipt.json';cache=json.loads(cachep.read_text());old_by_camera={r['camera']:r for r in cache['records']if r['rays']}
    labels=json.loads((campaign/'frozen_LC_semantic_LA_two_UNKNOWN_candidate_v1/label_qualification.json').read_text());unknown={(r['camera'],*r['pixel_xy'])for r in labels['unknown_queries']}
    basep=campaign/'frozen_LC_semantic_head_archives/LA_four_UNKNOWN_20261107_v1/runs/prior001_fresh2064/head.pt';base=torch.load(basep,map_location='cpu',weights_only=False);hashes[str(basep)]=sha(basep)
    bounds=base['head']['bounds'];coarse=FrozenActorGrid(bounds,torch.eye(4),128,initial_probability=.01);coarse.load_state_dict(base['head'],strict=True)
    heads={'base7937':coarse}
    for arm in ['coarse128','residual256']:
        p=Froot/'runs'/(arm+'_fresh2064')/'head.pt';st=torch.load(p,map_location='cpu',weights_only=False);hashes[str(p)]=sha(p)
        h=FrozenActorGrid(bounds,torch.eye(4),128,initial_probability=.01)if arm=='coarse128'else FrozenResidualActorGrid(coarse,256)
        h.load_state_dict(st['head'],strict=True);assert tensor_digest(h)==st['head_digest'];heads[arm]=h
    witnesses=json.loads((Froot/'CPU_witnesses109.json').read_text())['records'];points=torch.tensor([r['witness_xyz']for r in witnesses],dtype=torch.float32)
    for name,h in heads.items():
        for camera,q in extra.items():
            with torch.no_grad():
                point_p=h(q['xyz']);posterior=rendered_classes(point_p,q['weight'],q['ray_indices'],len(q['target']))[:,0]/q['alpha']
            for i,qid in enumerate(q['query_ids']):
                row=next(r for r in rows if r['query_id']==qid);row[name+'_conditional_posterior']=float(posterior[i])
                sample=q['ray_indices']==i;tau=(q['sigma'][sample].double()*(q['ends'][sample]-q['starts'][sample]).double())
                responsibility={}
                for theta in [.1,.5]:
                    selected=point_p[sample]>=theta;native=q['weight'][sample]
                    selected_tau=float(tau[selected].sum())
                    responsibility[str(theta)]=dict(native_visible_contribution_fraction=float(native[selected].sum()/q['alpha'][i]),
                        selected_tau=selected_tau,selected_density_opacity=float(-np.expm1(-selected_tau)),
                        selected_tau_from_native_zero_weight=float(tau[selected&(native==0)].sum()),
                        opacity_is_unlabeled_diagnostic_not_a_target=True)
                row[name+'_responsibility']=responsibility
    heads['initial_residual256']=FrozenResidualActorGrid(coarse,256)
    probes=[]
    for name in ['coarse128','residual256','initial_residual256']:
        h=heads[name];parameter=h.residual_logits if hasattr(h,'residual_logits')else h.logits
        original_digest=tensor_digest(h)
        for camera,q in extra.items():
            oldrow=old_by_camera[camera];oldp=local(oldrow['archive']);assert sha(oldp)==oldrow['archive_sha256'];hashes[str(oldp)]=sha(oldp)
            with np.load(oldp,allow_pickle=False)as z:
                confidence=z['label_weight'].copy()
                for i,uv in enumerate(z['pixel_xy']):
                    if(camera,*uv.tolist())in unknown:confidence[i]=0
                classes=rendered_classes(h(torch.from_numpy(z['xyz'])),torch.from_numpy(z['weight']),torch.from_numpy(z['ray_indices']),len(z['semantic_target']))
                original=conditional_class_objective(classes,torch.from_numpy(z['semantic_target']),torch.from_numpy(confidence));nold=len(confidence)
            newclasses=rendered_classes(h(q['xyz']),q['weight'],q['ray_indices'],len(q['target']))
            # Separate new-packet reduction preserves old-camera mean exactly.
            # Its confidence0 FDhair row remains diagnostic, never a target.
            incremental=conditional_class_objective(newclasses,q['target'],q['confidence'])*len(q['target'])/nold
            ga=torch.autograd.grad(original,parameter,retain_graph=True)[0]
            gb=torch.autograd.grad(original+incremental,parameter)[0]
            before=h(points).detach();new_before=rendered_classes(h(q['xyz']),q['weight'],q['ray_indices'],len(q['target']))[:,0].detach()/q['alpha']
            for label,g in [('old_only',ga),('old_plus_proposed15',gb)]:
                # Exact algebraic firstAdam update from zero moments, LR.03,
                # eps1e-8. No optimizer is created and no parameter is mutated.
                update=-.03*g/(g.abs()+1e-8)
                raw,inside=raw_logits(h,points);d,_=grid_values(update,points,h.bounds);after=(raw.detach()+d).sigmoid()*inside
                raw,inside=raw_logits(h,q['xyz']);d,_=grid_values(update,q['xyz'],h.bounds);newp=(raw.detach()+d).sigmoid()*inside
                after_q=rendered_classes(newp,q['weight'],q['ray_indices'],len(q['target']))[:,0]/q['alpha']
                probes.append(dict(head=name,camera=camera,diagnostic=label,old_packet_rays=nold,new_pure_class_rows=int((q['confidence']>0).sum()),
                    new_native_packet_context='EXPLICIT_STANDALONE21_CAPTURE_NOT_ORIGINAL128_RAY_PACKET',
                    max_abs_logit_update=float(update.abs().max()),original_parameters_unchanged=True,
                    witness109_max_abs_probability_change=float((after-before).abs().max()),witness109_min_probability=float(after.min()),
                    witness109_crossings05=int(((before>=.5)!=(after>=.5)).sum()),
                    extra_posterior_before_after=[dict(query_id=qid,target=float(q['target'][i]),proposed_weight=float(q['confidence'][i]),before=float(new_before[i]),after=float(after_q[i]))for i,qid in enumerate(q['query_ids'])]))
            assert tensor_digest(h)==original_digest
    for p,d in hashes.items():assert sha(p)==d,p
    assert not torch.cuda.is_initialized()
    proposal=dict(kind='CPU_ONLY_EXTRA15_CLASS_EVIDENCE_REVIEW',torch_version=torch.__version__,numpy_version=np.__version__,CUDA_initialized=False,optimizer_created=False,field_loaded=False,
        original_69888_queries_and_native_weights_unchanged=True,all_standalone21_current_fit_weights_remain0=True,
        proposed_future_only=15,floor_negative_rows=12,shoe_positive_rows=3,FD_hair_fit_weight=0,
        rows=rows,algebraic_first_Adam_diagnostics=probes,inputs_sha256=hashes,
        root_review_complete=False,GPU_authorized=False,physical_export_authorized=False,
        sampling_proposal='Same60camera sampler/history. On FD/MC/MD only, keep original old-camera objective EXACT; add sum(nativealpha-weighted classCE of qualified extras)/original_camera_ray_count as separate reduction. Do not concatenate or rewrite native cache arrays.',
        controlled_experiment_proposal='One new fresh2064 frozen128base7937 plus zero256residual candidate with proposed15 class rows; reuse measured88871 control. SAME initial function/Adam0/LR.03/seed42/history; sole new factor15 TRAIN class labels. New schema/data-binding family required before execution.',
        caveats=['The algebraic single-step probe is not a trained checkpoint or evidence of multi-step convergence.',
          'Groundtruth semantic class0 means visible floor, never opticalalpha0 or whole-ray-empty.',
          'Standalone native visibility uses its actual small camera packet; original128-ray allocator equivalence is not claimed.',
          'Per-ray source review plus actualLC visibility still requires explicit ROOT qualification before any fit.',
          'No heldout RGB/masks or new thresholds enter fitting; full3+9 physical gate remains mandatory.'])
    write(output/'CPU_measurements_and_proposal.json',proposal)
    write(output/'root_review_TEMPLATE.json',dict(reviewer=None,extra15_class_qualification_complete=False,extra15_technical_GPU_authorized=False,
        extra15_fresh2064_GPU_authorized=False,extra15_physical_GPU_authorized=False,proposal_sha256=sha(output/'CPU_measurements_and_proposal.json'),inputs_sha256=hashes))
    print(json.dumps(dict(output=str(output),rows=len(rows),future_only15=True,CPU_only=True,optimizer_created=False,CUDA_initialized=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.campaign,a.output)
