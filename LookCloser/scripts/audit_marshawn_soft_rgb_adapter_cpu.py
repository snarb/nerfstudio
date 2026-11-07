"""Actual frozen CPU sampler/loss check for confidence proposals; no model loaded."""
import argparse,copy,hashlib,json,os,pickle,random,sys
from pathlib import Path
from types import SimpleNamespace


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(c,candidate,profiles):
    os.environ['CUDA_VISIBLE_DEVICES']='';scripts=candidate/'code/LookCloser/scripts';sys.path.insert(0,str(scripts))
    import numpy as np
    import torch
    import marshawn_tuning_sampling as sampling
    import marshawn_bag_loss_correction as correction
    inventory=json.loads((candidate/'source_manifest.json').read_text())['files_sha256']
    for module in [sampling,correction]:assert Path(module.__file__).resolve().parent==scripts.resolve()and sha(module.__file__)==inventory['LookCloser/scripts/'+Path(module.__file__).name]
    analysis=json.loads((profiles/'analysis.json').read_text());source_summary=json.loads((profiles/'summary.json').read_text());source_rows={r['camera']:r for r in source_summary['rows']}
    tf=json.loads((c/'data/transforms.json').read_text());train=set(tf['train_filenames']);names=[c/'data'/f['file_path']for f in tf['frames']if f['file_path']in train];assert len(names)==63
    pipe=SimpleNamespace(datamanager=SimpleNamespace(train_pixel_sampler=SimpleNamespace(sample=lambda *a,**k:None,sample_count=0,num_rays_per_batch=4096,config=SimpleNamespace(keep_full_image=False)),train_dataset=SimpleNamespace(image_filenames=names)),model=SimpleNamespace(config=SimpleNamespace(reconstruction_loss_type='charbonnier'),get_loss_dict=lambda *a,**k:{'rgb_loss':None}))
    adapter=sampling.SamplingAdapter(pipe,c/'data','actor','body80',64)
    bagroot=candidate/'bag_loss_source';spec=json.loads((candidate/'spec.json').read_text());bag=correction.validate_manifest(bagroot/'manifest.json',spec['bag_loss_correction']['manifest_sha256']);bagrows={r['camera']:r for r in bag['records']}
    def rng():return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()))+torch.get_rng_state().numpy().tobytes()).hexdigest()
    def structure():
        h=hashlib.sha256()
        for row in adapter.rows:
            for key in ['core','band','valid','black','grey','actor_union','W32','W64']:h.update(np.ascontiguousarray(row[key]).tobytes())
            for key in ['core','band','valid']:h.update(row['pools'][key].tobytes())
            h.update(row['fas']['probabilities'].tobytes())
            for cells in row['fas']['cells']:
                for cell in cells:h.update(cell.tobytes())
        return h.hexdigest()
    original_structure=structure();initial=copy.deepcopy(adapter.state_dict());h,w=adapter.rows[0]['core'].shape;batch=dict(image=torch.zeros((1,1,1,3),dtype=torch.uint8).expand(63,h,w,3),image_idx=torch.arange(63));reference={};results=[]
    for variant in ['original','bag6522','delta2','delta4']:
        before=rng();changed=0
        for row,source in zip(adapter.rows,adapter.source_records):
            name=source['camera'];assert source['mask_sha256']==source_rows[name]['main_semantic_sha256'];weight=row['W64']
            if variant=='bag6522'and name in bagrows:
                b=bagrows[name];weight,chosen=correction.corrected_confidence(dict(row,weight=weight),np.load(bagroot/b['exclusion']),np.load(bagroot/b['unknown']),b['false_grey_pixels']);changed+=int(chosen.sum())
            elif variant in ['delta2','delta4']:
                with np.load(profiles/'transfer'/f'{name}.npz')as z:weight=z[variant]
            row['weight']=weight;row['proposal_means']=correction.proposal_means(adapter,row)
        if variant=='bag6522':assert changed==6522
        assert rng()==before and structure()==original_structure
        for expected in analysis['phases']:
            step=expected['step'];adapter.load_state_dict(initial);adapter.set_step(step);np.random.seed(42+step);torch_before=torch.get_rng_state().clone();sample=adapter.sample(batch);rng_after=rng();assert torch.equal(torch_before,torch.get_rng_state())
            ids=sample['indices'];confidence=sample['actor_confidence'];norm=sample['actor_proposal_mean_weight'];assert abs(norm-expected['normalizer'][variant])<1e-12
            prediction=torch.full((4096,3),.5,requires_grad=True);pipe.model.get_loss_dict({'rgb':prediction},sample)['rgb_loss'].backward();grad=prediction.grad.detach().clone()
            if variant=='original':reference[step]=dict(indices=ids.clone(),rng=rng_after,confidence=confidence.clone(),normalizer=norm,gradient=grad)
            else:
                ref=reference[step];assert torch.equal(ids,ref['indices'])and rng_after==ref['rng'];old=ref['confidence'];ratio=torch.zeros_like(old);positive=old>0;ratio[positive]=confidence[positive]/old[positive]*(ref['normalizer']/norm);torch.testing.assert_close(grad,ref['gradient']*ratio[:,None],rtol=3e-6,atol=1e-10)
            results.append(dict(variant=variant,step=step,normalizer=norm,same_indices_and_RNG=True,RGB_gradient_exact_single_weight_and_analytic_normalizer=True))
    assert structure()==original_structure and not torch.cuda.is_initialized()
    receipt=dict(passed=True,CPU_only=True,actual63camera_frozen_adapter=True,source_sampling_sha256=sha(sampling.__file__),source_correction_sha256=sha(correction.__file__),pools_labels_FAS_source_weights_unchanged_sha256=original_structure,confidence_installation_RNG_unchanged=True,phases=results,analysis_sha256=sha(profiles/'analysis.json'),model_optimizer_or_GPU_created=False,training_started=False,limitation='Proposal feasibility only. A production/resume policy binding is not installed by this audit.')
    (profiles/'actual_adapter_CPU.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(passed=True,phases=len(results))))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--profiles',type=Path,required=True);a=ap.parse_args();main(a.campaign,a.candidate,a.profiles)
