"""Evaluation-only incremental source-ordering diagnostic on CPU.

TRAIN confidence and both full counterfactual predictions are constructed
before reading target GT/masks. Only aggregate comparisons are saved; there
is no GT-selected label map, texture mosaic, gate, or production cache.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from luster_hybrid_common import read,write,sha,geometry,assert_source,native_rows
from train_perceptual_utility_luster_hybrid import load_metric,metric_state_sha
from bracket_perceptual_luster_hybrid import projected_utility
from evaluate_current_source_potential_luster import counterfactual


def agreement(cost_difference,error_difference,valid):
    assert cost_difference.shape==error_difference.shape==valid.shape
    good=valid&np.isfinite(cost_difference)&np.isfinite(error_difference)
    informative=good&(np.abs(cost_difference)>.01)&(np.abs(error_difference)>.001)
    correct=cost_difference*error_difference>0
    weights=np.abs(error_difference)
    return dict(common_pixels=int(good.sum()),informative_pixels=int(informative.sum()),
        agreement_fraction=float(correct[informative].mean()) if informative.any() else None,
        error_margin_weighted_agreement=float((correct[informative]*weights[informative]).sum()/weights[informative].sum()) if informative.any() else None,
        mean_absolute_error_margin=float(weights[informative].mean()) if informative.any() else None)


def run(root):
    pilot=root/'probes/bracket_perceptual_cpu';pilot.mkdir(parents=True,exist_ok=True)
    (pilot/'logs').mkdir(exist_ok=True)
    for key in ['geometry','preflight','variants','candidates']:
        path=pilot/key
        if not path.exists():path.symlink_to(root/key,target_is_directory=True)
        assert path.resolve()==(root/key).resolve()
    metric,mr=load_metric(pilot,'cpu');state=metric_state_sha(metric)
    config=read(root/'candidates/single_photometric_depth25/config.json')
    selection=read(root/'preflight/bracket_witness_all_geometry_coverage.json')
    native=native_rows();results=[]
    for target in [7,12,95,150]:
        sources=read(root/'candidates/single_photometric_depth25_seam'/f'cam{target:03d}_receipt.json')['sources']
        sources=[s for s in sources if selection[str(target)]['sources'][str(s)]['two_witness_eligible_fraction']>0][:2]
        assert len(sources)==2
        base=geometry(root,target)['rgb'];predictions=[];costs=[];known=[];inputs={}
        for source in sources:
            assert_source(native,target,source)
            confidence=projected_utility(pilot,target,source,config,metric,mr)
            d=root/'variants/cubic75/warps'/f'cam{target:03d}'/f'src{source:03d}'
            receipt=read(d/'receipt.json');x0,y0,x1,y1=receipt['box'];sl=np.s_[y0:y1,x0:x1]
            record={key:np.load(d/f'{key}.npy',mmap_mode='r') for key in receipt['arrays']}
            prediction=np.array(base)
            prediction[sl]=counterfactual(np.asarray(base[sl]),record,source,config)
            predictions.append(prediction)
            cost=np.zeros(base.shape[:2],np.float32);cost[sl]=np.load(confidence/'cost.npy')
            support=np.zeros(base.shape[:2],bool);support[sl]=np.load(confidence/'count.npy')>=2
            costs.append(cost);known.append(support)
            for p in [d/'receipt.json',confidence/'receipt.json',confidence/'cost.npy',confidence/'count.npy']:
                inputs[str(p)]=sha(p)
        labels=np.load(root/'candidates/single_photometric_depth25_seam'/f'cam{target:03d}_sources.npy',mmap_mode='r')
        # Evaluation begins only after prediction/confidence construction.
        protocol_path=root/'evaluation/protocol.json'
        entry=read(protocol_path)['cameras'][str(target)]
        assert sha(entry['source'])==entry['source_sha256'] and sha(entry['mask'])==entry['mask_sha256']
        gt=np.asarray(Image.open(entry['source']).convert('RGB'),np.float32)/255
        mask=np.asarray(Image.open(entry['mask']))>0
        yy,xx=np.where(mask);x0=max(0,int(xx.min())-16);x1=min(mask.shape[1],int(xx.max())+17)
        y0=max(0,int(yy.min())-16);y1=min(mask.shape[0],int(yy.max())+17);sl=np.s_[y0:y1,x0:x1]
        m=mask[sl]
        def tensor(rgb):
            value=np.where(m[...,None],rgb[sl],.5).astype(np.float32)
            return torch.from_numpy(value).permute(2,0,1)[None]*2-1
        with torch.inference_mode():
            truth=tensor(gt)
            errors=[metric(tensor(prediction),truth)[0,0].numpy() for prediction in predictions]
        active=m&known[0][sl]&known[1][sl]
        dc=costs[0][sl]-costs[1][sl];de=errors[0]-errors[1]
        chosen=labels[sl]
        pair_choice=(chosen==sources[0])|(chosen==sources[1])
        informative=active&pair_choice&(np.abs(dc)>.01)&(np.abs(de)>.001)
        baseline_dc=np.where(chosen==sources[0],-1.,1.)
        incremental=dict(common_informative_pixels=int(informative.sum()),
            baseline=agreement(baseline_dc,de,informative),
            bracket=agreement(dc,de,informative),
            changed_preference_pixels=int((informative&((baseline_dc<0)!=(dc<0))).sum()))
        tiles_c=[];tiles_e=[]
        for y in range(0,m.shape[0],64):
            for x in range(0,m.shape[1],64):
                ts=np.s_[y:y+64,x:x+64];valid=active[ts]
                if valid.size<4096 or valid.mean()<.5:continue
                tiles_c.append(float(dc[ts][valid].mean()));tiles_e.append(float(de[ts][valid].mean()))
        tc=np.asarray(tiles_c);te=np.asarray(tiles_e)
        row=dict(camera=target,sources=sources,incremental=incremental,pixels=agreement(dc,de,active),
            tiles64=agreement(tc,te,np.ones(tc.shape,bool)),tile_count=len(tc),
            counterfactual_person_lpips=[float(error[m].mean()) for error in errors],inputs=inputs,
            protocol_sha256=sha(protocol_path))
        results.append(row)
        write(root/'evaluation'/f'bracket_incremental_ranking_cpu_cam{target:03d}.json',row)
        print({k:v for k,v in row.items() if k!='inputs'},flush=True)
        del predictions,costs,known,gt,errors
    assert metric_state_sha(metric)==state
    write(root/'evaluation/bracket_incremental_ranking_cpu.json',dict(usage='evaluation_only_never_construction',
        code_sha256=sha(Path(__file__)),metric_unchanged=True,rows=results,
        selection='first two source-pool cameras with positive geometry-only two-witness coverage',
        limitation='Existing label comparison is restricted to pixels where the reference chooses one of this pair. CPU exploratory paired counterfactuals with incompatible image neighborhoods; not a final hybrid score, achievable bound, or per-pixel GT assignment.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
