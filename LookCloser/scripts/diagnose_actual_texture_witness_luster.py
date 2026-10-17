"""Evaluate completed TRAIN-only confidence; never construct GT source labels."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from diagnose_bracket_pair_ranking_luster import agreement
from evaluate_current_source_potential_luster import counterfactual
from luster_hybrid_common import geometry, read, sha, write
from train_perceptual_utility_luster_hybrid import load_metric, metric_state_sha


def run(root,pilot):
    summary=read(pilot/'summary.json');assert summary['metric_unchanged']
    records=summary['rows'];assert len(records)==2
    target=records[0]['target'];assert records[1]['target']==target
    sources=[r['source'] for r in records];assert len(set(sources))==2
    metric,mr=load_metric(pilot,'cpu');state=metric_state_sha(metric)
    base=geometry(root,target)['rgb'];config=read(root/'candidates/single_photometric_depth25/config.json')
    predictions=[];costs=[];known=[];direct=[];direct_known=[];inputs={str(pilot/'summary.json'):sha(pilot/'summary.json')}
    for source in sources:
        d=root/'variants/cubic75/warps'/f'cam{target:03d}'/f'src{source:03d}'
        r=read(d/'receipt.json');x0,y0,x1,y1=r['box'];sl=np.s_[y0:y1,x0:x1]
        a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
        pred=np.array(base);pred[sl]=counterfactual(np.asarray(base[sl]),a,source,config)
        predictions.append(pred)
        for directory,values,validity in [
            (pilot/f'cam{target:03d}'/f'src{source:03d}',costs,known),
            (root/'target_witness_perceptual_projected'/f'cam{target:03d}'/f'src{source:03d}',direct,direct_known)]:
            receipt=read(directory/'receipt.json');assert receipt['box']==r['box']
            value=np.zeros(base.shape[:2],np.float32);value[sl]=np.load(directory/'cost.npy')
            valid=np.zeros(base.shape[:2],bool);valid[sl]=np.load(directory/'count.npy')>=2
            values.append(value);validity.append(valid)
            for p in [directory/'receipt.json',directory/'cost.npy',directory/'count.npy']:
                inputs[str(p)]=sha(p)
        inputs[str(d/'receipt.json')]=sha(d/'receipt.json')
    # Prediction and both confidence variants are complete before evaluation RGB.
    protocol=root/'evaluation/protocol.json';entry=read(protocol)['cameras'][str(target)]
    assert sha(entry['source'])==entry['source_sha256'] and sha(entry['mask'])==entry['mask_sha256']
    truth=np.asarray(Image.open(entry['source']).convert('RGB'),np.float32)/255
    mask=np.asarray(Image.open(entry['mask']))>0
    yy,xx=np.where(mask);x0=max(0,int(xx.min())-16);x1=min(mask.shape[1],int(xx.max())+17)
    y0=max(0,int(yy.min())-16);y1=min(mask.shape[0],int(yy.max())+17);sl=np.s_[y0:y1,x0:x1];m=mask[sl]
    def tensor(value):
        rgb=np.where(m[...,None],value[sl],.5).astype(np.float32)
        return torch.from_numpy(rgb).permute(2,0,1)[None]*2-1
    with torch.inference_mode():
        gt=tensor(truth);errors=[metric(tensor(value),gt)[0,0].numpy() for value in predictions]
    de=errors[0]-errors[1];dc=costs[0][sl]-costs[1][sl];dd=direct[0][sl]-direct[1][sl]
    active=m&known[0][sl]&known[1][sl]
    common=active&direct_known[0][sl]&direct_known[1][sl]
    common &= (np.abs(dc)>.01)&(np.abs(dd)>.01)&(np.abs(de)>.001)
    labels_path=root/'candidates/single_photometric_depth25_seam'/f'cam{target:03d}_sources.npy'
    labels=np.load(labels_path,mmap_mode='r')[sl]
    existing=np.where(labels==sources[0],-1.,1.)
    selection=active&np.isin(labels,sources)
    informative=selection&(np.abs(dc)>.01)&(np.abs(de)>.001)
    result=dict(usage='evaluation_only_no_prediction_feedback',target=target,sources=sources,
        actual_texture=agreement(dc,de,active),
        matched_actual=agreement(dc,de,common),matched_direct=agreement(dd,de,common),
        incremental_actual=agreement(dc,de,informative),
        incremental_existing=agreement(existing,de,informative),
        counterfactual_person_lpips=[float(e[m].mean()) for e in errors],
        inputs={**inputs,str(labels_path):sha(labels_path),str(protocol):sha(protocol),str(mr):sha(mr)},
        code_sha256=sha(Path(__file__)),
        limitation='CPU exploratory source pair only; direct confidence uses CUDA. No GT-selected images or maps are saved; incompatible counterfactual neighborhoods are not an achievable final score.')
    assert metric_state_sha(metric)==state
    write(pilot/'pair_ranking_diagnostic.json',result)
    print({k:v for k,v in result.items() if k!='inputs'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pilot',type=Path,required=True);a=p.parse_args();run(a.root,a.pilot)
