"""Rebind only the interrupted SAME frame1670 checkpoint to the capped clone.

No model/optimizer/grid/RNG tensors change. Other frames must initialize cold.
Run on silver with its qualified Python; input/output campaigns stay separate.
"""
from pathlib import Path
import copy, hashlib, json, os, time

OLD=Path('/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_20261107')
NEW=Path('/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_140k_20261107')
FRAME=1670

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)

def tensor_state_hash(payload):
    import torch,numpy as np
    h=hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            h.update(('tensor:'+str(x.dtype)+':'+str(tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):
            h.update(('numpy:'+str(x.dtype)+':'+str(x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for k in sorted(x,key=lambda y:str(y)):add(k);add(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for v in x:add(v)
        else:h.update((type(x).__name__+':'+repr(x)).encode())
    for key in sorted(k for k in payload if k!='tuning_runtime'):add(key);add(payload[key])
    runtime=payload['tuning_runtime']
    for key in sorted(k for k in runtime if k not in {'spec_sha256','code_sha256'}):add(key);add(runtime[key])
    return h.hexdigest()

def migrate():
    import torch
    old=OLD/'frames'/('%06d'%FRAME);new=NEW/'frames'/('%06d'%FRAME)
    old_spec=json.loads((old/'spec.json').read_text());new_spec=json.loads((new/'spec.json').read_text())
    assert new_spec['hard_max_iterations']==140000
    def normalized(d):
        d=copy.deepcopy(d);d.pop('hard_max_iterations',None)
        d['data']='SAME_FRAME_DATA';d['output']='SAME_FRAME_OUTPUT'
        d['semantic_admission']['manifest']='SAME_FRAME_ADMISSION'
        d['semantic_admission']['release_receipt']='VERSIONED_RELEASE'
        return d
    assert normalized(old_spec)==normalized(new_spec),'Non-budget training configuration changed'
    assert sha(old/'data/ready.json')==sha(new/'data/ready.json')
    assert sha(old/'data/admission/manifest.json')==sha(new/'data/admission/manifest.json')
    assert json.loads((old/'train/cold_provenance.json').read_text())['frame']==FRAME
    cold=json.loads((old/'train/cold_setup_audit.json').read_text())
    assert cold['cold']and cold['occupied_cells']==0 and cold['nonzero_ema_cells']==0 and not any(cold['optimizer_state_sizes'].values())
    output=new/'train';output.mkdir(exist_ok=True);assert not(output/'latest.json').exists()
    assert (output/'cold_provenance.json').exists(),'New native binder must create frame provenance first'
    old_latest=json.loads((old/'train/latest.json').read_text());resume_step=old_latest['step']
    assert 0<=resume_step<139999,'Same-frame resume must be under the new hard cap'
    old_seal=json.loads((OLD/'code_seal.json').read_text());new_seal=json.loads((NEW/'code_seal.json').read_text())
    changed={k for k,v in old_seal['files_sha256'].items()if new_seal['files_sha256'][k]!=v}
    assert changed=={'LookCloser/scripts/train_marshawn_semantic_admission.py','LookCloser/scripts/marshawn_video_train.py'}
    checkpoints=sorted((old/'train/checkpoints').glob('step-?????????.ckpt'))
    receipt=dict(frame=FRAME,old_campaign=str(OLD),new_campaign=str(NEW),same_frame_only=True,
        resume_step=resume_step,hard_max_iterations=140000,host='silver',gpu=1,
        old_spec_sha256=sha(old/'spec.json'),new_spec_sha256=sha(new/'spec.json'),
        identical_data_ready_sha256=sha(new/'data/ready.json'),old_cold_setup_sha256=sha(old/'train/cold_setup_audit.json'),
        source_cold_origin_verified=True,created=time.time(),checkpoints=[])
    for source in checkpoints:
        step=int(source.stem.split('-')[1]);assert step<=resume_step<140000
        source_sha=sha(source);loaded=torch.load(source,map_location='cpu',weights_only=False)
        assert loaded['step']==step
        state=loaded['tuning_runtime']
        assert state['spec_sha256']==receipt['old_spec_sha256'] and state['data_sha256']==receipt['identical_data_ready_sha256']
        for name,digest in state['code_sha256'].items():
            candidates=[p for p in (NEW/'code').rglob(name)if p.is_file()]
            assert len(candidates)==1,(name,candidates)
            if name!='train_marshawn_semantic_admission.py':assert sha(candidates[0])==digest,name
        for name,digest in state['input_sha256'].items():
            if name=='semantic_manifest':target=new/'data/admission/manifest.json'
            elif name.startswith('semantic/'):target=new/'data/admission'/name[len('semantic/'):]
            else:target=new/'data'/name
            assert sha(target)==digest,name
        before=tensor_state_hash(loaded)
        state['spec_sha256']=receipt['new_spec_sha256']
        state['code_sha256']['train_marshawn_semantic_admission.py']=sha(NEW/'code/LookCloser/scripts/train_marshawn_semantic_admission.py')
        assert tensor_state_hash(loaded)==before
        dest=output/'checkpoints'/source.name;dest.parent.mkdir(exist_ok=True);temp=dest.with_suffix('.tmp')
        torch.save(loaded,temp);os.replace(temp,dest);del loaded
        reloaded=torch.load(dest,map_location='cpu',weights_only=False)
        assert tensor_state_hash(reloaded)==before;del reloaded
        receipt['checkpoints'].append(dict(step=step,source=str(source),source_sha256=source_sha,
            destination=str(dest),destination_sha256=sha(dest),unchanged_tensor_optimizer_grid_rng_state_sha256=before))
    chosen=next(v for v in receipt['checkpoints']if v['step']==resume_step)
    write(output/'latest.json',dict(checkpoint='checkpoints/'+Path(chosen['destination']).name,
        checkpoint_sha256=chosen['destination_sha256'],step=resume_step,terminal=False))
    for name in ['mode.json']:
        write(output/name,json.loads((old/'train'/name).read_text()))
    shutil=__import__('shutil')
    if (old/'train/renders').exists():shutil.copytree(old/'train/renders',output/'renders',dirs_exist_ok=True)
    path=NEW/'operations/same_frame_resume'/('%06d.json'%FRAME)
    receipt['authorized']=True;receipt['tensor_optimizer_grid_rng_bitwise_unchanged']=True
    write(path,receipt)
    print(json.dumps(dict(receipt=str(path),frame=FRAME,resume_step=resume_step,checkpoint_count=len(receipt['checkpoints']),unchanged_tensors=True)))

if __name__=='__main__':migrate()
