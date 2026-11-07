"""CPU-only validation of completed HD frequency maps; never assemble a dataset."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def main(root,host,output):
    if output.exists():raise ValueError('Preserve prior validation')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch
    inventory=read(root/'jobs.json');activation=read(root/'activation.json')
    release=read(root/'root_release.json')
    if release.get('activation_sha256')!=sha(root/'activation.json')or release.get('approved_map_fits')is not True:raise ValueError('Exact reviewed activation required')
    for path,digest in read(root/'bindings.json')['hosts'][host].items():
        if sha(path)!=digest:raise ValueError('Protected source or preparation changed: '+path)
    rows=[]
    for job in inventory:
        if job['host']!=host:continue
        spec=read(job['spec']);run=Path(spec['output']);receipt=read(run.parent/'receipt.json')
        if receipt['status']!='complete' or receipt['returncode']!=0:raise ValueError('Fit incomplete: '+job['name'])
        if receipt['gpu_uuid']!=activation['GPU_UUIDs'][host][str(job['gpu'])]:raise ValueError('Different physical GPU')
        complete=read(run/'complete.json');launch=read(run/'launch.json')
        if complete['plan_sha256']!=sha(job['spec']) or launch['plan_sha256']!=sha(job['spec']):raise ValueError('Spec/output lineage mismatch')
        if not launch['controller_owned_lease'] or launch['physical_GPU_UUID']!=receipt['gpu_uuid']:raise ValueError('Fitter lease mismatch')
        camera,variant=job['camera'],job['variant'];folder=run/(camera+'_'+variant)
        fit=read(folder/'complete.json');meta=read(folder/'frequency.json');source=spec['images'][camera]
        if len(complete['rows'])!=1 or complete['rows'][0]!=fit:raise ValueError('One immutable camera/variant result required')
        if fit['source_sha256']!=sha(source) or fit['map_sha256']!=sha(folder/'frequency.pt'):raise ValueError('Map/source changed')
        if fit['updates']!=16000 or fit['optimizer_instances']!=1:raise ValueError('HD1000x16 budget changed')
        threshold,final_lr=(.95,.001)if variant=='lr_decay'else(.93,None)
        if fit['threshold']!=threshold or fit['final_lr']!=final_lr:raise ValueError('Frequency factor changed')
        if abs(fit['lr_first']-.01)>1e-12 or abs(fit['lr_last']-(final_lr or .01))>1e-12:raise ValueError('Actual LR endpoints changed')
        expected=dict(image='train_'+camera+'.png',image_shape=[1012,1920],crop_coords_xywh=None,value_type='scalar_resolution',patch_size=8,stride=8,min_res=16,max_res=8192,n_levels=16,n_features=2,log2_hashmap_size=23)
        for key,value in expected.items():
            if meta.get(key)!=value:raise ValueError('Unexpected metadata '+key)
        if variant=='lr_decay':
            if meta.get('per_level_lr_schedule')!=dict(kind='exponential',initial=.01,final=.001,reset_LR_each_level=True,retain_Adam_state=True):raise ValueError('Per-level LR metadata changed')
        elif 'per_level_lr_schedule'in meta:raise ValueError('Constant-LR arm unexpectedly has decay metadata')
        schedule=16*(8192/16)**(np.arange(16)/15)
        np.testing.assert_allclose(meta['level_resolution_schedule'],schedule,rtol=1e-12,atol=1e-12)
        frequency=torch.load(folder/'frequency.pt',map_location='cpu',weights_only=True)
        if frequency.shape!=(126,240)or frequency.dtype!=torch.float32:raise ValueError('Wrong complete8x8-patch raster/dtype')
        values=frequency.numpy();levels=np.log(values/16)/np.log(8192/16)*15
        if not np.isfinite(levels).all()or np.max(abs(levels-np.rint(levels)))>1e-4:raise ValueError('Map is off the declared level schedule')
        labels=np.rint(levels).astype(int)
        if labels.min()<0 or labels.max()>15:raise ValueError('Level range changed')
        censor=np.load(folder/'censored.npy',allow_pickle=False)
        if censor.shape!=(126,240)or censor.dtype!=np.bool_ or int(censor.sum())!=fit['unresolved']:raise ValueError('Censor map/count mismatch')
        if not (labels[censor]==15).all():raise ValueError('Censored patches must retain final-level label')
        remaining=126*240
        for level in sorted(map(int,fit['counts'])):
            row=fit['counts'][str(level)]
            if row['evaluated']!=remaining or not 0<=row['passed']<=remaining:raise ValueError('SSIM coverage changed')
            remaining-=row['passed']
        if remaining!=fit['unresolved']:raise ValueError('Final unresolved count differs')
        rows.append(dict(name=job['name'],camera=camera,variant=variant,host=host,gpu_uuid=receipt['gpu_uuid'],source_sha256=fit['source_sha256'],files_sha256={str(p):sha(p)for p in [folder/'frequency.pt',folder/'frequency.json',folder/'censored.npy',folder/'complete.json',run/'complete.json',run/'launch.json',run.parent/'receipt.json']},initial_parameters=fit['initial_parameters'],first8_random_draws=fit['first8_random_draws'],final_rng=fit['final_rng'],unresolved=remaining,unresolved_fraction=fit['unresolved_fraction'],level_histogram=np.bincount(labels.reshape(-1),minlength=16).tolist(),seconds=fit['seconds']))
    expected_count=sum(j['host']==host for j in inventory)
    if len(rows)!=expected_count or len({(r['camera'],r['variant'])for r in rows})!=len(rows):raise ValueError('Incomplete or duplicate queue')
    assert not torch.cuda.is_initialized()
    output.write_text(json.dumps(dict(passed=True,host=host,rows=rows,activation_sha256=sha(root/'activation.json'),CPU_only=True,uncovered_bottom_rows=4,assembly_performed=False,READY_written=False,NeRF_training_authorized=False,quality_or_geometry_promotion=False),indent=2)+'\n')
    print(json.dumps(dict(passed=True,host=host,jobs=len(rows),validation_sha256=sha(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--host',choices=['local','silver'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.host,a.output)
