"""Export saved actual GPU occupancy capture, never a threshold reconstruction."""
import argparse,json,os,re,shutil
from pathlib import Path
from types import SimpleNamespace


def captured_model(payload,period):
    import torch
    capture=payload['actual_occupancy_capture'];state=payload['pipeline']
    required=['resolution','aabbs','occs','binaries']
    for key in required:
        if not torch.equal(state['_model.occupancy_grid.'+key],state['_model.adaptive_sampler.occupancy_grid.'+key]):raise ValueError('Saved occupancy owner and sampler aliases differ')
    if capture['radius']!=0 or capture['binary_warmup']:raise ValueError('Expected actual nonwarmup R0 capture')
    if not torch.equal(capture['active'],state['_model.occupancy_grid.binaries']):raise ValueError('Saved native capture differs from active traversal')
    grid=SimpleNamespace(**{key:state['_model.occupancy_grid.'+key]for key in required})
    return SimpleNamespace(config=SimpleNamespace(occupancy_update_interval=period),occupancy_grid=grid,_tuning_occupancy_capture=capture)


def main(c):
    import torch
    from marshawn_tuning_occupancy import export_snapshot,sha,load_snapshot,atomic_json
    expected={'control001':'0e6952376a408744318413f67e9d9ee1fd255bc5777b8a79fece5b23b5387b27','treatment1':'3e422b4b699cab13c42a92811aee3144c06634502f66640c65e7caca7d85ca11'}
    for arm,digest in expected.items():
        run=c/'jobs'/('late_distortion_diagnostic_'+arm)/'run';out=c/'independent_occupancy'/('late_distortion_'+arm+'_064752');out.mkdir(exist_ok=True)
        cp=run/'checkpoints/step-000064752.ckpt';dst=out/'checkpoint.ckpt'
        if not dst.exists():os.link(cp,dst)
        if sha(dst)!=digest:raise ValueError('Protected late-distortion checkpoint differs')
        config=run/'worker/lookcloser/frozen/config.yml';shutil.copy2(config,out/'config.yml')
        periods=re.findall(r'^\s+occupancy_update_interval:\s+(\d+)\s*$',config.read_text(),re.M)
        if len(periods)!=1:raise ValueError('Ambiguous native update period')
        payload=torch.load(dst,map_location='cpu',weights_only=False)
        if payload['step']!=64752:raise ValueError('Wrong matched diagnostic checkpoint step')
        model=captured_model(payload,int(periods[0]));receipt=out/'step-000064752_grid.json'
        if not receipt.exists():export_snapshot(model,dst,64752,out,sources={'config':sha(config),'archive_helper':sha(Path(__file__))})
        load_snapshot(receipt,dst)
        shutil.copytree(run/'renders/step-000064752',out/'renders',dirs_exist_ok=True)
        atomic_json(out/'archive_receipt.json',dict(checkpoint_sha256=digest,step=64752,source=str(cp),immutable_hardlink=True,core_transport='Serialized actual GPU capture in checkpoint; copied to CPU unchanged',EMA_threshold_reconstruction=False,actual_core_update_step=model._tuning_occupancy_capture['step'],owner_sampler_aliases_equal=True,grid_receipt_sha256=sha(receipt),source_config_sha256=sha(config),helper_sha256=sha(Path(__file__))))
        del payload,model
        print(json.dumps(dict(arm=arm,archive=str(out),checkpoint_sha256=digest)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);main(p.parse_args().campaign)
