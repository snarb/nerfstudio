from pathlib import Path
from types import SimpleNamespace
import csv,json,sys
import pytest,yaml
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from run_marshawn_patch_dssim_phase import choose_parent_and_phase,exact_eval_config


@pytest.mark.parametrize('phase_psnr,phase_lpips,winner',[(28.65,.32,'parent'),(28.69,.30,'phase'),(29.0,.34,'phase')])
def test_parent_and_phase_selection_and_exact_checkpoint(tmp_path,phase_psnr,phase_lpips,winner):
    parent=tmp_path/'immutable_parent'/'step-000008000.ckpt';parent.parent.mkdir();parent.write_bytes(b'parent')
    phase=tmp_path/'phase'/'step-000010000.ckpt';phase.parent.mkdir();phase.write_bytes(b'phase')
    metrics=tmp_path/'metrics.csv'
    with metrics.open('w')as f:
        w=csv.DictWriter(f,fieldnames=['step','eval_all_psnr','eval_all_ssim','eval_all_lpips']);w.writeheader()
        w.writerow(dict(step=10000,eval_all_psnr=phase_psnr,eval_all_ssim=.8,eval_all_lpips=phase_lpips))
    spec=dict(parent_candidate=dict(checkpoint=str(parent),step=8000,psnr=28.7,ssim=.797,lpips=.3177),
              parent_checkpoint_sha256='test',selection_receipt=str(tmp_path/'selection.json'))
    checkpoint,_=choose_parent_and_phase(metrics,phase.parent,spec)
    assert checkpoint==(parent if winner=='parent'else phase)
    assert not (phase.parent/'step-000008000.ckpt').exists()
    cfg=SimpleNamespace(load_checkpoint=parent,load_dir=phase.parent,load_step=10000,load_config='stale',
                        pipeline=SimpleNamespace(model=SimpleNamespace(eval_num_rays_per_chunk=999)))
    config=tmp_path/'config.yml';config.write_text(yaml.dump(cfg))
    bound=yaml.load(exact_eval_config(config,checkpoint,512).read_text(),Loader=yaml.Loader)
    assert bound.eval_checkpoint==checkpoint.resolve()
    assert bound.load_checkpoint is None and bound.load_dir is None and bound.load_step is None and bound.load_config is None
    assert bound.pipeline.model.eval_num_rays_per_chunk==512
