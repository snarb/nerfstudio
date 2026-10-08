#!/usr/bin/env python3
"""Evaluate spatial depth admission on a frozen checkpoint before any new training."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from run_luster_opacity import configuration, configure_runtime, evaluate, sha, write


def field_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.field.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(json.dumps(dict(feature_reweighting_strength=model.field.feature_reweighting_strength,
                                  enable_feature_reweighting=model.field.enable_feature_reweighting,
                                  density_activation=model.field.density_activation),sort_keys=True).encode())
    return digest.hexdigest()


def screen(selection_path, manifest_path, output):
    if output.exists():
        raise FileExistsError('Preserve earlier forward screens')
    selected = json.loads(selection_path.read_text())
    selected = selected.get('selected', selected)
    checkpoint = Path(selected['checkpoint'])
    source = checkpoint.parents[4]
    request = deepcopy(json.loads((source/'request.json').read_text()))
    request.update(output=str(output),resume=str(checkpoint),end_step=selected['step'],eval_steps=[selected['step']])
    request['model']['occupancy_prior_grid_path'] = str(manifest_path)
    configure_runtime(request)
    output.mkdir(parents=True)
    write(output/'request.json',request)
    cfg = configuration(request)
    cfg.save_config()
    trainer = cfg.setup(local_rank=0,world_size=1)
    trainer.setup()
    model = trainer.pipeline.model
    model.current_train_step = selected['step']
    before = field_fingerprint(model)
    row = evaluate(trainer,request,selected['step'])
    after = field_fingerprint(model)
    if after != before:
        raise ValueError('Forward-only admission screen changed field geometry')
    row['checkpoint'] = str(checkpoint)
    row['occupancy_admission_manifest'] = str(manifest_path)
    write(Path(row['renders'])/'metrics.json',row)
    write(output/'result.json',dict(scope='Forward-only frozen-checkpoint screen; no optimizer steps or new trained checkpoint',
          checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),prior_manifest=str(manifest_path),
          field_before=before,field_after=after,occupied_prior_cells=int(model.occupancy_prior_grid.sum()),
          metrics={key:row[key] for key in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']},
          visual_review='pending'))
    print(json.dumps({key:row[key] for key in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    screen(args.selection,args.manifest,args.output)
