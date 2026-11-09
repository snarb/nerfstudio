"""Post-render fixedV1 metrics and source guards; never a fit input or optimizer."""
import argparse
import json
import os
from pathlib import Path
from prepare_marshawn_frozen_semantic_cpu import sha, write


def run(spec_path, execute):
    s = json.loads(spec_path.read_text())
    if not execute:
        print(json.dumps(dict(phase='post_render_metrics_and_visual_gate', spec_sha256=sha(spec_path),
            fit=False, native_updates=0, optimizer_created=False, CUDA_started=False)))
        return
    release = json.loads(Path(s['root_release']).read_text())
    if release.get('reviewer') != 'root' or release.get('LC_physical_metrics_authorized') is not True or release['spec_sha256'].get(str(spec_path)) != sha(spec_path):
        raise ValueError('Exact root post-render evaluation release required')
    for path, digest in release['bound_files'].items():
        if sha(path) != digest: raise ValueError('Bound file changed: '+path)
    from marshawn_gpu_isolation import require_controller_lease
    lease = require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from scipy.ndimage import binary_dilation
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from marshawn_tuning_metrics import masked_actor_metrics, aggregate_actor_metrics
    from run_marshawn_frozen_semantic_probe import capture_health
    torch.set_num_threads(2); torch.cuda.set_device(0)
    if torch.cuda.device_count() != 1: raise ValueError('Exact physical device lease required')
    render = Path(s['renders']); completion = json.loads((render/'complete.json').read_text())
    if not all(completion['source_unchanged'].values()) or not completion['sigma_callback_is_None'] or not completion['actual_transmittance_recomputed']:
        raise ValueError('Physical render preservation/traversal gate failed')
    out = Path(s['output']); out.mkdir(parents=True, exist_ok=False)
    ready_path = Path(s['ready']); ready = json.loads(ready_path.read_text()); rows = {r['camera']: r for r in ready['records']}
    regions = json.loads(Path(s['regions']).read_text())
    names = ['identity_inside', 'soft_probability', 'hard_theta01', 'hard_theta05', 'exactzero_smoothstep01_05', 'native_weight_semantic_partition', 'native_full']
    lp = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda().eval().net.eval()
    records = {k: [] for k in names}; exterior = []; sources = []; actor_panels = []; detail_panels = []
    with torch.inference_mode():
        for name in ['DC', 'HC', 'KA']:
            r = rows[name]; target_path = ready_path.parent/r['image']
            if sha(target_path) != r['rgb_sha256'] or sha(r['semantic']) != r['semantic_sha256']:
                raise ValueError('Frozen eval-only source RGB/mask differs')
            sources.append(dict(camera=name, target=str(target_path), target_sha256=r['rgb_sha256'], semantic_sha256=r['semantic_sha256']))
            source_image = Image.open(target_path).convert('RGB')
            target = torch.tensor(np.asarray(source_image).copy()/255, dtype=torch.float32, device='cuda').permute(2, 0, 1)[None]
            archive = render/name/'physical_and_partition.npz'
            expected = next(v['physical_archive_sha256'] for v in completion['records'] if v['camera'] == name)
            if sha(archive) != expected: raise ValueError('Physical output archive differs')
            with np.load(r['semantic']) as masks:
                actor_masks = {actor: masks[actor].copy() for actor in ['black', 'grey']}
            exterior_mask = ~binary_dilation(actor_masks['black'] | actor_masks['grey'], iterations=3)
            with np.load(archive) as values:
                for variant in names:
                    rgba = values[variant]
                    prediction = torch.tensor(np.clip(rgba[..., :3], 0, 1).copy(), device='cuda').permute(2, 0, 1)[None]
                    for actor, mask in actor_masks.items():
                        measured = masked_actor_metrics(prediction, target, torch.tensor(mask, device='cuda')[None, None], lp, crop_xyxy=r['crops'][actor])
                        records[variant].append(dict(view=name, actor=actor, **measured))
                    exterior.append(dict(view=name, variant=variant,
                        alpha_gt01_outside_fixed_actor_dilation3=int(((rgba[..., 3] > .1) & exterior_mask).sum()),
                        meaning='Frozen-mask exterior diagnostic, not opticalalphaGT; sourcebagmask contamination remains.'))
            images = [source_image]+[Image.open(render/name/(v+'.png')).convert('RGB') for v in names[:6]]
            labels = ['SOURCE', 'box_only_nohead', 'soft_sigma_p', 'hard_p_ge_.1', 'hard_p_ge_.5', 'smooth_.1_to_.5', 'native_weight_partition_DIAG']
            for region, crop in regions[name].items():
                tiles = [im.crop(crop) for im in images]; w, h = tiles[0].size
                panel = Image.new('RGB', (len(tiles)*w, h+36), '#171717'); draw = ImageDraw.Draw(panel)
                for i, (tile, label) in enumerate(zip(tiles, labels)):
                    panel.paste(tile, (i*w, 36)); draw.text((i*w+2, 2), name+' '+region, fill='white'); draw.text((i*w+2, 18), label, fill='white')
                panel.save(out/(name+'_'+region+'.png')); panel.thumbnail((2000, 430)); detail_panels.append(panel)
            crops = list(r['crops'].values()); crop = [max(0, min(v[0] for v in crops)-48), max(0, min(v[1] for v in crops)-32), min(1920, max(v[2] for v in crops)+48), min(1012, max(v[3] for v in crops)+48)]
            actor_panel = Image.new('RGB', (7*300, 520), '#171717'); draw = ImageDraw.Draw(actor_panel)
            for i, (im, label) in enumerate(zip(images, labels)):
                tile = im.crop(crop); tile.thumbnail((295, 485)); actor_panel.paste(tile, (i*300+(295-tile.width)//2, 35)); draw.text((i*300+2, 3), name+' '+label, fill='white')
            actor_panels.append(actor_panel)
            write(out/'progress.json', dict(camera=name, completed_views=len(actor_panels), health=capture_health(torch, lease['expected_physical_uuid'])))
    # The frozen helper computes an internal stopping statistic as well. Export
    # PSNR/SSIM/LPIPS only, as required by the evaluation reporting contract.
    def reporting_only(value):
        if isinstance(value, dict): return {k: reporting_only(v) for k, v in value.items() if 'loss' not in k.lower()}
        if isinstance(value, list): return [reporting_only(v) for v in value]
        return value
    measured = {variant: reporting_only(aggregate_actor_metrics(rows)) for variant, rows in records.items()}
    baseline = json.loads(Path(s['native_baseline_evaluation']).read_text())['metrics']
    delta = {key: measured['native_full']['metrics'][key]-baseline[key] for key in ['psnr', 'ssim', 'lpips']}
    for key, tolerance in [('psnr', .003), ('ssim', .0001), ('lpips', .0002)]:
        if abs(delta[key]) > tolerance: raise ValueError('Direct-LC native-source metrics differ from selected126316 baseline: '+key+' '+str(delta[key]))
    by_view_reference=json.loads(Path(s['native_baseline_evaluation']).read_text())['by_view']
    per_view_deltas={}
    for camera in ['DC','HC','KA']:
        source_view=[r for r in records['native_full'] if r['view']==camera]
        per_view_deltas[camera]={k:float(np.mean([r[k]for r in source_view]))-by_view_reference[camera][k]for k in ['psnr','ssim','lpips']}
        for key,tolerance in [('psnr',.003),('ssim',.0001),('lpips',.0002)]:
            if abs(per_view_deltas[camera][key])>tolerance:raise ValueError('Per-view selected126316 source metric parity failed: '+camera+' '+key)
    identity_box_vs_source={k:measured['identity_inside']['metrics'][k]-measured['native_full']['metrics'][k]for k in ['psnr','ssim','lpips']}
    for filename, panels in [('all_actors.jpg', actor_panels), ('all_details.jpg', detail_panels)]:
        canvas = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels)), '#171717'); y = 0
        for panel in panels: canvas.paste(panel, (0, y)); y += panel.height
        canvas.save(out/filename, quality=94)
    write(out/'metrics.json', dict(variants=measured, exterior_diagnostics=exterior, nohead_baseline_metric_delta=delta,
        source_bindings=sources,per_view_source_metric_deltas=per_view_deltas,identity_box_vs_source_metric_deltas=identity_box_vs_source,
        path_mode=completion['path_mode'],full_visibility_certified=False, ready_sha256=sha(ready_path), region_crops_sha256=sha(s['regions']),
        render_completion_sha256=sha(render/'complete.json'), source_EVAL_RGB_and_masks_used_for='Post-render evaluation ONLY; no native/head model or optimizer instantiated.',
        selectors_automatically_promoted=False, mandatory_root_visual_review=True,
        semantic_coverage_is_not_optical_alpha_GT=True, metric_caveat='FixedV1 nominal actor labels include mixed boundaries and knownbag contamination; cleanup may worsen those pixels despite better physicalisolation.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--spec', type=Path, required=True); p.add_argument('--execute', action='store_true')
    a = p.parse_args(); run(a.spec, a.execute)
