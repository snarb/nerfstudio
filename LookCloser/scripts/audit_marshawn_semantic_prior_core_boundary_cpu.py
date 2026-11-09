"""Existing fixed-region PNG diagnostic for semantic-prior gates; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
from audit_marshawn_core_boundary_metrics_cpu import regions
from prepare_marshawn_frozen_semantic_cpu import sha, write


def run(c, output):
    import torch
    import torchmetrics
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from marshawn_tuning_metrics import masked_actor_metrics
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    prior99 = c/'frozen_native_semantic_render_recovery_v1'
    prior01 = c/'frozen_native_semantic_prior001_candidate_v1'
    arms = {'box_nohead': (prior99, 'identity_inside'), 'p01_hard01': (prior01, 'hard_theta01'),
            'p01_hard05': (prior01, 'hard_theta05'), 'p99_hard05': (prior99, 'hard_theta05')}
    ready_path = c/'data/ready.json'; ready = {r['camera']: r for r in json.loads(ready_path.read_text())['records']}
    region_path = c/'native_semantic_grid_20000/corrected_v2/review/region_crops.json'
    guard_regions = json.loads(region_path.read_text())
    net = LearnedPerceptualImagePatchSimilarity(normalize=True).net.eval().cpu()
    assert net.pnet_type == 'alex' and net.resize is None
    records, mask_receipts, body_panels, detail_panels = [], [], [], []
    bindings = {str(p): sha(p) for p in [ready_path, region_path]}
    for cam in ['DC', 'HC', 'KA']:
        row = ready[cam]; source_path = c/'data'/row['image']; mask_path = Path(row['semantic'])
        assert sha(source_path) == row['rgb_sha256'] and sha(mask_path) == row['semantic_sha256']
        bindings.update({str(p): sha(p) for p in [source_path, mask_path]})
        source = Image.open(source_path).convert('RGB')
        target = torch.from_numpy(np.asarray(source, np.float32).copy()/255).permute(2, 0, 1)[None]
        with np.load(mask_path) as z: masks = {actor: regions(z[actor]) for actor in ['black', 'grey']}
        images, labels = [source], ['SOURCE']
        for actor, rs in masks.items():
            for region, mask in rs.items():
                mask_receipts.append(dict(camera=cam, actor=actor, region=region, pixels=int(mask.sum()),
                    mask_sha256=__import__('hashlib').sha256(mask.tobytes()).hexdigest(), crop=row['crops'][actor]))
        for arm, (root, variant) in arms.items():
            path = root/'renders'/cam/(variant+'.png'); bindings[str(path)] = sha(path)
            image = Image.open(path).convert('RGB'); assert image.size == (1920, 1012)
            images.append(image); labels.append(arm)
            pred = torch.from_numpy(np.asarray(image, np.float32).copy()/255).permute(2, 0, 1)[None]
            for actor, rs in masks.items():
                for region, mask in rs.items():
                    with torch.inference_mode():
                        metric = masked_actor_metrics(pred, target, torch.from_numpy(mask.copy())[None, None], net, crop_xyxy=row['crops'][actor])
                    records.append(dict(camera=cam, actor=actor, region=region, arm=arm,
                        **{key: metric[key] for key in ['psnr', 'ssim', 'lpips']}, pixels=int(mask.sum()), crop_xyxy=row['crops'][actor]))
            write(output/'progress.json', dict(camera=cam, arm=arm, records=len(records), seconds=time.monotonic()-started))
        for region, crop in guard_regions[cam].items():
            tiles = [im.crop(crop) for im in images]; w, h = tiles[0].size
            panel = Image.new('RGB', (len(tiles)*w, h+35), '#171717'); draw = ImageDraw.Draw(panel)
            for i, (tile, label) in enumerate(zip(tiles, labels)):
                panel.paste(tile, (i*w, 35)); draw.text((i*w+2, 2), cam+' '+region, fill='white'); draw.text((i*w+2, 17), label, fill='white')
            panel.save(output/(cam+'_'+region+'.png')); panel.thumbnail((1800, 430)); detail_panels.append(panel)
        crops = list(row['crops'].values()); crop = [max(0, min(v[0] for v in crops)-48), max(0, min(v[1] for v in crops)-32), min(1920, max(v[2] for v in crops)+48), min(1012, max(v[3] for v in crops)+48)]
        panel = Image.new('RGB', (5*330, 560), '#171717'); draw = ImageDraw.Draw(panel)
        for i, (image, label) in enumerate(zip(images, labels)):
            tile = image.crop(crop); tile.thumbnail((325, 520)); panel.paste(tile, (i*330+(325-tile.width)//2, 35)); draw.text((i*330+2, 3), cam+' '+label, fill='white')
        body_panels.append(panel)
    aggregate = {region: {arm: {key: float(np.mean([r[key] for r in records if r['region'] == region and r['arm'] == arm])) for key in ['psnr', 'ssim', 'lpips']} for arm in arms} for region in ['whole_PNG_control', 'core_gt8px', 'boundary_le8px']}
    for filename, panels in [('all_actors_fixed_guard.jpg', body_panels), ('all_details_fixed_guard.jpg', detail_panels)]:
        canvas = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels)), '#171717'); y = 0
        for panel in panels: canvas.paste(panel, (0, y)); y += panel.height
        canvas.save(output/filename, quality=94)
    with (output/'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['camera', 'actor', 'region', 'arm', 'psnr', 'ssim', 'lpips', 'pixels', 'crop_xyxy']); writer.writeheader(); writer.writerows(records)
    assert not torch.cuda.is_initialized()
    write(output/'metrics.json', dict(kind='EXISTING_FIXED_CORE_BOUNDARY_PNG_CPU_DIAGNOSTIC_NOT_PRIMARY_RERANK', aggregate=aggregate,
        records=records, mask_receipts=mask_receipts, inputs_sha256=bindings, region_helper_sha256=sha(Path(__file__).with_name('audit_marshawn_core_boundary_metrics_cpu.py')),
        metric_helper_sha256=sha(Path(__file__).with_name('marshawn_tuning_metrics.py')), helper_sha256=sha(__file__),
        torch_version=torch.__version__, torchmetrics_version=torchmetrics.__version__, CUDA_initialized=False, labels_changed=False,
        primary_V1_metrics_unchanged=True, seconds=time.monotonic()-started,
        definition='Exactexistingregions(): EuclideanEDT>8HDpixels with image exteriorfalse; boundaryoriginalmask minuscore; sameoriginalsealedcrop, Alex/noresize/.5neutralexterior. AllarmsPNG8, wholePNGcontrol included.',
        limitations=['Regionmasks are not opticalalphaGT or newfitlabels.','Core excludes thinhair/contours and cannot certify them; wrongbaglabels mayremainincore.','Onlycomparearms within a region; LPIPSpartitions are not additive. PNGquantization differs from primaryfloatmetrics.']))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--campaign', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.campaign, a.output)
