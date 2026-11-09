#!/usr/bin/env python3
"""Progressive report of measured Marshawn AABB/halo runs; never starts training.

--report-only writes JSON/CSV/Markdown without importing Pillow or torch.
--html additionally writes portable thumbnails, comparison sheets and index.html.
--score-rois measures fixed unmasked saved-PNG rectangles (not semantic masks).
ROI JSON maps physical camera IDs to {mainpair:[x0,y0,x1,y1],background:[...]},
in native prepared-image pixels. The same rectangles apply to every branch.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
from datetime import datetime, timezone

METRICS = ('psnr', 'ssim', 'lpips')
DEFAULT_ROIS = {
    'DC': {'mainpair': [700, 60, 1190, 850], 'background': [80, 70, 620, 400]},
    'HC': {'mainpair': [830, 100, 1320, 870], 'background': [80, 80, 700, 410]},
    'KA': {'mainpair': [790, 80, 1300, 780], 'background': [1350, 140, 1850, 600]},
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def finite_metrics(source):
    try:
        result = {key: float(source[key]) for key in METRICS}
    except (KeyError, TypeError, ValueError):
        return None
    return result if all(math.isfinite(value) for value in result.values()) else None


def branch_directory(spec):
    return Path(spec['output_dir']) / spec['name'] / 'lookcloser' / 'cold_seed42'


def load_views(data):
    metadata = read(Path(data) / 'transforms.json')
    filenames = set(metadata['val_filenames'])
    return [dict(index=index, camera=frame.get('physical_camera', Path(frame['file_path']).stem),
                 file_path=frame['file_path'], width=frame['w'], height=frame['h'])
            for index, frame in enumerate(sorted((f for f in metadata['frames'] if f['file_path'] in filenames),
                                                 key=lambda f: f['file_path']))]


def collect(root):
    root = Path(root)
    manifest = read(root / 'campaign.json')
    if manifest.get('transforms_sha256') and digest(Path(manifest['data'])/'transforms.json') != manifest['transforms_sha256']:
        raise ValueError('prepared transforms changed since this campaign was initialized')
    views = load_views(manifest['data'])
    rows = []
    for spec_path in manifest['variants']:
        spec = read(spec_path)
        box = spec['aabb']
        extent = [box[1][k] - box[0][k] for k in range(3)]
        row = dict(name=spec['name'], scope=spec['scope'], size=spec['size'], radius=spec['radius'],
                   aabb=box, aabb_volume=math.prod(extent), aabb_extent=extent,
                   run_directory=str(branch_directory(spec)), checkpoint=None, metrics=None,
                   renders={}, status='incomplete', warnings=[], visual_review='pending')
        directory = branch_directory(spec)
        summary = directory / 'run_summary.json'
        status_file = root / 'status' / (spec['name'] + '.json')
        if status_file.is_file():
            try:
                status = read(status_file)
                row['controller_status'] = status.get('status')
                row['visual_review'] = status.get('visual_review', 'pending')
            except (ValueError, OSError) as error:
                row['warnings'].append(f'Unreadable status: {error}')
        if not summary.is_file():
            row['warnings'].append('No completed run_summary.json; no final metrics available.')
            rows.append(row)
            continue
        row['summary_path'] = str(summary)
        row['summary_sha256'] = digest(summary)
        try:
            payload = read(summary)
            evaluation = payload.get('eval') or {}
            selected = payload.get('selected_checkpoint')
            evaluated = evaluation.get('checkpoint')
            if not selected or not evaluated or Path(selected).resolve() != Path(evaluated).resolve():
                raise ValueError('selected checkpoint differs from the evaluated checkpoint')
            match = re.fullmatch(r'step-(\d+)\.ckpt', Path(selected).name)
            if not match or int(match.group(1)) >= 22000:
                raise ValueError('checkpoint is missing an in-budget step ID')
            row['checkpoint'] = selected
            row['checkpoint_id'] = Path(selected).name
            row['checkpoint_step'] = int(match.group(1))
            if not Path(selected).is_file():
                row['warnings'].append('Selected checkpoint file is missing.')
            row['metrics'] = finite_metrics(evaluation.get('results'))
            if row['metrics'] is None:
                raise ValueError('PSNR, SSIM and LPIPS must all be finite measured evaluator results')
            row['metric_source'] = evaluation.get('eval_json', str(summary))
            row['selection'] = payload.get('selected_checkpoint_reason')
            render_dir = Path(evaluation['render_dir'])
            for view in views:
                render = render_dir / f"eval_img_{view['index']:04d}.png"
                if render.is_file():
                    row['renders'][view['camera']] = str(render)
                else:
                    row['warnings'].append(f"Missing RGB comparison for {view['camera']}: {render}")
            if payload.get('train_returncode') not in (0, -2):
                row['warnings'].append('Training process did not finish with an accepted return code.')
            if payload.get('train_returncode') == -2 and payload.get('stop_reason') != 'plateau':
                row['warnings'].append('Interrupted training is not a completed comparison branch.')
            row['status'] = 'evaluated' if not row['warnings'] else 'incomplete'
        except (OSError, KeyError, ValueError, TypeError) as error:
            row['warnings'].append(str(error))
            row['status'] = 'invalid_summary'
        rows.append(row)
    return dict(generated_utc=datetime.now(timezone.utc).isoformat(), campaign=str(root.resolve()),
        report_generator_sha256=digest(__file__),
        data=manifest['data'], transforms_sha256=manifest.get('transforms_sha256'), views=views, branches=rows,
        definitions=dict(full_frame='Measured evaluator PSNR/SSIM/LPIPS from run_summary.eval.results.',
            roi='Unmasked, fixed native rectangles from saved 8-bit GT|prediction PNG panels; separate from floating full-frame metrics.',
            background='A fixed region outside the main pair; this is not a certified empty-space mask.',
            visual_review='Images and occupancy still require explicit manual review; generating this report does not pass that gate.'),
        warnings=[f"{sum(r['status'] != 'evaluated' for r in rows)}/{len(rows)} branches incomplete or invalid."])


def panels(path, expected_width, expected_height):
    from PIL import Image
    with Image.open(path) as source:
        source = source.convert('RGB')
        if source.size != (2 * expected_width, expected_height):
            raise ValueError(f'expected exact GT|prediction raster {(2*expected_width,expected_height)}, got {source.size}: {path}')
        return source.crop((0, 0, expected_width, expected_height)), source.crop((expected_width, 0, 2*expected_width, expected_height))


def validate_roi(box, width, height):
    if len(box) != 4 or any(type(value) is not int for value in box):
        raise ValueError('ROI must contain four integer native coordinates')
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('ROI extends outside the native image')
    if x1 - x0 < 64 or y1 - y0 < 64:
        raise ValueError('ROI must be at least 64x64 for LPIPS; no resizing is allowed')
    return box


class RoiScorer:
    def __init__(self, device):
        import torch
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from torchmetrics.functional.image import structural_similarity_index_measure
        self.torch = torch
        self.device = device
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval()

    def score(self, gt, prediction):
        import numpy as np
        torch = self.torch
        def tensor(image):
            return torch.from_numpy(np.array(image, copy=True)).permute(2,0,1).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.
        a, b = tensor(gt), tensor(prediction)
        with torch.inference_mode():
            mse = (a-b).square().mean()
            # Perfect PNG equality is reported explicitly instead of serializing infinity.
            psnr = None if float(mse) == 0 else float(-10 * torch.log10(mse))
            value = dict(psnr=psnr, ssim=float(self.ssim(a,b,data_range=1.)), lpips=float(self.lpips(a,b)))
        self.lpips.reset()
        if any(v is not None and not math.isfinite(v) for v in value.values()):
            raise ValueError('ROI metric returned a nonfinite value')
        if psnr is None:
            value['perfect_pixel_match'] = True
        return value


def make_visuals(report, output, rois, html_requested, score_requested, device):
    from PIL import Image, ImageDraw
    assets = output / 'assets'
    assets.mkdir(exist_ok=True)
    if score_requested:
        for view in report['views']:
            if any(region not in rois.get(view['camera'],{}) for region in ('mainpair','background')):
                raise ValueError(f"--score-rois requires mainpair and background rectangles for {view['camera']}")
    scorer = RoiScorer(device) if score_requested else None
    sheets = []
    baseline_hash = {}
    for view in report['views']:
        camera = view['camera']
        with Image.open(Path(report['data']) / view['file_path']) as source_gt:
            expected_gt_sha = hashlib.sha256(source_gt.convert('RGB').tobytes()).hexdigest()
        regions = {'full': None, **rois.get(camera, {})}
        for region, box in regions.items():
            if box is not None:
                validate_roi(box, view['width'], view['height'])
            tiles = []
            for row in report['branches']:
                path = row['renders'].get(camera)
                if not path:
                    continue
                try:
                    gt, pred = panels(path, view['width'], view['height'])
                    gt_sha = hashlib.sha256(gt.tobytes()).hexdigest()
                    if gt_sha != expected_gt_sha:
                        raise ValueError('GT panel differs from the prepared camera image; verify eval camera indexing')
                    previous = baseline_hash.setdefault(camera, gt_sha)
                    if previous != gt_sha:
                        raise ValueError('GT panel differs across branches; comparison is invalid')
                    if box is not None:
                        gt, pred = gt.crop(box), pred.crop(box)
                    if scorer is not None and box is not None:
                        metrics = scorer.score(gt,pred)
                        row.setdefault('roi_metrics', {}).setdefault(camera,{})[region] = dict(
                            rectangle=box, **metrics, source_render=path, source_render_sha256=digest(path), gt_panel_sha256=gt_sha)
                    if html_requested:
                        if box is not None:
                            native=Image.new('RGB',(2*gt.width,gt.height))
                            native.paste(gt,(0,0));native.paste(pred,(gt.width,0))
                            native_name=re.sub(r'[^A-Za-z0-9_.-]','_',f"{row['name']}_{camera}_{region}_native")+'.png'
                            native.save(assets/native_name)
                            row.setdefault('native_crops',{}).setdefault(camera,{})[region]='assets/'+native_name
                        if not tiles:
                            tiles.append(('GT', gt.copy()))
                        tiles.append((row['name'], pred.copy()))
                except (OSError, ValueError, RuntimeError) as error:
                    row['warnings'].append(f'{camera}/{region}: {error}')
                    report['warnings'].append(f"Image comparison/scoring failed: {row['name']} {camera}/{region}")
            if tiles and html_requested:
                columns = min(4,len(tiles))
                width = 300
                height = max(round(image.height * width / image.width) for _,image in tiles)
                canvas = Image.new('RGB',(columns*width,math.ceil(len(tiles)/columns)*(height+28)),(25,25,25))
                draw = ImageDraw.Draw(canvas)
                for index,(label,image) in enumerate(tiles):
                    image.thumbnail((width,height))
                    x=(index%columns)*width;y=(index//columns)*(height+28)
                    draw.text((x+5,y+6),label,fill='white')
                    canvas.paste(image,(x,y+28))
                filename=re.sub(r'[^A-Za-z0-9_.-]','_',f'{camera}_{region}')+'.jpg'
                canvas.save(assets/filename,quality=92)
                sheets.append(dict(camera=camera,region=region,path='assets/'+filename))
    report['rois'] = rois
    report['comparison_sheets'] = sheets
    if score_requested:
        import torchmetrics
        for row in report['branches']:
            for region in ('mainpair','background'):
                values=[regions[region] for regions in row.get('roi_metrics',{}).values() if region in regions]
                if values:
                    row.setdefault('roi_mean',{})[region] = {
                        key: (sum(v[key] for v in values)/len(values) if all(v[key] is not None for v in values) else None)
                        for key in METRICS}
                    row['roi_mean'][region]['views'] = len(values)
        report['roi_metric_definition'] = dict(source='saved 8-bit PNG panels, RGB/255, no masks, no compositing, no resize',
            psnr='-10 log10 mean RGB squared error; exact equality has null PSNR plus perfect_pixel_match=true',
            ssim='torchmetrics structural_similarity_index_measure, data_range=1',
            lpips='torchmetrics LearnedPerceptualImagePatchSimilarity AlexNet normalize=True',
            averaging='unweighted per-view arithmetic mean, view count explicit', device=device)
        report['roi_metric_definition'].update(torch=scorer.torch.__version__,torchmetrics=torchmetrics.__version__)


def formatted(value):
    return '—' if value is None else f'{value:.6f}'


def write_reports(report, output, html_requested):
    dump(output/'summary.json',report)
    fields=['name','status','scope','size','radius','aabb_volume','checkpoint_id','checkpoint_step',*METRICS]
    with (output/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in report['branches']:
            writer.writerow({key: (row.get('metrics') or {}).get(key) if key in METRICS else row.get(key) for key in fields})
    lines=['# Marshawn AABB / occupancy comparison','',report['warnings'][0],'',
           '| Variant | Status | Volume | Radius | Checkpoint | PSNR | SSIM | LPIPS |',
           '|---|---|---:|---:|---|---:|---:|---:|']
    for row in report['branches']:
        metrics=row.get('metrics') or {}
        lines.append('| '+' | '.join([row['name'],row['status'],formatted(row['aabb_volume']),str(row['radius']),
                     row.get('checkpoint_id') or '—',*[formatted(metrics.get(key)) for key in METRICS]])+' |')
    lines+=['','Full-frame metrics come from the original floating evaluator. ROI metrics, when requested, use saved 8-bit rectangles. Background rectangles are not empty-space ground truth. All occupancy and visual acceptance gates remain separate.','']
    roi_rows=[]
    for row in report['branches']:
        for camera, regions in row.get('roi_metrics',{}).items():
            for region, metrics in regions.items():
                roi_rows.append(dict(variant=row['name'],camera=camera,region=region,
                    rectangle=json.dumps(metrics['rectangle']),**{key:metrics[key] for key in METRICS}))
    roi_headers=['variant','camera','region','rectangle',*METRICS]
    if roi_rows:
        with (output/'roi_metrics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=roi_headers);writer.writeheader();writer.writerows(roi_rows)
        lines+=['| Variant | Camera | ROI | Native rectangle | PSNR | SSIM | LPIPS |',
                '|---|---|---|---|---:|---:|---:|']
        for row in roi_rows:
            lines.append('| '+' | '.join([row['variant'],row['camera'],row['region'],row['rectangle'],
                *[formatted(row[key]) for key in METRICS]])+' |')
        lines.append('')
    for row in report['branches']:
        for warning in row['warnings']:
            lines.append(f"- {row['name']}: {warning}")
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    if html_requested:
        e=html.escape
        rows=[]
        for row in report['branches']:
            metrics=row.get('metrics') or {}
            values=[row['name'],row['status'],formatted(row['aabb_volume']),str(row['radius']),row.get('checkpoint_id') or '—',
                    *[formatted(metrics.get(key)) for key in METRICS]]
            rows.append('<tr>'+''.join('<td>'+e(value)+'</td>' for value in values)+'</tr>')
        pictures=''
        for sheet in report.get('comparison_sheets',[]):
            native_links=[]
            for row in report['branches']:
                path=row.get('native_crops',{}).get(sheet['camera'],{}).get(sheet['region'])
                if path:
                    native_links.append(f"<a href='{e(path)}'>{e(row['name'])}</a>")
            pictures+=f"<figure><figcaption>{e(sheet['camera'])} · {e(sheet['region'])}</figcaption><a href='{e(sheet['path'])}'><img loading='lazy' src='{e(sheet['path'])}'></a><p>Native GT | prediction crops: {' · '.join(native_links)}</p></figure>"
        roi_table=''
        if roi_rows:
            roi_table='<h2>Saved-PNG rectangle metrics</h2><table><tr>'+''.join('<th>'+e(key)+'</th>' for key in roi_headers)+'</tr>'
            for row in roi_rows:
                roi_table+='<tr>'+''.join('<td>'+e(formatted(row[key]) if key in METRICS else str(row[key]))+'</td>' for key in roi_headers)+'</tr>'
            roi_table+='</table>'
        warnings=''.join('<li>'+e(row['name']+': '+warning)+'</li>' for row in report['branches'] for warning in row['warnings'])
        body=f"<!doctype html><html><meta charset='utf-8'><title>Marshawn AABB comparison</title><style>body{{font:15px system-ui;margin:30px;max-width:1500px}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ccc}}img{{max-width:100%;height:auto}}figure{{margin:25px 0}}.warning{{color:#934000}}</style><h1>Marshawn AABB / occupancy comparison</h1><p class='warning'>{e(report['warnings'][0])}</p><p>Full-frame metrics: original evaluator. ROI metrics: fixed unmasked saved-PNG rectangles. Background does not mean verified empty space. Manual visual and occupancy review remains pending.</p><p><a href='summary.json'>Inspectable JSON and ROI metrics</a> · <a href='summary.csv'>CSV</a></p><table><tr>{''.join('<th>'+e(v)+'</th>' for v in ['Variant','Status','Volume','Radius','Checkpoint','PSNR','SSIM','LPIPS'])}</tr>{''.join(rows)}</table><ul>{warnings}</ul>{pictures}</html>"
        body=body.replace('</table><ul>','</table>'+roi_table+'<ul>',1)
        (output/'index.html').write_text(body)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--html',action='store_true')
    parser.add_argument('--score-rois',action='store_true')
    parser.add_argument('--rois',type=Path)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.report_only and (args.html or args.score_rois):
        parser.error('--report-only cannot combine with image/scoring options')
    report=collect(args.campaign)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.html or args.score_rois:
        rois=read(args.rois) if args.rois else DEFAULT_ROIS
        make_visuals(report,args.output,rois,args.html,args.score_rois,args.device)
    write_reports(report,args.output,args.html)
    print(json.dumps(dict(branches=len(report['branches']),evaluated=sum(r['status']=='evaluated' for r in report['branches']),
                         output=str(args.output.resolve()),warnings=report['warnings'])))


if __name__=='__main__':main()
