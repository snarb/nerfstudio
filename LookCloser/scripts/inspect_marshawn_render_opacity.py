"""Expose transparency in saved black-background renders without modifying a model.

These are encoded model-domain composites, not physically linear relighting.
An eroded actor mask is a diagnostic region, not an opaque-material annotation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    if args.output.exists():
        raise ValueError('Preserve prior diagnostic generation')
    config = args.config.read_text()
    if 'background_color: black' not in config:
        raise ValueError('Saved RGB must use the original black-background renderer')
    records = json.loads((args.data/'ready.json').read_text())['records']
    records = {row['camera']: row for row in records}
    metrics = json.loads((args.renders/'metrics.json').read_text())
    args.output.mkdir(parents=True)
    rows, inputs, outputs = [], {str(args.config): sha(args.config)}, {}
    for camera in ('DC', 'HC', 'KA'):
        rgb_path, alpha_path = args.renders/(camera+'.png'), args.renders/(camera+'_alpha.png')
        target_path = args.data/records[camera]['image']
        mask_path = args.data/'semantics'/(camera+'.npz')
        rgb = np.asarray(Image.open(rgb_path).convert('RGB'), dtype=np.float32)/255
        alpha_raw = np.asarray(Image.open(alpha_path))
        if alpha_raw.ndim != 2 or alpha_raw.max() > 65535:
            raise ValueError('Expected saved scalar16-bit alpha')
        alpha = alpha_raw.astype(np.float32)/65535
        if rgb.shape[:2] != alpha.shape or np.any(rgb > alpha[...,None]+1/255+1/65535):
            raise ValueError('RGB/alpha violate black-background premultiplied bounds')
        gt = Image.open(target_path).convert('RGB')
        yy, xx = np.indices(alpha.shape)
        checker = np.where(((yy//24+xx//24)%2)[...,None], .25, .75)
        images = [gt, Image.open(rgb_path).convert('RGB')]
        for background in (.5, checker):
            composed = rgb+(1-alpha[...,None])*background
            images.append(Image.fromarray(np.uint8(np.clip(composed,0,1)*255+.5)))
        images.append(Image.fromarray(np.uint8(alpha*255+.5)).convert('RGB'))
        with np.load(mask_path, allow_pickle=False) as masks:
            for actor in ('black', 'grey'):
                crop = records[camera]['crops'][actor]
                width, height = crop[2]-crop[0], crop[3]-crop[1]
                sheet = Image.new('RGB', (5*width, height+24), (32,32,32))
                draw = ImageDraw.Draw(sheet)
                for column, (label, image) in enumerate(zip(
                        ['GT', 'original black BG', 'model over gray', 'model over checker', 'alpha'], images)):
                    draw.text((column*width+3,4), label, fill='white')
                    sheet.paste(image.crop(crop), (column*width,24))
                destination = args.output/(camera+'_'+actor+'.png')
                sheet.save(destination)
                outputs[destination.name] = sha(destination)
                for erosion in (3, 8, 16):
                    region = binary_erosion(masks[actor], iterations=erosion)
                    values = alpha[region]
                    rows.append(dict(camera=camera, actor=actor, erosion_HD_pixels=erosion,
                                     pixels=int(region.sum()),
                                     below05_fraction=float((values<.5).mean()) if len(values) else None,
                                     below09_fraction=float((values<.9).mean()) if len(values) else None,
                                     below099_fraction=float((values<.99).mean()) if len(values) else None,
                                     alpha_quantiles=np.quantile(values,[0,.1,.5,.9,1]).tolist() if len(values) else []))
        for path in (rgb_path,alpha_path,target_path,mask_path):
            inputs[str(path)] = sha(path)
    report = dict(kind='SAVED_RENDER_OPACITY_DIAGNOSTIC', inputs_sha256=inputs,
                  outputs_sha256=outputs, rows=rows, model_loaded=False,
                  original_renders_changed=False, primary_metrics_recomputed=False,
                  primary_metrics={k:metrics['metrics'][k] for k in ('psnr','ssim','lpips')},
                  limitation='Quantized encoded-domain composites expose learned transmission. Erosion does not certify opaque material, depth or temporal correctness.')
    (args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'output':str(args.output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data','renders','config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
