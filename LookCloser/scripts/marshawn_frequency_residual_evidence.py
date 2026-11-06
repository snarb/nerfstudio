"""Read-only, deterministic post-fit evidence from the actual trained 2D model."""
import hashlib
import json
import pickle
import random
from pathlib import Path


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().contiguous().view(__import__('torch').uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def rng_digest():
    import numpy as np
    import torch
    data = pickle.dumps((random.getstate(), np.random.get_state()), protocol=4)
    data += torch.get_rng_state().cpu().numpy().tobytes()
    for state in torch.cuda.get_rng_state_all():
        data += state.cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def save_evidence(model, image, prep, spec, output):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    output = Path(output) / 'residual_evidence'
    output.mkdir(exist_ok=True)
    policy = spec['postfit_evidence']
    if policy['source_image_sha256'] != hashlib.sha256(Path(spec['image']).read_bytes()).hexdigest():
        raise ValueError('Evidence ROI source changed')
    before = {'model': model_digest(model), 'rng': rng_digest()}
    modes = {name: module.training for name, module in model.named_modules()}
    result = {'before': before, 'patches': {}, 'level': 15, 'SSIM_window': 7, 'patch_size': 8, 'source_image_sha256': policy['source_image_sha256'], 'residual_visualization': 'absolute RGB residual multiplied by 8, clipped only for PNG; NPZ stores raw floats'}
    model.eval()
    try:
        with torch.no_grad():
            for name, (x0, y0, x1, y1) in policy['rois_xyxy'].items():
                if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]) or any(v % 8 for v in (x0, y0, x1, y1)):
                    raise ValueError('Evidence ROI must lie on original 8-pixel patch lattice')
                ys, xs = torch.meshgrid(torch.arange(y0, y1, device=image.device, dtype=torch.float32)+.5, torch.arange(x0, x1, device=image.device, dtype=torch.float32)+.5, indexing='ij')
                uv = torch.stack([xs/image.shape[1], ys/image.shape[0]], dim=-1).reshape(-1, 2)
                pred = model.render_masked(uv, max_active_level=15).reshape(y1-y0, x1-x0, 3).float()
                gt = image[y0:y1, x0:x1].float()
                def patches(x):
                    return x.reshape((y1-y0)//8, 8, (x1-x0)//8, 8, 3).permute(0, 2, 4, 1, 3).reshape(-1, 3, 8, 8).contiguous()
                scores = prep.compute_ssim(patches(gt), patches(pred), window_size=7, size_average=False)
                g, p, s = [x.detach().cpu().numpy() for x in (gt, pred, scores)]
                residual = p - g
                np.savez_compressed(output/(name+'.npz'), gt=g, prediction=p, residual=residual, ssim8=s)
                panels = [Image.fromarray(prep.to_uint8_np(a)) for a in (g, p, np.abs(residual)*8)]
                sheet = Image.new('RGB', (panels[0].width*3, panels[0].height+24))
                draw = ImageDraw.Draw(sheet)
                for i, (panel, label) in enumerate(zip(panels, ('GT', 'actual model L15', '|residual| x8'))):
                    sheet.paste(panel, (i*panel.width, 24)); draw.text((i*panel.width+2, 4), label, fill='white')
                sheet.save(output/(name+'.png'))
                result['patches'][name] = {'source_xyxy': [x0, y0, x1, y1], 'ssim8': s.tolist(), 'mean_ssim8': float(s.mean()), 'pass_fraction_ssim095': float((s >= .95).mean()), 'mean_abs_rgb_residual': float(np.abs(residual).mean()), 'max_abs_rgb_residual': float(np.abs(residual).max())}
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
    result['after'] = {'model': model_digest(model), 'rng': rng_digest()}
    if result['before'] != result['after']:
        raise ValueError('Post-fit evidence changed model or RNG')
    result['model_and_rng_unchanged'] = True
    result['artifacts'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output/'receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result
