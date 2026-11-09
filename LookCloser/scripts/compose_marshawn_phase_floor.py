"""CPU matched native floor contacts; separate GT/parent/phase, no fitting."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

ROOT=Path('/fsx/tmp/marshawn_001660_hd')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    import torch
    from PIL import Image,ImageDraw
    from torchmetrics.functional import structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    torch.set_num_threads(2);lpips=LearnedPerceptualImagePatchSimilarity(normalize=True).eval();rows=[];contacts=[]
    for camera in (a.camera or ['HC','DC','KA']):
        parent=ROOT/'native_png_linear_distortion_STAGED'/('distortion_8k_floor_native_'+camera)
        phase=a.stage/f'gate{a.step}_floor_native_{camera}'
        pr=json.loads((parent/'receipt.json').read_text());qr=json.loads((phase/'receipt.json').read_text());p=np.load(parent/f'eval_{camera}_buffers.npz');q=np.load(phase/f'eval_{camera}_buffers.npz')
        if not np.array_equal(p['gt'],q['gt'])or p['rgb'].shape!=q['rgb'].shape:raise ValueError('Changedtarget/raster '+camera)
        bbox=pr['rows'][0]['bbox']
        if bbox!=qr['rows'][0]['bbox']or bbox!=[850,650,1450,980]:raise ValueError('Nativecropmismatch')
        metrics={}
        gt=torch.tensor(p['gt']).permute(2,0,1)[None]
        for name,rgb in [('parent8000',p['rgb']),('phase'+str(a.step),q['rgb'])]:
            pred=torch.tensor(rgb).permute(2,0,1)[None]
            with torch.inference_mode():ssim=float(structural_similarity_index_measure(pred,gt,data_range=1.));distance=float(lpips(pred,gt))
            lpips.reset();mse=float(torch.mean((pred-gt)**2));metrics[name]=dict(psnr=float(-10*np.log10(max(mse,1e-12))),ssim=ssim,lpips=distance)
        h,w=p['gt'].shape[:2];panel=Image.new('RGB',(3*w,h+26),'black');draw=ImageDraw.Draw(panel)
        for col,(rgb,label)in enumerate([(p['gt'],'GT'),(p['rgb'],'parent8k nativecore0'),(q['rgb'],f'phase{a.step} nativecore0')]):
            panel.paste(Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)),(col*w,26));draw.text((col*w+5,5),camera+' '+label,fill='white')
        name=f'gate{a.step}_matched_floor_{camera}.png';panel.save(a.stage/name);contacts.append(panel)
        opaque=(p['opacity']>.95)&(q['opacity']>.95);depth_delta=q['depth']-p['depth'];selection=json.loads((ROOT/'floor_ray_diagnosis/selection.json').read_text());floor_queries=[]
        for query in selection['queries']:
            if query['camera']!=camera or query['split']!='eval':continue
            x,y=query['x']-bbox[0],query['y']-bbox[1]
            if 0<=x<p['rgb'].shape[1]and 0<=y<p['rgb'].shape[0]:
                floor_queries.append(dict(id=query['id'],pixel_xy=[query['x'],query['y']],parent_expected_depth=float(p['depth'][y,x]),phase_expected_depth=float(q['depth'][y,x]),parent_opacity=float(p['opacity'][y,x]),phase_opacity=float(q['opacity'][y,x])))
        rows.append(dict(camera=camera,bbox=bbox,GT_exact=True,parent_checkpoint_sha256=pr['checkpoint_sha256'],phase_checkpoint_sha256=qr['checkpoint_sha256'],parent_render_receipt_sha256=sha(parent/'receipt.json'),phase_render_receipt_sha256=sha(phase/'receipt.json'),metrics=metrics,contact=name,contact_sha256=sha(a.stage/name),depth_drift_diagnostic=dict(opaque_pixels=int(opaque.sum()),median_abs_expected_depth_change=float(np.median(abs(depth_delta[opaque])))if opaque.any()else None,p95_abs_expected_depth_change=float(np.quantile(abs(depth_delta[opaque]),.95))if opaque.any()else None,floor_queries=floor_queries,caveat='Expectedmodeldepth, notindependenttruefloor orbodygeometry; cropalsoincludesprops/actors')))
    width=contacts[0].width;height=sum(c.height for c in contacts);combined=Image.new('RGB',(width,height),'black');y=0
    for contact in contacts:combined.paste(contact,(0,y));y+=contact.height
    combined.save(a.stage/f'gate{a.step}_matched_floor_contact.png')
    result=dict(rows=rows,native_crop_pixels=True,no_resizing=True,panels_separate_not_GT_composited=True,parent_policy='Originalstep8000nativecore0,.003; deliberatelyNOTguardeddownload variant',phase_policy='Savedphasenativecore0,.003',CPU_metrics='PSNR/SSIM/LPIPSonunmodifiedsameGTnativecrop',script_sha256=sha(__file__))
    (a.stage/f'gate{a.step}_matched_floor_contact.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(stage=str(a.stage),step=a.step,cameras=len(rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);p.add_argument('--step',type=int,default=9000);p.add_argument('--camera',action='append',choices=['HC','DC','KA']);main(p.parse_args())
