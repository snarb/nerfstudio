"""Post-hoc DC eval-ray diagnostics. No depth labels and never training input."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
from audit_marshawn_tuning_checkpoint import ray_for_pixel


def prepare(c,out):
    out.mkdir(exist_ok=False,parents=True);data=c/'data';review=c/'exp_followup_review_15188'
    images={'GT':np.array(Image.open(data/'images/eval_DC.png').convert('RGB'))}
    for name,folder in [('parent','structural_exp_hplus1'),('actor','exp_followup_actor'),('half','exp_followup_step_half')]:images[name]=np.array(Image.open(review/folder/'DC.png').convert('RGB'))
    with np.load(data/'semantics/DC.npz')as z:masks={a:z[a].copy()for a in ['black','grey']}
    err=np.mean((images['actor'].astype(float)-images['GT'].astype(float))**2,axis=2);yy,xx=np.indices(err.shape)
    regions=[('black','shirt_error',[820,120,1050,320],4,True),('grey','garment_error',[930,245,1110,395],4,True),
             ('black','pants_control',[760,340,950,505],2,False),('grey','pants_control',[930,390,1115,535],2,False)]
    rows={Path(r['file_path']).stem.split('_')[-1]:r for r in json.loads((data/'transforms.json').read_text())['frames']};pixels=[];rays=[]
    for actor,region,box,count,largest in regions:
        x0,y0,x1,y1=box;valid=binary_erosion(masks[actor],iterations=4)&(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
        for _ in range(count):
            if not valid.any():raise ValueError('No remaining declared clothing ROI pixel')
            scores=np.where(valid,err,-np.inf if largest else np.inf);flat=np.argmax(scores)if largest else np.argmin(scores);y,x=np.unravel_index(flat,err.shape)
            pixel=dict(camera=0,camera_name='DC',x=int(x),y=int(y),actor=actor,region=region,semantic='POSTHOC_EVAL_RGB_DIAGNOSTIC_NO_DEPTH_GROUND_TRUTH',
                diagnostic_group='DC_'+region,selection='largest RGB error within declared clothing ROI'if largest else'low-error clothing control',gt_rgb=(images['GT'][y,x]/255).tolist(),
                observed_actor_png_rgb=(images['actor'][y,x]/255).tolist(),saved_png_rgb={name:(im[y,x]/255).tolist()for name,im in images.items()},training_allowed=False)
            pixels.append(pixel);rays.append(ray_for_pixel(rows['DC'],int(x),int(y)));valid&=(xx-x)**2+(yy-y)**2>=28**2
    q=dict(kind='POSTHOC12_DC_EVAL_ERROR_DIAGNOSTIC',split='eval',camera_order=['DC','HC','KA'],pixels=pixels,rays=rays,training_allowed=False,
        source='DC GT and fixed15188 actor-objective RGB errors;12 diagnostic rays selected after observing failure. Not an independent selection metric.',
        depth_ground_truth_available=False,source_images_sha256={str(data/'images/eval_DC.png'):hashlib.sha256((data/'images/eval_DC.png').read_bytes()).hexdigest()},
        query_rule=regions,prepared_transforms_sha256=hashlib.sha256((data/'transforms.json').read_bytes()).hexdigest())
    for name in ['structural_exp_hplus1','exp_followup_actor','exp_followup_step_half']:
        image=review/name/'DC.png';q['source_images_sha256'][str(image)]=hashlib.sha256(image.read_bytes()).hexdigest()
    (out/'queries.json').write_text(json.dumps(q,indent=2)+'\n')
    atlas=Image.new('RGB',(620,len(pixels)*104),(25,25,25));draw=ImageDraw.Draw(atlas)
    for i,p in enumerate(pixels):
        draw.text((4,i*104+12),f'{i} {p["actor"]}\n{p["region"]}\nx{p["x"]} y{p["y"]}',fill='white')
        for j,(name,im)in enumerate(images.items()):
            x,y=p['x'],p['y'];crop=Image.fromarray(im).crop((x-32,y-32,x+32,y+32)).resize((88,88));ox=150+j*116;oy=i*104+15;atlas.paste(crop,(ox,oy));draw.text((ox,i*104+1),name,fill='white');draw.ellipse((ox+42,oy+42,ox+46,oy+46),outline='yellow')
    atlas.save(out/'source_patch_review.jpg',quality=93);print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.campaign,a.output)
