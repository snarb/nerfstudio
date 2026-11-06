"""Native-crop SAM3 candidates for manual temporal identity review; no training."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from prepare_marshawn_color_variants import sha


def main(a):
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.35)
    model=build_sam3_image_model(checkpoint_path=str(a.checkpoint),
        load_from_HF=False,enable_inst_interactivity=True)
    proc=Sam3Processor(model,confidence_threshold=.25)
    records=[]
    for frame in a.frames:
        for cam in a.cameras:
            folder=a.sources/str(frame)/cam
            out=a.output/str(frame)/cam
            if (out/'result.json').exists():
                records.append(json.loads((out/'result.json').read_text()));continue
            limit=time.monotonic()+1800
            while True:
                p=folder/'receipt.json'
                row=json.loads(p.read_text()) if p.exists() else {}
                if row.get('native_image') and Path(row['native_image']).exists():break
                if time.monotonic()>limit:raise TimeoutError(str(p))
                time.sleep(2)
            source=Path(row['native_image'])
            assert sha(source)==row['native_image_sha256']
            im=Image.open(source).convert('RGB');w,h=im.size
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                state=proc.set_image(im)
                result=proc.set_text_prompt(prompt='person',state=state)
            masks=result['masks'].detach().cpu().numpy().reshape(-1,h,w)
            boxes=result['boxes'].float().cpu().numpy()
            scores=result['scores'].float().cpu().numpy()
            out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/'candidates.npz',masks=masks,boxes=boxes,scores=scores)
            thumb=im.copy();thumb.thumbnail((340,360))
            panels=[]
            for k in range(-1,len(masks)):
                tile=Image.new('RGB',(350,390));tile.paste(thumb,(0,25))
                d=ImageDraw.Draw(tile);d.text((4,4),f'{frame} {cam} '+('SOURCE' if k<0 else f'ID {k} score {scores[k]:.3f}'),fill='white')
                if k>=0:
                    m=np.array(Image.fromarray(masks[k]).resize(thumb.size,Image.Resampling.NEAREST))
                    arr=np.array(thumb);arr[m]=(arr[m]*.45+np.array([0,255,255])*.55).astype('uint8')
                    tile.paste(Image.fromarray(arr),(0,25))
                panels.append(tile)
            sheet=Image.new('RGB',(min(4,len(panels))*350,((len(panels)+3)//4)*390))
            for k,tile in enumerate(panels):sheet.paste(tile,(k%4*350,k//4*390))
            sheet.save(out/'candidates.jpg',quality=92)
            record=dict(frame=frame,camera=cam,candidates=len(masks),scores=scores.tolist(),
                source_receipt=str(p),native_image_sha256=sha(source),
                candidates_sha256=sha(out/'candidates.npz'),
                native_raster=[w,h],internal_encoder_resolution=proc.resolution,
                checkpoint_sha256=sha(a.checkpoint),identity_review='PENDING',optimizer_updates=0)
            (out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
            records.append(record);print(frame,cam,len(masks),flush=True)
            del state,result,masks,im
            torch.cuda.empty_cache()
    import hashlib
    batch=hashlib.sha256(json.dumps([a.frames,a.cameras]).encode()).hexdigest()[:12]
    (a.output/('batch_'+batch+'.json')).write_text(json.dumps(dict(records=records),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['sources','output','checkpoint']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frames',nargs='+',type=int,required=True)
    p.add_argument('--cameras',nargs='+',required=True)
    main(p.parse_args())
