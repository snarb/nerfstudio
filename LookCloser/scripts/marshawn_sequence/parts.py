"""Current-image SAM3 people/head proposals, always intersected with subject masks.

Camera-local person IDs are evaluation domains, not cross-view identity claims.
Occluded heads have empty proposals; they never become bounding-box proxies.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from common import read, atomic, sha


def run(data, checkpoint, only=None):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    root=Path(data)
    ready=read(root/'native_ready.json') if (root/'native_ready.json').exists() else dict(records=[read(p) for p in sorted((root/'qc').glob('*.json'))])
    out=root/'parts';out.mkdir(exist_ok=True)
    torch.set_num_threads(3);torch.manual_seed(42)
    proc=Sam3Processor(build_sam3_image_model(checkpoint_path=checkpoint,load_from_HF=False,device='cuda'),confidence_threshold=.15)
    def segment(image,prompt):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=proc.set_image(image)
            state=proc.set_text_prompt(prompt=prompt,state=state)
            masks=state['masks'].detach().cpu().numpy().reshape(-1,image.height,image.width)>.5
            scores=state['scores'].detach().float().cpu().numpy()
        return masks,scores
    for row in ready['records']:
        c=row['camera']
        if only and c not in only:continue
        if (out/(c+'.json')).exists():continue
        image=Image.open(root/row['image']).convert('RGB')
        with np.load(root/'semantics'/(c+'.npz')) as z: subject=z['black']|z['grey']
        masks,scores=segment(image,'person')
        selected=[]
        for i in np.argsort(-scores):
            mask=masks[i];area=int(mask.sum())
            if area>100 and (mask&subject).sum()/area>.5:
                selected.append((mask&subject,float(scores[i])))
        # Keep distinct visible people rather than multiple prompts for one person.
        labels=np.zeros(subject.shape,np.uint8);records=[]
        for mask,score in selected:
            counts=np.bincount(labels[mask],minlength=len(records)+1)
            if records:
                parent=int(np.argmax(counts[1:]))+1
                covered=int(counts[1:].sum());visible=mask&(labels==0)
                parent_size=int((labels==parent).sum())
                # A second SAM proposal can isolate the same person's leg/hair.
                # Merge only a small residual with substantial measured mask overlap.
                if counts[parent]/max(1,mask.sum())>.15 and visible.sum()<.2*parent_size:
                    labels[visible]=parent
                    continue
                if covered/max(1,mask.sum())>.65:continue
            if any((mask&(labels==r['id'])).sum()/max(1,(mask|(labels==r['id'])).sum())>.55 for r in records):continue
            visible=mask&(labels==0)
            if visible.sum()<100:continue
            ident=len(records)+1
            if ident>8:raise ValueError('Unexpectedly many subject person instances '+c)
            labels[visible]=ident
            records.append(dict(id=ident,score=score,pixels=int(visible.sum())))
        if not records:raise ValueError('No SAM person accepted '+c)
        head_union=np.zeros(subject.shape,bool)
        for prompt in ('head','hair','face'):
            candidates,_=segment(image,prompt)
            for mask in candidates:
                if mask.sum() and (mask&subject).sum()/mask.sum()>.5:
                    yy,xx=np.nonzero(mask)
                    x0=max(0,int(xx.min())-48);y0=max(0,int(yy.min())-48)
                    x1=min(image.width,int(xx.max())+49);y1=min(image.height,int(yy.max())+49)
                    # Refine at native pixel scale when the candidate fits the native1008 canvas.
                    if max(x1-x0,y1-y0)<=1008:
                        fine,_=segment(image.crop((x0,y0,x1,y1)),prompt)
                        seed=mask[y0:y1,x0:x1]
                        for f in fine:
                            if f.sum() and (f&seed).sum()/f.sum()>.25:
                                head_union[y0:y1,x0:x1]|=f
                    else:head_union|=mask
        head_union&=subject
        unknown=np.zeros(subject.shape,bool)
        for mask in masks:
            if mask.sum() and (mask&subject).sum()/mask.sum()<=.5:unknown|=mask
        arrays={'labels':labels,'head_union':head_union,'unknown_people':unknown}
        for r in records:
            ident=r['id'];mask=labels==ident;head=head_union&mask
            r['pixels']=int(mask.sum())
            arrays['person'+str(ident)]=mask;arrays['head'+str(ident)]=head
            yy,xx=np.nonzero(mask);r['xyxy']=[max(0,int(xx.min())-16),max(0,int(yy.min())-16),min(image.width,int(xx.max())+17),min(image.height,int(yy.max())+17)]
            r['head_pixels']=int(head.sum())
        np.savez_compressed(out/(c+'.npz'),**arrays)
        small=np.array(image);colors=np.array([[255,80,80],[60,220,90],[50,140,255],[240,220,50],[220,70,220],[70,220,220]],np.uint8)
        for r in records:
            mask=labels==r['id'];small[mask]=(small[mask]*.5+colors[(r['id']-1)%len(colors)]*.5).astype(np.uint8)
        small[head_union]=[255,255,255]
        im=Image.fromarray(small);im.thumbnail((900,900));ImageDraw.Draw(im).text((5,5),c+' SAM people / white heads',fill='white');im.save(out/(c+'_review.jpg'))
        atomic(out/(c+'.json'),dict(camera=c,records=records,source_rgb_sha256=row['rgb_sha256'],source_mask_sha256=row['source_mask_sha256'],archive_sha256=sha(out/(c+'.npz')),sam_checkpoint_sha256=sha(checkpoint),head_policy='current-image head/hair/face; native refinement for candidate ROI<=1008; intersect subject and instance; no bbox proxy',unassigned_subject_pixels=int((subject&(labels==0)).sum())))
        print('PARTS',c,len(records),[r['head_pixels'] for r in records],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--checkpoint',default='/home/nvuser/sam3env/weights/sam3.pt');p.add_argument('--only',nargs='+');a=p.parse_args();run(a.data,a.checkpoint,a.only)
