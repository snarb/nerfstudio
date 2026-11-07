"""Fresh SAM3 detections and native 1:1 boundary tiles for every video source.

Previous-frame labels guide identity only. No previous mask is used as the final
mask, and no radiance-field state is read. Ambiguity raises before publication.
"""
from pathlib import Path
import argparse,json,os,time
import cv2,numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt,binary_erosion
from prepare_marshawn_tuning import sha,write
from marshawn_video_prep_mask_components import qualify

BASE=Path('/fsx/tmp/marshawn_001660_hd/two_actor_isolation/confidence_native_hair_v2/manifest.json')
OVERLAY=Path('/fsx/tmp/marshawn_001660_hd/two_actor_isolation/local_sam3_overlay')

def small(mask):return cv2.resize(mask.astype('uint8'),(512,270),interpolation=cv2.INTER_AREA)>0

def seed(root,frame,camera):
    for previous in range(frame-5,max(1659,frame-65),-5):
        path=root/'frames'/f'{previous:06d}'/'native_masks'/f'{camera}.npz'
        if path.is_file():
            with np.load(path) as z:return {a:small(z[a]) for a in ['black','grey']},previous,sha(path)
    if frame>1720:raise RuntimeError('No recent identity labels; prepare earlier frames first: '+str(frame)+' '+camera)
    row=next(r for r in json.loads(BASE.read_text())['rows'] if r['camera']==camera)
    if sha(row['raw_archive'])!=row['raw_archive_sha256']:raise ValueError('Seed labels changed')
    with np.load(row['raw_archive']) as z:
        x0,y0,x1,y1=z['crop_xyxy'];result={}
        for a in ['black','grey']:
            full=np.zeros((4320,8192),bool);full[y0:y1,x0:x1]=z[a];result[a]=small(full)
    return result,1660,row['raw_archive_sha256']

def select(masks,reference):
    coarse=np.stack([small(m) for m in masks]);selected={};stats={}
    for actor,ref in reference.items():
        union=(coarse|ref).sum((1,2));iou=(coarse&ref).sum((1,2))/np.maximum(union,1);order=np.argsort(iou)[::-1];best=int(order[0]);runner=float(iou[order[1]]) if len(order)>1 else 0.
        if iou[best]<.10:
            selected[actor]=None;stats[actor]=dict(instance=None,IoU=float(iou[best]),visible_supported=False,reason='No qualified current-frame detection; visibility UNKNOWN, no fabricatedforeground');continue
        if runner>.6*iou[best] and iou[best]-runner<.04:
            # Duplicate detections of the same person may be merged, but a
            # distinct competing person must never silently receive fighter ID.
            second=int(order[1]);inter=(coarse[best]&coarse[second]).sum()/max(1,min(coarse[best].sum(),coarse[second].sum()))
            if inter<.8:
                selected[actor]=None;stats[actor]=dict(instance=None,visible_supported=False,reason='Ambiguous current-frame identity; visibility UNKNOWN');continue
        selected[actor]=best;stats[actor]=dict(instance=best,visible_supported=True,IoU=float(iou[best]),runner_IoU=runner)
    if selected['black'] is not None and selected['black']==selected['grey']:raise ValueError('Actors assigned same instance')
    return selected,stats

def refine(model,proc,image,coarse,veto):
    """Add-only recovery from ≤1008px native tiles; never upscale whole image."""
    h,w=coarse.shape;distance=distance_transform_edt(~coarse);allowed=(distance<=64)&~veto
    boundary=coarse&~binary_erosion(coarse,iterations=6);yy,xx=np.where(boundary)
    tiles=set()
    for x,y in zip(xx[::max(1,len(xx)//20000)],yy[::max(1,len(yy)//20000)]):
        tx=min(max(0,int(x)//800*800-104),w-1008);ty=min(max(0,int(y)//800*800-104),h-1008);tiles.add((tx,ty))
    output=coarse.copy();receipts=[]
    for x,y in sorted(tiles):
        target=coarse[y:y+1008,x:x+1008];bad=veto[y:y+1008,x:x+1008]
        if target.sum()<16:continue
        interior=distance_transform_edt(target);fy,fx=np.unravel_index(interior.argmax(),interior.shape)
        pts=[[fx,fy]];labels=[1]
        by,bx=np.where(bad)
        if len(bx):pts.append([int(np.median(bx)),int(np.median(by))]);labels.append(0)
        state=proc.set_image(image.crop((x,y,x+1008,y+1008)))
        masks,scores,_=model.predict_inst(state,point_coords=np.asarray(pts,np.float32),point_labels=np.asarray(labels,np.int32),multimask_output=True)
        masks=np.asarray(masks,dtype=bool);scores=np.asarray(scores);overlap=(masks&target).sum((1,2))/np.maximum((masks|target).sum((1,2)),1);k=int(overlap.argmax());addition=masks[k]&allowed[y:y+1008,x:x+1008]&~target
        if overlap[k]>=.25:output[y:y+1008,x:x+1008]|=addition
        receipts.append(dict(x=x,y=y,w=1008,h=1008,IoU=float(overlap[k]),model_score=float(scores[k]),added_pixels=int(addition.sum()) if overlap[k]>=.25 else 0))
        del state
    output[veto]=False
    return output,receipts

def main(a):
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(min(.9,16*1024**3/torch.cuda.get_device_properties(0).total_memory))
    checkpoint=a.checkpoint
    if sha(checkpoint)!='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e':raise ValueError('Wrong SAM3 checkpoint')
    model=build_sam3_image_model(checkpoint_path=str(checkpoint),load_from_HF=False,enable_inst_interactivity=True);proc=Sam3Processor(model,confidence_threshold=.25)
    out=a.root/'frames'/f'{a.frame:06d}'/'native_masks';out.mkdir(parents=True,exist_ok=True)
    for camera in a.cameras:
        if (out/(camera+'.json')).exists():
            previous_record=json.loads((out/(camera+'.json')).read_text())
            if previous_record['frame']!=a.frame or previous_record['archive_sha256']!=sha(out/(camera+'.npz')):raise ValueError('Existing native-mask artifact changed')
            if previous_record.get('native_cleanup_version')==1:continue
            (out/(camera+'.json')).rename(out/(camera+'.pre_cleanup.json'))
        receipt=a.root/'sources'/str(a.frame)/camera/'receipt.json';r=json.loads(receipt.read_text());path=Path(r['raw_native_image'])
        if sha(path)!=r['raw_native_image_sha256']:raise ValueError('Native image changed')
        image=Image.open(path).convert('RGB');reference,previous,seedsha=seed(a.root,a.frame,camera)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=proc.set_image(image);result=proc.set_text_prompt(prompt='person',state=state);masks=result['masks'].detach().cpu().numpy().reshape(-1,4320,8192)
            selected,stats=select(masks,reference);selected_ids=[k for k in selected.values() if k is not None];nominal=masks[selected_ids].any(0) if selected_ids else np.zeros((4320,8192),bool);others=[];ambiguous=[]
            for k,m in enumerate(masks):
                if k in selected.values():continue
                overlap=(m&nominal).sum()/max(1,m.sum())
                (ambiguous if overlap>.1 else others).append(k)
            veto=masks[others].any(0) if others else np.zeros_like(nominal);unknown=veto.copy()
            if ambiguous:unknown|=masks[ambiguous].any(0)&~nominal
            if any(k is None for k in selected.values()):unknown[:]=True
            chosen={a:masks[k].copy() if k is not None else np.zeros_like(nominal) for a,k in selected.items()};del state,result,masks
            component_receipts={}
            for actor in chosen:
                chosen[actor],discarded,component_receipts[actor]=qualify(chosen[actor],reference[actor]);unknown|=discarded
            refined={};tiles={}
            for actor,mask in chosen.items():
                if mask.any():refined[actor],tiles[actor]=refine(model,proc,image,mask,veto)
                else:refined[actor],tiles[actor]=mask,[]
        overlap=refined['black']&refined['grey'];refined['black'][overlap]=False;refined['grey'][overlap]=False
        np.savez_compressed(out/(camera+'.npz'),**refined,source_ineligible=veto,unknown=unknown,crop_xyxy=np.asarray([0,0,8192,4320]))
        record=dict(native_cleanup_version=1,component_qualification=component_receipts,mask_code_sha256=sha(__file__),frame=a.frame,camera=camera,source_receipt_sha256=sha(receipt),raw_sha256=r['raw_sha256'],raw_native_image_sha256=sha(path),archive_sha256=sha(out/(camera+'.npz')),checkpoint_sha256=sha(checkpoint),identity_reference_frame=previous,identity_reference_sha256=seedsha,identity=stats,tiles=tiles,native_raster=[8192,4320],native_boundary_tile_raster=1008,no_NeRF_state_loaded=True,mask_transfer=False,policy='FreshglobalSAM3candidates; previouslabelsidentityonly; add-onlynativeinteractiveboundarytiles within64px; verifiedotherpersonveto; ambiguousduplicatesUNKNOWN',finished=time.time())
        write(out/(camera+'.json'),record);print(a.frame,camera,'complete',flush=True);torch.cuda.empty_cache()
    expected=set(a.cameras)
    write(out/'batch_ready.json',dict(frame=a.frame,cameras=sorted(expected),records={c:sha(out/(c+'.json')) for c in expected}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',type=int,required=True);p.add_argument('--cameras',nargs='+',required=True);p.add_argument('--checkpoint',type=Path,default=OVERLAY/'weights/sam3.pt');main(p.parse_args())
