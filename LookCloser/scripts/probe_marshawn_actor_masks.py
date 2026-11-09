"""Native RAW-crop SAM3 pilot. Candidates only; no automatic actor acceptance."""
from pathlib import Path
import argparse,json,time,hashlib,os
import numpy as np
from PIL import Image,ImageDraw

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def main(a):
 import torch
 from sam3.model_builder import build_sam3_image_model
 from sam3.model.sam3_image_processor import Sam3Processor
 import sam3.model_builder as builder
 import sam3.model.sam3_image_processor as processor
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(16*1024**3/torch.cuda.get_device_properties(0).total_memory);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
 a.output.mkdir(exist_ok=False);request=json.loads(a.request.read_text());records=[];model=build_sam3_image_model(checkpoint_path=str(a.checkpoint),load_from_HF=False,enable_inst_interactivity=True);proc=Sam3Processor(model,confidence_threshold=.25)
 for row in request['records']:
  image=Image.open(row['image']).convert('RGB');assert sha(row['image'])==row['image_sha256'];w,h=image.size
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   state=proc.set_image(image);result=proc.set_text_prompt(prompt='person',state=state)
  masks=result['masks'].detach().cpu().numpy().reshape(-1,h,w);scores=result['scores'].float().cpu().numpy().reshape(-1);boxes=result['boxes'].float().cpu().numpy();folder=a.output/row['camera'];folder.mkdir();np.savez_compressed(folder/'candidates.npz',masks=masks,scores=scores,boxes=boxes)
  base=np.asarray(image);panels=[];thumb=image.copy();thumb.thumbnail((560,630));tile=Image.new('RGB',(560,660));tile.paste(thumb,((560-thumb.width)//2,25));ImageDraw.Draw(tile).text((4,4),'RAWnativecrop '+row['camera'],fill='white');panels.append(tile)
  for j,mask in enumerate(masks):
   assert mask.dtype==np.bool_;Image.fromarray(mask.astype('uint8')*255).save(folder/('candidate_%02d.png'%j));ov=base.copy();ov[mask]=(ov[mask]*.55+np.array([0,255,255])*.45).astype('uint8');im=Image.fromarray(ov);im.thumbnail((560,630));tile=Image.new('RGB',(560,660));tile.paste(im,((560-im.width)//2,25));ImageDraw.Draw(tile).text((4,4),'candidate%d score%.4f area%d'%(j,scores[j],mask.sum()),fill='white');panels.append(tile)
  sheet=Image.new('RGB',(min(4,len(panels))*560,((len(panels)+3)//4)*660))
  for j,p in enumerate(panels):sheet.paste(p,((j%4)*560,(j//4)*660))
  sheet.save(folder/'candidate_review.jpg');records.append({'camera':row['camera'],'native_crop_raster':[w,h],'raw_master_raster':[8192,4320],'crop_offset':row['crop_xyxy'][:2],'candidates':len(masks),'scores':scores.tolist(),'areas':[int(x.sum())for x in masks],'visual_selection':'PENDING','no_HF_or_cloud_upload':True});print(json.dumps(records[-1]),flush=True)
  del state,result,masks;torch.cuda.empty_cache()
 receipt={'kind':'SAM3_NATIVE_RAW_CROP_INSTANCE_CANDIDATES_ONLY','request_sha256':sha(a.request),'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha(a.checkpoint),'model_builder_sha256':sha(builder.__file__),'processor_sha256':sha(processor.__file__),'script_sha256':sha(__file__),'SAM3p1_gated_not_available':True,'model_input_internal_resolution':proc.resolution,'no_8k_encoder_accuracy_claim':True,'must_refine_native_boundary_tiles_before_accepted8Kmask':True,'runtime':{'torch':torch.__version__,'GPU':torch.cuda.get_device_name(0)},'precision':'bfloat16 TF32false','records':records,'optimizer_updates':0,'NN_training':False,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()};(a.output/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');print('COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['request','checkpoint','output']:p.add_argument('--'+k,type=Path,required=True)
 main(p.parse_args())
