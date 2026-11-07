"""Read-only native1:1 current hair candidates; no source mask writes."""
from pathlib import Path
import argparse,json,os,sys,time
import numpy as np
from PIL import Image,ImageDraw

def main(a):
 s=json.loads(a.spec.read_text());o=Path(s['overlay']);sys.path[:0]=[str(o/'src'),str(o/'deps'),s['frozen_scripts']]
 import torch
 from prepare_marshawn_tuning import sha,write
 from marshawn_gpu_isolation import require_controller_lease,verify_cuda
 from sam3.model_builder import build_sam3_image_model
 from sam3.model.sam3_image_processor import Sam3Processor
 gpu=require_controller_lease(os.environ['CREW_PILOT_LEASE_RECEIPT']);gpu.update(verify_cuda(gpu['expected_physical_uuid'],torch));os.fstat(int(os.environ['LOOKCLOSER_GPU_LEASE_FD']));torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(min(.9,16*1024**3/torch.cuda.get_device_properties(0).total_memory))
 assert sha(s['raw_native_image'])==s['raw_native_image_sha256'];assert sha(o/'weights/sam3.pt')==s['checkpoint_sha256'];image=Image.open(s['raw_native_image']).convert('RGB').crop(tuple(s['crop_xyxy']));model=build_sam3_image_model(checkpoint_path=str(o/'weights/sam3.pt'),load_from_HF=False,enable_inst_interactivity=True);proc=Sam3Processor(model,confidence_threshold=.05);out=Path(s['output']);out.mkdir(parents=True,exist_ok=True);records=[]
 for prompt in s.get('prompts',['hair','braided hair','dreadlocks']):
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   state=proc.set_image(image);result=proc.set_text_prompt(prompt=prompt,state=state);masks=result['masks'].detach().cpu().numpy().reshape(-1,1008,1008).astype(bool);scores=result['scores'].detach().float().cpu().numpy()
  name=prompt.replace(' ','_');archive=out/(name+'.npz');np.savez_compressed(archive,masks=masks,scores=scores,crop_xyxy=np.asarray(s['crop_xyxy']));count=min(len(masks),6);sheet=Image.new('RGB',(768*(count+1),800));base=np.array(image)
  for j in range(count+1):
   p=base.copy()
   if j:mask=masks[j-1];p[mask]=(p[mask]*.4+np.array([0,255,255])*.6).astype('uint8')
   im=Image.fromarray(p);im.thumbnail((768,768));sheet.paste(im,(j*768,24));ImageDraw.Draw(sheet).text((j*768+3,3),prompt+' '+('SOURCE'if not j else f'{j-1} score{scores[j-1]:.4f}'),fill='white')
  sheet.save(out/(name+'_contact.jpg'),quality=96);records.append(dict(prompt=prompt,candidates=len(masks),scores=scores.tolist(),archive_sha256=sha(archive)));print(prompt,records[-1],flush=True)
 write(out/'receipt.json',dict(worker_code_sha256=sha(__file__),prompts=s.get('prompts'),source_spec_sha256=sha(a.spec),source_receipt_sha256=s['source_receipt_sha256'],checkpoint_sha256=s['checkpoint_sha256'],records=records,gpu=gpu,no_live_inputs_changed=True,created=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);main(p.parse_args())
