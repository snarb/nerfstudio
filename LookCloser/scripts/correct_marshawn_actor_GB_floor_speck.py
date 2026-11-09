"""One explicit manually inspected source-floor false positive, not a filter."""
from pathlib import Path
import argparse,hashlib,json,os,shutil
import cv2,numpy as np
from PIL import Image
from marshawn_actor_confidence import native_confidence
from marshawn_actor_coordinates import raw_mask_coverage_to_HD
from prepare_marshawn_actor_confidence_data import full

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(a):
 a.output.mkdir(exist_ok=False);old=json.loads(a.source.read_text());src=a.source.parent
 for sub in ['archives','raw','masks','native_eval_masks']:
  (a.output/sub).mkdir()
  for p in (src/sub).iterdir():os.link(p,a.output/sub/p.name)
 m=json.loads(json.dumps(old));m['parent_manifest']={'path':str(a.source),'sha256':sha(a.source)};r=next(r for r in m['rows']if r['camera']=='GB');z=np.load(r['raw_archive']);data={k:z[k]for k in z.files};nom=data['nominal'];n,l,s,c=cv2.connectedComponentsWithStats(nom.astype('uint8'),8);idx=[i for i in range(1,n)if s[i].tolist()==[1460,1887,17,15,215]];assert len(idx)==1;remove=l==idx[0];unknown=cv2.dilate(remove.astype('uint8'),np.ones((17,17),np.uint8))>0;nom=nom&~remove;veto=data['source_ineligible']|unknown;data.update(nominal=nom,source_ineligible=veto);p=a.output/'raw/GB.npz';p.unlink();np.savez_compressed(p,**data)
 row=r['camera_row'];vetoHD=raw_mask_coverage_to_HD(full(veto,data['crop_xyxy']),row)>1e-6;weights={};stats={}
 for policy,width in [('hard',0),('W32',32),('W64',64)]:
  w,st=native_confidence(nom,veto,width);HD=raw_mask_coverage_to_HD(full(w,data['crop_xyxy']),row);HD[vetoHD]=0;weights[policy]=HD;stats[policy]=st
 proposal=weights['W64']>0;p=a.output/'archives/GB.npz';p.unlink();np.savez_compressed(p,**weights,proposal=proposal,source_ineligible=vetoHD,nominal_coverage=weights['hard']);p=a.output/'masks/GB.png';p.unlink();Image.fromarray(proposal.astype('uint8')*255).save(p);core=(weights['hard']>=.999999)&proposal
 r.update(common_proposal_pixels=int(proposal.sum()),source_ineligible_pixels=int(vetoHD.sum()),native_stats=stats,HD_stats={pol:{'sum_weights':float(w.sum(dtype=np.float64)),'positive_pixels':int((w>0).sum()),'outside_core_weight':float(w[~core].sum(dtype=np.float64)),'outside_core_weight_ratio':float(w[~core].sum(dtype=np.float64)/max(w[core].sum(dtype=np.float64),1e-30))}for pol,w in weights.items()})
 for r in m['rows']:
  for key,folder in [('archive','archives'),('raw_archive','raw')]:r[key]=str(a.output/folder/Path(r[key]).name);r[key+'_sha256']=sha(r[key])
 for r in m['native_eval']:r['mask']=str(a.output/'native_eval_masks'/Path(r['mask']).name)
 tf=json.loads((src/'transforms.json').read_text())
 for f in tf['frames']:f['mask_path']=str(a.output/'masks'/f"{f['physical_camera']}.png")
 (a.output/'transforms.json').write_text(json.dumps(tf,indent=2)+'\n');m['data_dir']=str(a.output);m['transforms_sha256']=sha(a.output/'transforms.json');N=sum(r['common_proposal_pixels']for r in m['rows']if r['split']=='TRAIN');m['global_train_proposal_count']=N;m['normalizer_policy_train']={pol:sum(r['HD_stats'][pol]['sum_weights']for r in m['rows']if r['split']=='TRAIN')/N for pol in weights};m['manual_nominal_correction']={'camera':'GB','crop_component_bbox_xywh':[1460,1887,17,15],'exact_removed_RAW_pixels':215,'classification':'plain source floor, no person/hair/boot','actual_source_patch':str(src/'review_numerical/GB_components/5_area215.png'),'patch_sha256':sha(src/'review_numerical/GB_components/5_area215.png'),'local_8native_uncertainty_veto':True,'NO_generic_component_size_filter':True,'all_other_disconnected_boot_hair_components_preserved':True,'correction_script_sha256':sha(__file__)};m['approved_for_training']=False;(a.output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m['normalizer_policy_train']))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
