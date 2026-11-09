"""Source-bound actor confidence artifacts. Never alpha labels or edited RGB.

A provisional build is deliberately ineligible for training; final release also
requires a separately reviewed native hair decision for every physical camera.
"""
from pathlib import Path
import argparse,copy,hashlib,json
import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from marshawn_actor_confidence import native_confidence
from marshawn_actor_coordinates import raw_mask_coverage_to_HD,raw_mask_to_native_undistorted
R=Path('/fsx/tmp/marshawn_001660_hd');A=R/'two_actor_isolation'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def full(crop,box):
 out=np.zeros((4320,8192),np.float32);x0,y0,x1,y1=box;assert crop.shape==(y1-y0,x1-x0);out[y0:y1,x0:x1]=crop;return out

def build(a):
 cv2.setNumThreads(2);a.output.mkdir(parents=True,exist_ok=False)
 for sub in ['archives','raw','masks','native_eval_masks','review']: (a.output/sub).mkdir()
 data=R/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png';tf=json.loads((data/'transforms.json').read_text());raw=json.loads((A/'raw_native_sources/request.json').read_text());raw={r['camera']:r for r in raw['records']};ident=json.loads((A/'actor_identity66.json').read_text());ids=ident['rows'];decisions=json.loads(a.hair_review.read_text())if a.hair_review else {'rows':[]};hair={r['camera']:r for r in decisions['rows']};frames={r['physical_camera']:r for r in tf['frames']};nativecam={r['physical_camera']:r for r in json.loads((A/'native_eval_RGB/cameras.json').read_text())['frames']};totals={p:0. for p in ['hard','W32','W64']};N=0;rows=[];native_eval=[]
 for ident in ids[a.start:a.stop]:
  cam=ident['camera'];src=raw[cam];row=frames[cam];p=Path(ident['candidates_npz']);assert sha(p)==ident['source_candidate_npz_sha256'];z=np.load(p);m=z['masks'];black=m[int(ident['black_actor_instance'])];grey=m[int(ident['grey_actor_instance'])];nominal=black|grey;other=m[ident['verified_other_person_ids']].any(0)if ident['verified_other_person_ids']else np.zeros_like(nominal);unknown=np.zeros_like(nominal);hair_added=0;decision=hair.get(cam)
  # Explicit source-occlusion masks may veto; ambiguous duplicate IDs alone NEVER do.
  if decision and decision.get('source_unknown_mask'):
   unknown=np.asarray(Image.open(decision['source_unknown_mask']))>0;assert unknown.shape==nominal.shape
  veto=other|unknown
  if decision and decision.get('accept_add_only',False):
   hz=np.load(decision['candidate_archive']);candidate=hz['masks'][int(decision['selected_instance'])];tilebox=decision['crop_xyxy'];box=src['crop_xyxy'];ox=tilebox[0]-box[0];oy=tilebox[1]-box[1];assert ox>=0 and oy>=0 and ox+candidate.shape[1]<=nominal.shape[1] and oy+candidate.shape[0]<=nominal.shape[0];proposal=np.zeros_like(nominal);proposal[oy:oy+candidate.shape[0],ox:ox+candidate.shape[1]]=candidate;distance=distance_transform_edt(~nominal);addition=proposal&~nominal&(distance<=32)&~veto;hair_added=int(addition.sum());nominal|=addition
  rawpath=a.output/'raw'/f'{cam}.npz';np.savez_compressed(rawpath,nominal=nominal,source_ineligible=veto,black=black,grey=grey,crop_xyxy=np.asarray(src['crop_xyxy'],np.int32))
  rawv=full(veto,src['crop_xyxy']);vetoHD=raw_mask_coverage_to_HD(rawv,row)>1e-6;del rawv
  weights={};stats={}
  for policy,width in [('hard',0),('W32',32),('W64',64)]:
   wc,st=native_confidence(nominal,veto,width);HD=raw_mask_coverage_to_HD(full(wc,src['crop_xyxy']),row);HD[vetoHD]=0;weights[policy]=HD;stats[policy]=st
  proposal=weights['W64']>0;assert not np.any(proposal&vetoHD);archive=a.output/'archives'/f'{cam}.npz';np.savez_compressed(archive,**weights,proposal=proposal,source_ineligible=vetoHD,nominal_coverage=weights['hard']);maskpath=a.output/'masks'/f'{cam}.png';Image.fromarray(proposal.astype(np.uint8)*255).save(maskpath)
  semanticHD=raw_mask_coverage_to_HD(full(nominal,src['crop_xyxy']),row)>=.5;semanticHD&=~vetoHD
  if cam in nativecam:
   native,nrow=raw_mask_to_native_undistorted(full(nominal,src['crop_xyxy']),row);nv,_=raw_mask_to_native_undistorted(full(veto,src['crop_xyxy']),row);emask=(native>=.5)&(nv<=1e-6);mp=a.output/'native_eval_masks'/f'{cam}.png';Image.fromarray(emask.astype(np.uint8)*255).save(mp);rgb=A/'native_eval_RGB'/nativecam[cam]['file_path'];native_eval.append({'camera':cam,'rgb':str(rgb),'rgb_sha256':sha(rgb),'mask':str(mp),'mask_sha256':sha(mp),'camera_row':nativecam[cam],'foreground_pixels':int(emask.sum()),'domain_not_opacity_GT':True})
  istrain=Path(row['file_path']).name.startswith('train_');count=int(proposal.sum());core=(weights['hard']>=.999999)&proposal;hdstats={pol:{'sum_weights':float(w.sum(dtype=np.float64)),'positive_pixels':int((w>0).sum()),'outside_core_weight':float(w[~core].sum(dtype=np.float64)),'outside_core_weight_ratio':float(w[~core].sum(dtype=np.float64)/max(w[core].sum(dtype=np.float64),1e-30))}for pol,w in weights.items()}
  if istrain:
   N+=count
   for pol in totals:totals[pol]+=hdstats[pol]['sum_weights']
  rgb=data/row['file_path'];rows.append({'camera':cam,'split':'TRAIN'if istrain else'EVAL','rgb':str(rgb),'rgb_sha256':sha(rgb),'archive':str(archive),'archive_sha256':sha(archive),'raw_archive':str(rawpath),'raw_archive_sha256':sha(rawpath),'common_proposal_pixels':count,'source_ineligible_pixels':int(vetoHD.sum()),'native_hair_added_pixels':hair_added,'native_hair_reviewed':bool(decision and decision.get('manually_reviewed',False)),'native_stats':stats,'HD_stats':hdstats,'camera_row':row,'identity_receipt':ident['identity_receipt']})
  source=np.asarray(Image.open(rgb).convert('RGB'));panels=[]
  for label,img in [('RGB',source),('nominal',np.where(semanticHD[...,None],source,source//4)),('W64',np.clip(source*.4+weights['W64'][...,None]*np.array([0,150,0]),0,255).astype(np.uint8)),('veto',np.where(vetoHD[...,None],np.array([255,0,0],np.uint8),source))]:
   im=Image.fromarray(img);im.thumbnail((480,253));panel=Image.new('RGB',(480,278));panel.paste(im,(0,25));ImageDraw.Draw(panel).text((4,4),cam+' '+label,fill='white');panels.append(panel)
  sheet=Image.new('RGB',(960,556))
  for j,im in enumerate(panels):sheet.paste(im,((j%2)*480,(j//2)*278))
  sheet.save(a.output/'review'/f'{cam}.jpg');print(json.dumps({'camera':cam,'proposal':count,'native_hair_added':hair_added,'W64_outside_ratio':hdstats['W64']['outside_core_weight_ratio']}),flush=True)
 # Preserve canonical image bytes, calibration and split. Only proposal mask path changes.
 subset={r['camera']for r in rows};outtf=copy.deepcopy(tf);outtf['frames']=[r for r in outtf['frames']if r['physical_camera']in subset]
 for row in outtf['frames']:
  row['file_path']=str(data/row['file_path']);row['mask_path']=str(a.output/'masks'/f"{row['physical_camera']}.png")
 
 for key in ['train_filenames','val_filenames','test_filenames']:
  if key in outtf:outtf[key]=[str(data/f)for f in outtf[key]]
 (a.output/'transforms.json').write_text(json.dumps(outtf,indent=2)+'\n');manifest={'kind':'TWO_ACTOR_RGB_CONFIDENCE_SOURCE_BOUND','approved_for_training':False,'approval_reason':'Requires all66 native-hair/manual occlusion/boundary and geometry review; this build never self-authorizes','data_dir':str(a.output),'source_data_dir':str(data),'source_transforms_sha256':sha(data/'transforms.json'),'transforms_sha256':sha(a.output/'transforms.json'),'identity_manifest_sha256':sha(A/'actor_identity66.json'),'hair_review':str(a.hair_review)if a.hair_review else None,'hair_review_sha256':sha(a.hair_review)if a.hair_review else None,'helper_sha256':sha(Path(__file__).with_name('marshawn_actor_confidence.py')),'builder_sha256':sha(__file__),'aabb':None,'aabb_selection':'OLD subject boxes USERREJECTED; arrays AABB-independent, revised physicalbox ROOTapproval required','rows':rows,'native_eval':native_eval,'global_train_proposal_count':N,'normalizer_policy_train':{p:totals[p]/max(1,N)for p in totals},'sampling':'GLOBALuniformcommonW64proposal; confidence weightedONCE/fixedTRAINglobalmean','policy_widths_native_pixels':{'hard':0,'W32':32,'W64':64},'source_RGB_unchanged':True,'no_alpha_GT':True,'uses_eval_rgb_to_fit':False,'veto_policy':'verifiedother-person union OR explicit source-occlusion UNKNOWN; ANY calibratedprefiltercoverage>1e-6 overridesall HD weights; ambiguousduplicateIDs not automatic veto','native_hair66_complete':len(rows)==66 and all(r['native_hair_reviewed']for r in rows)};(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--hair-review',type=Path);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=None);build(p.parse_args())
