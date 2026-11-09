"""Bounded TRAIN-only fixed-reference-ray plane sweep, no geometric constraints applied."""
from pathlib import Path
import json,hashlib,sys
import cv2,numpy as np
from PIL import Image,ImageDraw
from audit_marshawn_geometry import projection,project_points
R=Path('/fsx/tmp/marshawn_001660_hd');O=R/'floor_plane_sweep_HD_PILOT'
H=np.arange(-.03,.03001,.001)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rays(row,uv):
 pose=np.array(row['transform_matrix']);D=np.c_[(uv[:,0]-row['cx'])/row['fl_x'],-(uv[:,1]-row['cy'])/row['fl_y'],-np.ones(len(uv))]@pose[:3,:3].T;return pose[:3,3],D

def sample(image,row,uv):
 # Canonical COLMAP centres are half-integer; OpenCV integer centres arezero.
 uv=np.asarray(uv);return cv2.remap(image,(uv[:,0]-.5).astype(np.float32).reshape(1,-1),(uv[:,1]-.5).astype(np.float32).reshape(1,-1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan')).ravel()

def sweep(reference,center,targets,rows,images,n,d,size):
 rr=np.arange(-(size//2),size//2+1);xx,yy=np.meshgrid(rr,rr);uv=center+np.c_[xx.ravel(),yy.ravel()];a=sample(images[reference],rows[reference],uv);mean=float(a.mean());std=float(a.std());a=a-mean;o,D=rays(rows[reference],uv);den=D@n;curves={};valid=[]
 for cam in targets:
  scores=[]
  for h in H:
   t=(h-d-o@n)/den;xyz=o+t[:,None]*D;u,front=project_points(rows[cam],xyz);b=sample(images[cam],rows[cam],u);norm=float(np.linalg.norm(a)*np.linalg.norm(b-b.mean()));score=float(a@(b-b.mean())/norm)if norm>1e-15 and np.isfinite(b).all()and front.all()and(t>0).all()else -1.;scores.append(score)
  curves[cam]=scores
 return curves,mean,std

def peak(curve):
 c=np.array(curve);idx=int(c.argmax());h=float(H[idx]);outside=abs(H-h)>=.005-1e-10;secondary=float(c[outside].max());width=float(np.ptp(H[c>=c[idx]-.03]));return dict(height=h,ZNCC=float(c[idx]),outside005_second_ZNCC=secondary,uniqueness_margin=float(c[idx]-secondary),nearpeak_width=width,at_sweep_edge=idx in[0,len(H)-1])

def assess(ref,center,fit,validation,rows,images,n,d,size,known=None):
 curves,mean,std=sweep(ref,center,fit+[validation],rows,images,n,d,size);joint=np.mean([curves[c]for c in fit],axis=0);p=peak(joint);per={c:peak(curves[c])for c in fit+[validation]};fitspread=float(np.ptp([per[c]['height']for c in fit]));idx=int(np.argmax(joint));validation_at=float(curves[validation][idx]);consistent=abs(per[validation]['height']-p['height'])<=.0050001;unique=p['uniqueness_margin']>=.03 and p['nearpeak_width']<=.0100001 and not p['at_sweep_edge'];photometric=p['ZNCC']>=.6 and validation_at>=.6;relative_texture=std/max(abs(mean),1e-12);known_error=abs(p['height']-known)if known is not None else None;ok=unique and photometric and fitspread<=.0050001 and consistent and relative_texture>.02 and (known_error is None or known_error<=.01)
 # Reverse sweep keeps target referencepixel fixed, independently warps patches back.
 o,D=rays(rows[ref],np.array(center)[None]);X=o+((p['height']-d-o@n)/(D[0]@n))*D[0];reverse_center=project_points(rows[fit[0]],X[None])[0][0];rev,_,_=sweep(fit[0],reverse_center,[ref,fit[1]],rows,images,n,d,size);reverse=peak(np.mean(list(rev.values()),axis=0));roundtrip=abs(reverse['height']-p['height']);ok&=roundtrip<=.0050001
 return dict(reference=ref,reference_pixel_COLMAP=np.asarray(center).tolist(),fit_targets=fit,independent_validation_view=validation,patch_size=size,reference_mean=mean,reference_std=std,relative_texture=relative_texture,joint= p,target_peaks=per,fit_target_height_spread=fitspread,validation_ZNCC_at_fit_height=validation_at,validation_height_agrees=bool(consistent),reverse_peak=reverse,roundtrip_height_delta=roundtrip,known_point_height=known,known_height_error=known_error,passed=bool(ok),curves=curves,joint_curve=joint.tolist())

def main():
 cv2.setNumThreads(2);O.mkdir(exist_ok=True);data=R/'color_variants/gain4_scalar_trainfit_poseeval/sdr_png';linear=R/'color_variants/gain4_scalar_trainfit_poseeval/linear_rec709_exr';tf=json.loads((data/'transforms.json').read_text());rows={q['physical_camera']:q for q in tf['frames']if q['file_path']in tf['train_filenames']};names=['MA','NA','NB','OA'];images={};sources=[]
 for name in names:
  path=linear/'images'/Path(rows[name]['file_path']).with_suffix('.exr').name;rgb=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)[...,::-1];image=(rgb@np.array([.2126,.7152,.0722])).astype(np.float32);images[name]=cv2.GaussianBlur(image,(3,3),.7);sources.append(dict(camera=name,path=str(path),sha256=sha(path)))
 proposal=json.loads((R/'floor_band_prior_PROPOSAL/proposal.json').read_text());n=np.array(proposal['fit_normal']);d=proposal['fit_offset'];xyz=np.load(R/'geometry/train_witnesses.npz')['xyz'];results=[]
 for pid in [8290,253,3738]:
  center=project_points(rows['MA'],xyz[pid,None])[0][0];known=float(xyz[pid]@n+d)
  for size in [7,11,15]:
   q=assess('MA',center,['NA','NB'],'OA',rows,images,n,d,size,known);q['known_point_id']=pid;results.append(q);print(pid,size,q['passed'],q['joint'],q['known_height_error'],q['validation_ZNCC_at_fit_height'],flush=True)
 # Strictpredeclaredsanitygate; do not force densefloor afterfailedknownpoint matching.
 primary=[q for q in results if q['patch_size']==11];passed=all(q['passed']for q in primary);report=dict(kind='BOUNDED_PLANE_SWEEP_READONLY_PILOT',results=results,plane_heights=H.tolist(),fixed_reference_pixel_patches=True,plane_normal=n.tolist(),plane_offset=d,scope='Parallelplane±.03world,step.001; referencepatch pixels remainFIXED, pointsintersectcandidateplanealongreferenceRAYS; photometrictexture only estimatesdepth',sanity_pass=passed,dense_domain_points_run=0,decision='Knownpointgatepassed; next128pointpilotallowed'if passed else'STOP_INCONCLUSIVE: knownfloorwitness primary11patchsanitynotallunique/consistent; no densification/prior/carve',criteria='Primary11patch: jointZNCC>=.6, independentthirdviewZNCC>=.6, fitpeakspread<=.005, heldoutpeakheightdifference<=.005, reverseheightdifference<=.005, peakmargin>=.03vsoutside±.005, nearpeakwidth<=.01, known3Dheightwithin.01, relativepatchstd/mean>.02, interiorpeak',source_records=sources,script_sha256=sha(__file__),normalization_sha256=sha(R/'data/preparation.json'),calibration_sha256=sha(data/'transforms.json'),no_eval_RGB_used=True,no_NN_or_prior_changes=True,annotations='Manualactualsourcecontact excludes8246rigfoot/hardware andAA3738actorocclusion; MA/NA/NB/OA chosenfloorvisibleknownpoints',limitation='Knownsparsefloorwitnesses notindependentphysicalGT; smoothparallel-plane patchassumption mayfailonwrinkles/viewdependentappearance. ZNCCnormalizespatchscale/offset, notspatialBRDFchanges.')
 (O/'HD_known_sanity.json').write_text(json.dumps(report,indent=2)+'\n')
 # Compact correlation profiles; imageaudit, notestimateddepthimageortrainingartifact.
 sheet=Image.new('RGB',(800,3*250),(25,25,25));draw=ImageDraw.Draw(sheet)
 for j,q in enumerate(primary):
  y=j*250;draw.text((8,y+3),str(q['known_point_id'])+' primary11 PASS='+str(q['passed'])+' knownh='+str(round(q['known_point_height'],4)),fill='white');box=[50,y+30,760,y+220];draw.rectangle(box,outline='gray')
  for name,c,color in [('joint',q['joint_curve'],'white')]+[(k,v,col)for(k,v),col in zip(q['curves'].items(),['red','lime','cyan'])]:
   coords=[(50+(h+.03)/.06*710,y+220-(s+1)/2*190)for h,s in zip(H,c)];draw.line(coords,fill=color,width=2)
  x=50+(q['known_point_height']+.03)/.06*710;draw.line((x,y+30,x,y+220),fill='yellow',width=2)
 sheet.save(O/'HD_known_ZNCC_profiles.png');print(report['decision'],flush=True)
if __name__=='__main__':main()
