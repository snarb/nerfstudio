"""PROPOSAL ONLY: finite, train-photo-scoped soft floor-band hypothesis."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np,cv2
from scipy.spatial import Delaunay,cKDTree
from PIL import Image,ImageDraw
from audit_marshawn_geometry import project_points
R=Path('/fsx/tmp/marshawn_001660_hd');OUT=R/'floor_band_prior_PROPOSAL'
POLYGONS={
'MA':[[285,440],[405,430],[430,475],[435,555],[395,620],[275,615],[245,550]],
'NA':[[390,445],[510,440],[535,495],[535,570],[505,610],[380,610],[360,550]],
'NB':[[410,430],[535,430],[555,480],[555,555],[510,600],[405,600],[385,540]],
'AB':[[1760,540],[1890,540],[1890,800],[1760,800]],
'BB':[[1660,540],[1810,540],[1810,790],[1660,790]]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fit(p):
 c=p.mean(0);_,_,v=np.linalg.svd(p-c);n=v[-1];n*=np.sign(n[2]);return n,-float(n@c)
def main():
 OUT.mkdir(exist_ok=True);data=R/'color_variants/gain4_scalar_trainfit_poseeval/sdr_png';tf=json.loads((data/'transforms.json').read_text());rows={q['physical_camera']:q for q in tf['frames']if q['file_path']in tf['train_filenames']};assert set(POLYGONS)<=set(rows)
 f=np.load(R/'splatfacto_gain4/floor_geometry_diagnosis/manual_floor_fit.npz');p=np.load(R/'geometry/train_witnesses.npz')['xyz'];ids=np.r_[f['point_indices'][f['selected_inliers']],8290];support=p[ids];n,d=fit(support);res=support@n+d;loo=[]
 for j,idx in enumerate(ids):
  nn,dd=fit(np.delete(support,j,axis=0));loo.append(dict(heldout_point_id=int(idx),heldout_signed_distance=float(support[j]@nn+dd),normal_degrees=float(np.degrees(np.arccos(np.clip(n@nn,-1,1))))))
 old=json.loads((R/'native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit/floor_reference_support.json').read_text());hits=np.array([q['reference_plane_XYZ']for q in old['rows']]);low=hits[:,:2].min(0)-.08;high=hits[:,:2].max(0)+.08;xx,yy=np.meshgrid(np.arange(low[0],high[0]+.002,.004),np.arange(low[1],high[1]+.002,.004));xy=np.stack([xx.ravel(),yy.ravel()],1);centers=np.c_[xy,-(xy@n[:2]+d)/n[2]];half=.02;vote=[]
 for name,poly in POLYGONS.items():
  row=rows[name];poly=np.array(poly,np.float32);ok=np.ones(len(centers),bool)
  for v in [-half,0,half]:
   uv,front=project_points(row,centers+v*n);inside=np.array([cv2.pointPolygonTest(poly,tuple(u),True)>8 for u in uv]);ok&=front&inside
  vote.append(ok)
 vote=np.stack(vote,1);keep=vote.sum(1)>=3;centers=centers[keep];vote=vote[keep];assert len(centers)>0
 # Membership in the sampled finite XY patch uses nearest accepted grid centre, never outside patch.
 tree=cKDTree(centers[:,:2]);dist,j=tree.query(p[:,:2]);in_domain=dist<=.004/np.sqrt(2)+1e-8;height=p@n+d;selected=np.flatnonzero(in_domain);hitdist,hj=tree.query(hits[:,:2]);inside_hull=Delaunay(support[:,:2]).find_simplex(centers[:,:2])>=0
 np.savez_compressed(OUT/'soft_floor_bands_PROPOSAL.npz',centers=centers,normal=n,halfwidth=half,band_start=centers-half*n,band_end=centers+half*n,clear_floor_votes=vote,source_cameras=np.array(list(POLYGONS)),inside_measured_support_XY_hull=inside_hull)
 visuals=[]
 for name,poly in POLYGONS.items():
  row=rows[name];src=data/row['file_path'];im=Image.open(src).convert('RGB');draw=ImageDraw.Draw(im);draw.line([tuple(x)for x in poly+[poly[0]]],fill='yellow',width=3)
  uv,_=project_points(row,centers[::max(1,len(centers)//300)]);uvlo,_=project_points(row,centers[::max(1,len(centers)//150)]-half*n);uvhi,_=project_points(row,centers[::max(1,len(centers)//150)]+half*n)
  for pts,color in [(uv,'lime'),(uvlo,'blue'),(uvhi,'red')]:
   for x,y in pts:draw.ellipse((x-1,y-1,x+1,y+1),fill=color)
  for k,(x,y)in enumerate(project_points(row,hits)[0]):draw.text((x+3,y),str(k),fill='magenta')
  im.save(OUT/(name+'_prior_projection.jpg'),quality=95);center=np.median(uv,0);x,y=center;bbox=[max(0,int(x)-280),max(0,int(y)-240),min(1920,int(x)+280),min(1012,int(y)+240)];im.crop(bbox).save(OUT/(name+'_prior_local.jpg'),quality=95);visuals.append(dict(camera=name,source=str(src),source_sha256=sha(src),clear_floor_polygon=poly,accepted_center_votes=int(vote[:,list(POLYGONS).index(name)].sum()),local_crop_bbox=bbox))
 receipt=dict(kind='FINITE_SOFT_GEOMETRIC_HYPOTHESIS_NOT_MEASURED_DEPTH',approved_for_training=False,source_train_floor_ids=ids.tolist(),fit_normal=n.tolist(),fit_offset=d,signed_fit_residuals=res.tolist(),fit_abs_median=float(np.median(abs(res))),fit_abs_max=float(max(abs(res))),leave_one_out=loo,leave_one_out_max_abs=float(max(abs(q['heldout_signed_distance'])for q in loo)),band_halfwidth_prepared_world=half,domain='Finite XY sampling .004world around9oldplane intersections±.08world; retain only>=3trainphotos whereBOTHnormalbandendpoints+centre lie inside hand-annotated clearfloorpolygon with8HDpixel erosion',samples=len(centers),inside_measured_support_hull_samples=int(inside_hull.sum()),outside_measured_support_hull_samples=int((~inside_hull).sum()),all9_referencehit_distances_to_domain=hitdist.tolist(),floor_source_visuals=visuals,all9022_audit=dict(in_finite_domain=len(selected),point_ids=selected.tolist(),signed_plane_heights=height[selected].tolist(),within_band=int((abs(height[selected])<=half).sum()),above_band=int((height[selected]>half).sum()),below_band=int((height[selected]<-half).sum())),no_eval_RGB_or_gains_fitted=True,no_nn_training=True,no_new_measured_points=True,no_los_negative_constraints=True,no_carving=True,physical_claim_outside_support_hull=False,limitations=['A planar floor band outside sparse support hull is an explicit scene prior, not measured geometry.','CleartrainRGB floor observations constrain semantic scope only; no independent depth certificate.','Floor coverings/panels may warp or change height. Bandwidth.02 is a chosensoftassumption, not calibrated confidence.','Existing points above band may be actors/props and are not contradictions; belowband points requiremanualreview before use.','No presumed opaque floor is used to generate free/belowfloor constraints.'],sha256={str(x):sha(x)for x in [Path(__file__),R/'geometry/train_witnesses.npz',R/'splatfacto_gain4/floor_geometry_diagnosis/manual_floor_fit.npz',R/'native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit/raw8k_floor_sfm/manual_review_geometry.json',data/'transforms.json',OUT/'soft_floor_bands_PROPOSAL.npz']})
 (OUT/'proposal.json').write_text(json.dumps(receipt,indent=2)+'\n');print('samples',len(centers),'insidehull',inside_hull.sum(),'witnesses',receipt['all9022_audit'],'fit',n,d,'maxres',max(abs(res)))
if __name__=='__main__':main()
