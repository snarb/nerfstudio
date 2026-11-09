"""At most128fixed-ref MA TRAIN floor pixels; unchanged11px ZNCC sanity criteria."""
from pathlib import Path
import json,numpy as np,cv2
from PIL import Image,ImageDraw
from scipy.spatial import Delaunay
from audit_marshawn_geometry import project_points
from pilot_marshawn_floor_plane_sweep import R,O,H,sha,rays,assess

def main():
 cv2.setNumThreads(2);data=R/'color_variants/gain4_scalar_trainfit_poseeval/sdr_png';linear=R/'color_variants/gain4_scalar_trainfit_poseeval/linear_rec709_exr';tf=json.loads((data/'transforms.json').read_text());rows={q['physical_camera']:q for q in tf['frames']if q['file_path']in tf['train_filenames']};proposal=json.loads((R/'floor_band_prior_PROPOSAL/proposal.json').read_text());poly={q['camera']:np.array(q['clear_floor_polygon'],np.float32)for q in proposal['floor_source_visuals']};extra=json.loads((R/'native_png_random10k_floor_band_PRIOR_STAGED/directional_floor14k_DIAGNOSTIC/audit.json').read_text())['extra_train_floor_polygons_manual'];poly['LB']=np.array(extra['LB'],np.float32);n=np.array(proposal['fit_normal']);d=proposal['fit_offset'];z=np.load(R/'floor_band_prior_PROPOSAL/finite_domain_triangles_PROPOSAL.npz');tri=z['triangles_xyz'];cells={tuple(k)for k in z['valid_xy_cell_indices']};origin=z['grid_xy_origin'];step=float(z['grid_xy_step']);names=['MA','NA','NB','LB'];images={}
 for name in names:
  image=cv2.imread(str(linear/'images'/Path(rows[name]['file_path']).with_suffix('.exr').name),cv2.IMREAD_UNCHANGED)[...,::-1]@np.array([.2126,.7152,.0722]);images[name]=cv2.GaussianBlur(image.astype(np.float32),(3,3),.7)
 rng=np.random.RandomState(42);indices=rng.choice(len(tri),4096,p=z['triangle_area']/z['triangle_area'].sum());uv=rng.rand(4096,2);root=np.sqrt(uv[:,0]);bary=np.c_[1-root,root*(1-uv[:,1]),root*uv[:,1]];world=(tri[indices]*bary[:,:,None]).sum(1);ref=project_points(rows['MA'],world)[0];ref=np.floor(ref)+.5;unique=[];seen=set()
 for center in ref:
  key=tuple(center)
  if key in seen:continue
  seen.add(key);o,D=rays(rows['MA'],center[None]);X=o+(-(d+o@n)/(D[0]@n))*D[0];cell=tuple(np.floor((X[:2]-origin)/step).astype(int))
  if cell not in cells:continue
  ok=True
  for h in [H[0],H[-1]]:
   point=o+((h-d-o@n)/(D[0]@n))*D[0]
   for name in names:
    u,front=project_points(rows[name],point[None]);ok&=bool(front[0]and cv2.pointPolygonTest(poly[name],tuple(u[0]),True)>12)
  if ok:unique.append(center)
  if len(unique)==128:break
 results=[];f=np.load(R/'splatfacto_gain4/floor_geometry_diagnosis/manual_floor_fit.npz');allpoints=np.load(R/'geometry/train_witnesses.npz')['xyz'];support=allpoints[np.r_[f['point_indices'][f['selected_inliers']],8290]];hull=Delaunay(support[:,:2])
 for j,center in enumerate(unique):
  q=assess('MA',center,['NA','NB'],'LB',rows,images,n,d,11);q['candidate_id']=j;o,D=rays(rows['MA'],center[None]);X=o+((q['joint']['height']-d-o@n)/(D[0]@n))*D[0];q['candidate_XYZ']=X.tolist();q['inside_old16_XY_hull']=bool(hull.find_simplex(X[:2])>=0);results.append(q)
 accepted=[q for q in results if q['passed']];reasons={key:0 for key in ['low_texture','joint_low_score','independent_validation_low_score','fit_peak_disagreement','validation_peak_disagreement','nonunique_peak','roundtrip_disagreement','edge_peak']}
 for q in results:
  reasons['low_texture']+=q['relative_texture']<=.02;reasons['joint_low_score']+=q['joint']['ZNCC']<.6;reasons['independent_validation_low_score']+=q['validation_ZNCC_at_fit_height']<.6;reasons['fit_peak_disagreement']+=q['fit_target_height_spread']>.0050001;reasons['validation_peak_disagreement']+=not q['validation_height_agrees'];reasons['nonunique_peak']+=q['joint']['uniqueness_margin']<.03 or q['joint']['nearpeak_width']>.0100001;reasons['roundtrip_disagreement']+=q['roundtrip_height_delta']>.0050001;reasons['edge_peak']+=q['joint']['at_sweep_edge']
 report=dict(kind='READONLY_PHOTOMETRIC_DEPTH_PILOT_NOT_PRIOR',samples=len(results),accepted=len(accepted),accepted_outside_old16_support_hull=sum(not q['inside_old16_XY_hull']for q in accepted),points=results,reject_counts_overlapping=reasons,reference='MA',fit=['NA','NB'],validation='LB',primary_patch11_unchanged=True,thresholds_unchanged_from_known_sanity=True,protocol='Rootaccepted2flatknownsanities; crease3738remainsrejected and15pxwindowremainssensitivityonly. No parameter retuning for128interiorpoints.',sampling='Fixedseed42uniformareaΩ; quantize MAreferencepixelcentre; requireallh±.03centerprojections12pxinside4actualTRAINfloorpolygons. Keepreferencepixelpatchfixed.',calibration_is_assumed_given_rig_not_depth_GT=True,no_eval_RGB=True,no_NN_no_constraints_no_carve=True,script_sha256=sha(__file__),source_bindings={str(p):sha(p)for p in [O/'HD_known_sanity.json',R/'floor_band_prior_PROPOSAL/proposal.json',R/'floor_band_prior_PROPOSAL/finite_domain_triangles_PROPOSAL.npz',data/'transforms.json']})
 (O/'HD_128_pilot.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(O/'HD_128_candidates_NOT_PRIOR.npz',XYZ=np.array([q['candidate_XYZ']for q in results]),accepted=np.array([q['passed']for q in results]),reference_uv=np.array(unique))
 # Review actual sources for ALLaccepted (or12highestjointscore failures ifnone).
 chosen=accepted if accepted else sorted(results,key=lambda q:q['joint']['ZNCC'],reverse=True)[:12];(O/'pilot_manual_contact_ids.json').write_text(json.dumps([q['candidate_id']for q in chosen]))
 for start in range(0,len(chosen),4):
  sheet=Image.new('RGB',(4*165,4*185),(25,25,25));draw=ImageDraw.Draw(sheet)
  for rr,q in enumerate(chosen[start:start+4]):
   X=np.array(q['candidate_XYZ']);
   for cc,name in enumerate(names):
    u,v=project_points(rows[name],X[None])[0][0];image=Image.open(data/rows[name]['file_path']).crop((int(u)-64,int(v)-64,int(u)+64,int(v)+64)).convert('RGB');dr=ImageDraw.Draw(image);dr.ellipse((60,60,68,68),outline='red',width=2);sheet.paste(image,(cc*165,rr*185+43));draw.text((cc*165+3,rr*185+2),f"p{q['candidate_id']} {name} h{q['joint']['height']:+.3f}",fill='white');draw.text((cc*165+3,rr*185+17),f"NCC{q['joint']['ZNCC']:.2f} val{q['validation_ZNCC_at_fit_height']:.2f}",fill='white')
  sheet.save(O/f'HD128_review_{start//4+1:02d}.jpg',quality=95)
 print('POINTS',len(results),'accepted',len(accepted),'outsideoldhull',report['accepted_outside_old16_support_hull'],'reject',reasons,flush=True)
if __name__=='__main__':main()
