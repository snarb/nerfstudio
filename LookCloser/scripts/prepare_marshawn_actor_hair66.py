"""Native1:1 RAW hair-scope tiles for ALL66; visibility remains manual."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
from PIL import Image
R=Path('/fsx/tmp/marshawn_001660_hd');O=R/'two_actor_isolation'
def main():
 sys.path[:0]=[str(R/'code_native'),str(Path(__file__).parent)]
 from marshawn_actor_coordinates import raw_array_indices
 D=O/'native_hair66_sources';D.mkdir(exist_ok=False);req=json.loads((O/'raw_native_sources/request.json').read_text());identity={r['camera']:r for r in json.loads((O/'actor_identity66.json').read_text())['rows']};tf=json.loads((R/'data/transforms.json').read_text());frames={f['physical_camera']:f for f in tf['frames']};P=np.load(O/'geometry_precheck/strict_subject_candidate_points.npz')['xyz'];head=P[P[:,2]>=.15];assert len(head)>=10;rows=[]
 for old in req['records']:
  cam=old['camera'];f=frames[cam];p=np.array(f['transform_matrix']);q=(head-p[:3,3])@p[:3,:3];uv=np.c_[f['fl_x']*q[:,0]/-q[:,2]+f['cx'],f['fl_y']*(-q[:,1])/-q[:,2]+f['cy']];raw=raw_array_indices(uv,f);valid=(q[:,2]<0)&(raw[:,0]>=0)&(raw[:,0]<8192)&(raw[:,1]>=0)&(raw[:,1]<4320);centre=np.median(raw[valid]if valid.any()else raw,axis=0);bx0,by0,bx1,by1=old['crop_xyxy'];x0=int(np.clip(round(centre[0])-504,bx0,bx1-1008));y0=int(np.clip(round(centre[1])-504,by0,by1-1008));im=Image.open(old['image']).convert('RGB').crop((x0-bx0,y0-by0,x0-bx0+1008,y0-by0+1008));assert im.size==(1008,1008);path=D/(cam+'_hair_native.png');im.save(path)
  who=identity[cam];m=np.load(who['candidates_npz'])['masks'];black=m[who['black_actor_instance']];union=m[[who['black_actor_instance'],who['grey_actor_instance']]].any(0);xx=x0-bx0;yy=y0-by0;np.savez_compressed(D/(cam+'_coarse.npz'),black=black[yy:yy+1008,xx:xx+1008],two_actor_union=union[yy:yy+1008,xx:xx+1008]);new=dict(old);new.update(image=str(path),image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),crop_xyxy=[x0,y0,x0+1008,y0+1008],crop_size=[1008,1008],parent_actor_crop=old['crop_xyxy'],parent_image=old['image'],input_crop_kind='native1to1_RAW_hair_scope',scope_measured_headpoint_count=int(valid.sum()),scope_headpoint_cohort_height_threshold=.15,source_head_visibility='UNREVIEWED_maybetruncatedoroutside_sensor_NOTtorsoashair',black_instance=who['black_actor_instance'],grey_instance=who['grey_actor_instance'],coarse_reference=str(D/(cam+'_coarse.npz')));rows.append(new)
 (D/'request.json').write_text(json.dumps({'records':rows,'source_raster':[8192,4320],'every_inference_tile1008_nativepixels':True,'NOinputupsample':True,'geometry_only_scopes_NOTsegmentation_truth':True,'actual_source_visibility_manualgate_required':True,'instancesafe_update':'BLACKselectednativeTEXTproposalADDONLY≤32nativeoutsideCOARSEtwoactorunion; neversubtractGRAY/noopacityGT; mask is RGBconfidence','no_NN_training':True},indent=2)+'\n');print('all66hairtiles',len(rows),'headcohort',len(head))
if __name__=='__main__':main()
