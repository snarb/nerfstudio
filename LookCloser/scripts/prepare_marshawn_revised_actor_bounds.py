"""Fresh foot-inclusive AABB proposals, source overlays before any render clip.
Manual boot pixels scope approximate floor-ray witnesses; the floor plane is a
scene assumption with generous bounds margin, not measured boot depth.
"""
from pathlib import Path
import json,itertools,hashlib
import numpy as np
from PIL import Image,ImageDraw
R=Path('/fsx/tmp/marshawn_001660_hd');A=R/'two_actor_isolation'
def main():
 out=A/'revised_actor_bounds/source_projection_v1';out.mkdir(parents=True,exist_ok=False);data=R/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png';tf=json.loads((data/'transforms.json').read_text());frames={r['physical_camera']:r for r in tf['frames']};old=np.asarray(json.loads((R/'geometry/boxes_approved.json').read_text())['scopes']['subject']['loose']);n=np.array([.0298,-.13036,.99102]);n/=np.linalg.norm(n);d=.16821
 # Actual TRAIN boot/sole centres manually viewed in full unmasked images.
 manual={'AA':[[1049,594],[1000,472],[1063,518],[1185,521]],'GA':[[841,486],[1049,457],[1008,550],[971,484]],'OA':[[1143,731],[1150,615],[1010,678],[1128,708]]};witness=[]
 for cam,pixels in manual.items():
  row=frames[cam];pose=np.asarray(row['transform_matrix']);origin=pose[:3,3]
  for j,(x,y)in enumerate(pixels):
   local=np.array([(x+.5-row['cx'])/row['fl_x'],-(y+.5-row['cy'])/row['fl_y'],-1.]);direction=pose[:3,:3]@local;direction/=np.linalg.norm(direction);t=-(origin@n+d)/(direction@n);xyz=origin+t*direction;witness.append({'camera':cam,'boot_scope_index':j,'pixel_array_xy':[x,y],'approx_floor_ray_hit_xyz':xyz.tolist(),'depth_NOT_measured':True})
 hits=np.asarray([r['approx_floor_ray_hit_xyz']for r in witness]);base=np.stack([np.minimum(old[0],hits.min(0)),np.maximum(old[1],hits.max(0))]);boxes={}
 for label,lower,upper in [('FOOT_SAFE', [.035,.035,.060],[.035,.035,.025]),('FOOT_WIDE',[.065,.065,.090],[.065,.065,.045]),('FOOT_MAX',[.10,.10,.12],[.10,.10,.07])]:boxes[label]=[ (base[0]-lower).tolist(),(base[1]+upper).tolist()]
 records=[];tiles=[]
 for row in tf['frames']:
  cam=row['physical_camera'];im=Image.open(data/row['file_path']).convert('RGB');pose=np.asarray(row['transform_matrix']);draw=ImageDraw.Draw(im);projections={}
  for label,color in [('FOOT_MAX','yellow'),('FOOT_WIDE','cyan'),('FOOT_SAFE','red')]:
   b=np.array(boxes[label]);corners=np.array(list(itertools.product(*zip(b[0],b[1]))));v=(corners-pose[:3,3])@pose[:3,:3];uv=np.c_[row['fl_x']*v[:,0]/-v[:,2]+row['cx'],row['fl_y']*(-v[:,1])/-v[:,2]+row['cy']];projections[label]=uv.tolist()
   for i in range(8):
    for j in range(i+1,8):
     if np.count_nonzero(corners[i]!=corners[j])==1:draw.line([tuple(uv[i]),tuple(uv[j])],fill=color,width=2)
  for p in manual.get(cam,[]):draw.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5),outline='lime',width=2)
  im.save(out/(cam+'_FULL_SOURCE_boxes.jpg'),quality=94);uv=np.asarray(projections['FOOT_MAX']);mn=np.maximum(np.floor(uv.min(0)-25),0).astype(int);mx=np.minimum(np.ceil(uv.max(0)+25),[im.width,im.height]).astype(int);crop=im.crop((*mn,*mx));crop.thumbnail((480,480));panel=Image.new('RGB',(480,510));panel.paste(crop,((480-crop.width)//2,25));ImageDraw.Draw(panel).text((3,3),cam+' redSAFE/cyanWIDE/yellowMAX fullsource',fill='white');tiles.append(panel);records.append({'camera':cam,'source':str(data/row['file_path']),'projections':projections})
 for k in range(0,len(tiles),12):
  page=Image.new('RGB',(1920,1530))
  for j,tile in enumerate(tiles[k:k+12]):page.paste(tile,((j%4)*480,(j//4)*510))
  page.save(out/('contact_%02d.jpg'%(k//12+1)),quality=94)
 receipt={'kind':'REVISED_FOOT_INCLUSIVE_PROPOSALS_NOT_APPROVED','approved_for_training':False,'old_boxes_user_rejected':True,'scope':'ONLY two central fighting actors; other lying/background people excluded','source_transforms_sha256':hashlib.sha256((data/'transforms.json').read_bytes()).hexdigest(),'boxes':boxes,'manual_boot_scope_observations':witness,'floor_model_assumption':{'n':n.tolist(),'d':d,'not_measured_boot_depth':True,'manual_scope_pixel_identity_not_multi_view_feature_correspondence':True},'new_base_envelope_includes_old_loose_and_approx_floor_hits':base.tolist(),'explicit_extra_lower_margin_world':[.06,.09,.12],'no_teacher_clip_in_images':True,'records':records};(out/'proposals.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(boxes,indent=2))
if __name__=='__main__':main()
