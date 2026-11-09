"""Actor-only geometry review candidates; no segmentation or training claim."""
from pathlib import Path
import argparse,json,hashlib,itertools
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
R=Path('/fsx/tmp/marshawn_001660_hd')
def main(a):
 a.output.mkdir(parents=True,exist_ok=False);tf=json.loads((a.data/'transforms.json').read_text());box=json.loads((R/'geometry/boxes_approved.json').read_text())['scopes']['subject'];subjects=np.load(R/'geometry/subject_witnesses.npz')['xyz'];xyz=np.load(R/'geometry/train_witnesses.npz')['xyz'];dist,idx=cKDTree(xyz).query(subjects);unique=np.unique(idx[dist<1e-8]);points=xyz[unique];records=[];tiles=[]
 for f in tf['frames']:
  pose=np.array(f['transform_matrix']);im=Image.open(a.data/f['file_path']).convert('RGB');full=im.copy();draw=ImageDraw.Draw(full);alluv={}
  for label,col in [('loose','yellow'),('medium','cyan'),('tight','magenta')]:
   b=np.array(box[label]);corners=np.array(list(itertools.product(*zip(b[0],b[1]))));v=(corners-pose[:3,3])@pose[:3,:3];z=-v[:,2];assert (z>0).all();u=f['fl_x']*v[:,0]/z+f['cx'];vv=f['fl_y']*(-v[:,1])/z+f['cy'];uv=np.c_[u,vv];alluv[label]=uv
   for i in range(8):
    for j in range(i+1,8):
     if np.count_nonzero(corners[i]!=corners[j])==1:draw.line([tuple(uv[i]),tuple(uv[j])],fill=col,width=2)
  # calibrated projected witness points, semantic attribution NOT assumed
  local=(points-pose[:3,3])@pose[:3,:3];pp=np.c_[f['fl_x']*local[:,0]/-local[:,2]+f['cx'],f['fl_y']*(-local[:,1])/-local[:,2]+f['cy']]
  for x,y in pp[::3]:draw.ellipse((x-1,y-1,x+1,y+1),fill='#ff8c00')
  uv=alluv['loose'];mn=np.floor(uv.min(0)-[45,45]).astype(int);mx=np.ceil(uv.max(0)+[45,45]).astype(int);mn=np.maximum(mn,0);mx=np.minimum(mx,[im.width,im.height]);crop=full.crop((*mn,*mx));crop.thumbnail((400,440));tile=Image.new('RGB',(400,470),'#151515');tile.paste(crop,((400-crop.width)//2,28));dr=ImageDraw.Draw(tile);dr.text((4,3),f['physical_camera']+' '+('TRAIN'if f['file_path']in tf['train_filenames']else'EVAL')+' yellowL/cyanM/magentaT',fill='white');tile.save(a.output/(f['physical_camera']+'_review.jpg'));tiles.append(tile)
  full.save(a.output/(f['physical_camera']+'_full_overlay.jpg'),quality=90);im.crop((*mn,*mx)).save(a.output/(f['physical_camera']+'_source_crop.png'))
  records.append({'camera':f['physical_camera'],'split':'TRAIN'if f['file_path']in tf['train_filenames']else'EVAL','source':str(a.data/f['file_path']),'source_sha256':hashlib.sha256((a.data/f['file_path']).read_bytes()).hexdigest(),'crop_xyxy':[*mn.tolist(),*mx.tolist()],'projected_boxes':{k:v.tolist()for k,v in alluv.items()},'visibility':'Mustinspectactualsource; projectedAABBisnotforegroundmask'})
 for p in range((len(tiles)+11)//12):
  page=Image.new('RGB',(4*400,3*470),'black')
  for j,t in enumerate(tiles[p*12:p*12+12]):page.paste(t,((j%4)*400,(j//4)*470))
  page.save(a.output/('contact_%02d.jpg'%(p+1)),quality=92)
 report={'kind':'TWO_ACTOR_BBOX_REVIEW_NOT_SEGMENTATION','all66_views':len(records),'provisional_candidates':box,'raw_subject_points':len(subjects),'exact_strict_geometry_matched_points':len(unique),'unmatched_subject_points':np.where(dist>=1e-8)[0].tolist(),'max_nearest_distance':float(dist.max()),'matched_original_pointIDs':unique.tolist(),'body_point_semantics_not_assumed':True,'rays_intersecting_box_not_actor_only_RGB':True,'records':records,'no_training':True,'no_masks_generated':True,'transforms_sha256':hashlib.sha256((a.data/'transforms.json').read_bytes()).hexdigest()};(a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(a.output/'strict_subject_candidate_points.npz',xyz=points,original_point_ids=unique);print(json.dumps({k:report[k]for k in ['all66_views','raw_subject_points','exact_strict_geometry_matched_points','unmatched_subject_points','max_nearest_distance']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=R/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png');p.add_argument('--output',type=Path,required=True);main(p.parse_args())
