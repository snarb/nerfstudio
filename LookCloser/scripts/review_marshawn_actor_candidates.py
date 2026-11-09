"""Source-bound review aids; geometry votes suggest IDs but do not accept masks."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from PIL import Image,ImageDraw
R=Path('/fsx/tmp/marshawn_001660_hd')
def main(a):
 sys.path[:0]=[str(R/'code_native'),str(Path(__file__).parent)]
 from marshawn_actor_coordinates import raw_array_indices
 req=json.loads(a.request.read_text());tf=json.loads((R/'data/transforms.json').read_text());rows={f['physical_camera']:f for f in tf['frames']};g=np.load(R/'two_actor_isolation/geometry_precheck/strict_subject_candidate_points.npz');pts=g['xyz'];ids=g['original_point_ids'];obs=np.load(R/'free_space_candidates/scene_loose.npz');records=[];tiles=[]
 for row in req['records']:
  cam=row['camera'];path=a.candidates/cam/'candidates.npz'
  if not path.exists():continue
  z=np.load(path);m=z['masks'];scores=z['scores'];f=rows[cam];p=np.array(f['transform_matrix']);q=(pts-p[:3,3])@p[:3,:3];uv=np.c_[f['fl_x']*q[:,0]/-q[:,2]+f['cx'],f['fl_y']*(-q[:,1])/-q[:,2]+f['cy']];raw=raw_array_indices(uv,f)-row['crop_xyxy'][:2];ix=np.rint(raw).astype(int);valid=(q[:,2]<0)&(ix[:,0]>=0)&(ix[:,0]<m.shape[2])&(ix[:,1]>=0)&(ix[:,1]<m.shape[1]);observed=np.isin(ids,obs['point_idx'][obs['camera_idx']==list(obs['camera_names']).index(cam)])if cam in obs['camera_names']else np.ones(len(ids),bool);valid&=observed;ix=ix[valid];votes=[int(mm[ix[:,1],ix[:,0]].sum())for mm in m];order=sorted(range(len(m)),key=lambda i:(votes[i],scores[i]),reverse=True);image=Image.open(row['image']).convert('RGB');base=np.asarray(image);cell=[]
  for k in order[:4]:
   ov=base.copy();ov[m[k]]=(ov[m[k]]*.55+np.array([0,255,255])*.45).astype('uint8');im=Image.fromarray(ov);im.thumbnail((360,440));tile=Image.new('RGB',(360,465));tile.paste(im,((360-im.width)//2,25));ImageDraw.Draw(tile).text((4,4),'%s ID%d votes%d s%.3f'%(cam,k,votes[k],scores[k]),fill='white');cell.append(tile)
  page=Image.new('RGB',(4*360,465))
  for k,t in enumerate(cell):page.paste(t,(k*360,0))
  page.save(a.candidates/cam/'geometry_vote_review.jpg');tiles.append(page);records.append({'camera':cam,'geometry_vote_counts':votes,'geometry_vote_order':order,'suggested_two_NOTaccepted':order[:2],'votes_use_actual_train_trackcameras':cam in obs['camera_names'],'eval_votes_are_projection_only':cam not in obs['camera_names'],'bodypoints_not_semanticGT':True})
 for p in range((len(tiles)+3)//4):
  out=Image.new('RGB',(4*360,4*465))
  for j,t in enumerate(tiles[p*4:p*4+4]):out.paste(t,(0,j*465))
  out.save(a.candidates/('vote_contact_%02d.jpg'%(p+1)))
 (a.candidates/'geometry_vote_review.json').write_text(json.dumps({'records':records,'no_automatic_selection':True},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True);main(p.parse_args())
