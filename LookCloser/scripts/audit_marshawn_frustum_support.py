#!/usr/bin/env python3
"""Conservative calibration-only voxel/frustum support; no field or image reads."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np


def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()


def camera_planes(camera,near=.052):
 """Inside n.X+b>=0. OpenGL camera right/up/back, image y downward."""
 M=np.asarray(camera['transform_matrix'],float);R=M[:3,:3];C=M[:3,3]
 if M.shape not in [(3,4),(4,4)]or not np.isfinite(M).all():raise ValueError('Finite c2w required')
 if not np.allclose(R.T@R,np.eye(3),rtol=0,atol=1e-6)or np.linalg.det(R)<.999999:raise ValueError('Rigid right-handed c2w required')
 fx,fy,cx,cy,w,h=[float(camera[k])for k in ['fl_x','fl_y','cx','cy','w','h']]
 if min(fx,fy,w,h,near)<=0:raise ValueError('Positive intrinsics/raster/near required')
 right,up,forward=R[:,0],R[:,1],-R[:,2]
 normals=np.array([forward,fx*right+cx*forward,-fx*right+(w-cx)*forward,-fy*up+cy*forward,fy*up+(h-cy)*forward])
 offsets=-normals@C;offsets[0]-=near
 norms=np.linalg.norm(normals,axis=1)
 return np.column_stack([normals,offsets])/norms[:,None]


def conservative_axial_near(camera,ray_distance_near=.052):
 """Native near is Euclidean ray distance; bound its raster-corner axial depth."""
 x=max(abs(float(camera['cx'])),abs(float(camera['w'])-float(camera['cx'])))/float(camera['fl_x'])
 y=max(abs(float(camera['cy'])),abs(float(camera['h'])-float(camera['cy'])))/float(camera['fl_y'])
 return float(ray_distance_near)/np.sqrt(1+x*x+y*y)


def project(points,camera):
 M=np.asarray(camera['transform_matrix'],float);local=(np.asarray(points)-M[:3,3])@M[:3,:3];depth=-local[:,2]
 uv=np.full((len(local),2),np.nan);valid=depth>0
 uv[valid,0]=camera['fl_x']*local[valid,0]/depth[valid]+camera['cx']
 uv[valid,1]=-camera['fl_y']*local[valid,1]/depth[valid]+camera['cy']
 return uv,depth


def points_inside(points,planes,tol=1e-7):
 return (np.asarray(points)@planes[:,:3].T+planes[:,3]>=-tol).all(1)


def cells_possible(centers,halfsize,planes,tol=1e-7):
 """Exact maxima over 8 AABB corners; NOT an exact polyhedron intersection."""
 maxima=np.asarray(centers)@planes[:,:3].T+planes[:,3]+np.abs(planes[:,:3])@np.asarray(halfsize)
 return (maxima>=-tol).all(1)


def grid_support(box,resolution,cameras,chunk=65536,ray_distance_near=True):
 box=np.asarray(box,float);res=np.asarray(resolution,int)
 if box.shape!=(2,3)or res.shape!=(3,)or np.any(res<=0)or np.any(box[1]<=box[0]):raise ValueError('Valid grid required')
 half=(box[1]-box[0])/res/2;out=np.zeros(tuple(res),bool);flat=out.reshape(-1);per_camera=[]
 for camera in cameras:
  axial=conservative_axial_near(camera)if ray_distance_near else .052
  planes=camera_planes(camera,axial);count=0
  for start in range(0,len(flat),chunk):
   stop=min(start+chunk,len(flat));ij=np.column_stack(np.unravel_index(np.arange(start,stop),tuple(res)));centers=box[0]+(ij+.5)*(2*half)
   supported=cells_possible(centers,half,planes);flat[start:stop]|=supported;count+=int(supported.sum())
  per_camera.append(dict(camera=camera['physical_camera'],possibly_intersected_cells=count,conservative_axial_near=axial))
 return out[None],per_camera


def main(args):
 from marshawn_occupancy_audit import dilate_one,write_atlas
 from marshawn_release_common import point_cells,authoritative_grid
 tf=json.loads(args.transforms.read_text());train=set(tf['train_filenames']);cameras=tf['frames'];training=[f for f in cameras if f['file_path']in train]
 assert len(training)==63 and len(cameras)==66 and len(set(f['physical_camera']for f in cameras))==66
 with np.load(args.r0_grid)as z:core0=z['core'].copy();active0=z['active'].copy();box=z['world_aabb'].copy();boxes=z['cascade_aabbs'].copy();res=z['resolution'].copy();seed=z['observed_voxel_seed'].copy()
 assert np.array_equal(active0,core0|seed)and int((active0&~core0).sum())==35
 core1,active1,ema1,boxes1,policy1=authoritative_grid(args.r1_grid,args.r1_receipt,args.r1_checkpoint)
 assert policy1['radius']==1 and policy1['checkpoint_step']==20000
 assert np.array_equal(boxes,boxes1)and np.array_equal(box.reshape(1,6),boxes)
 support63,counts63=grid_support(box,res,training);support66,counts66=grid_support(box,res,cameras)
 axial63,_=grid_support(box,res,training,ray_distance_near=False)
 assert not(support63&~support66).any()
 halo63=dilate_one(support63);halo66=dilate_one(support66)
 xyz=np.load(args.witnesses)['xyz'];obs=np.load(args.observations);by={f['physical_camera']:f for f in cameras};camera_names=obs['camera_names'];inside_obs=[];maxerror=0.;mindepth=float('inf');minedge=float('inf')
 for ci,name in enumerate(camera_names):
  ids=np.flatnonzero(obs['camera_indices']==ci);points=xyz[obs['point_indices'][ids]];cam=by[str(name)];uv,depth=project(points,cam);inside=points_inside(points,camera_planes(cam));inside_obs.extend(inside.tolist());maxerror=max(maxerror,float(np.max(np.abs(uv-obs['uv'][ids]))));mindepth=min(mindepth,float(depth.min()));minedge=min(minedge,float(np.min(np.column_stack([uv,cam['w']-uv[:,0],cam['h']-uv[:,1]]))))
 assert len(inside_obs)==44938 and all(inside_obs)and maxerror<1e-5
 _,inside,known63=point_cells(xyz,support63,boxes);_,_,known66=point_cells(xyz,support66,boxes)
 assert inside.all()and known63.all()and known66.all()
 from scipy.spatial import cKDTree
 _,sid=cKDTree(xyz).query(np.load(args.subject)['xyz']);sid=np.unique(sid);assert len(sid)==495
 seed_neighborhood=dilate_one(seed);assert not(seed_neighborhood&~halo63).any()
 masks=dict(TRAIN63_possible=support63,ALL66_possible=support66,TRAIN63_support_ONEhalo=halo63,ALL66_support_ONEhalo=halo66)
 args.output.mkdir(exist_ok=False);records={}
 for label,active in [('R0_12k_MIN35',active0),('R1_20k_STORED_HALO1',active1)]:
  records[label]={}
  for support_name,support in masks.items():
   outside=active&~support;ids=np.argwhere(outside)
   records[label][support_name]=dict(active_cells=int(active.sum()),outside_cells=int(outside.sum()),outside_fraction=float(outside.sum()/active.sum()),remaining_cells=int((active&support).sum()),outside_cell_examples=ids[:16].tolist())
 np.savez_compressed(args.output/'calibration_support_ONLY_NOT_APPLIED.npz',**masks,R0_12k_active=active0,R1_20k_stored_active=active1,core_R0=core0,core_R1=core1,world_aabb=box,cascade_aabbs=boxes,resolution=res,observed_witness_seed=seed,R0_candidate_raw=active0&support63,R0_candidate_support_halo1=active0&halo63,R1_candidate_raw=active1&support63,R1_candidate_support_halo1=active1&halo63)
 # Small 3-plane image, diagnostic support only; no full candidate atlas or culling.
 from PIL import Image,ImageDraw
 panel=Image.new('RGB',(3*384,4*408),'#202020');draw=ImageDraw.Draw(panel)
 for j,(label,support)in enumerate(masks.items()):
  for axis in range(3):
   pixels=np.take(support[0],64,axis=axis).T.astype(np.uint8)*255;image=Image.fromarray(pixels).resize((384,384),Image.Resampling.NEAREST).convert('RGB');panel.paste(image,(axis*384,j*408+24));draw.text((axis*384+3,j*408+3),label+' '+ 'XYZ'[axis]+'=64',fill='white')
 panel.save(args.output/'support_midplanes.png')
 report=dict(kind='CONSERVATIVE_CALIBRATION_FRUSTUM_SUPPORT_NO_CULLING',camera_counts=dict(TRAIN=63,ALL=66),near_ray_distance=.052,near_policy='Axial plane=.052*minimum cos(theta) across raster corners, conservatively retaining every native ray with Euclidean t>=.052. Unbounded far cone.',near_vs_standard_axial052_changed_union_cells=int((support63!=axial63).sum()),raster_convention='u in[0,W],v in[0,H]; prepared OpenGL c2w right/up/back; storedUV already halfpixel coordinate, no extra+.5',plane_test='Outside a camera ONLY if all8 voxel corners are strictly outside at leastONE same plane; equivalent center+abs(normal)*halfextent; touching/straddling cells retained; tol1e-7world',support_limit='Conservative overapproximation of possible frustum intersection; passing every plane separately does not prove exact intersection or observed surface',world_aabb=box.tolist(),resolution=res.tolist(),support_counts={k:int(v.sum())for k,v in masks.items()},per_camera63=counts63,per_camera66=counts66,grid_counts=records,known_geometry=dict(all9022_inTRAIN_support=True,BODY495_inTRAIN_support=True,actual44938_observations_inside_OWN_camera=True,max_projected_vs_cached_UV_error_pixels=maxerror,min_observed_depth=mindepth,min_observed_raster_edge_distance=minedge,all9022_witness_ONEhalo_neighborhoods_in_support_ONEhalo=True,extra_point_protection_needed=False,reprojection_consistency_only_not_new_calibration_proof=True),no_eval_RGB_or_images_read=True,no_field_or_density_queries=True,no_worker_or_mask_changes=True,no_candidate_applied=True,geometry_of_outside='UNOBSERVED by these input cones; PHYSICALLY UNKNOWN, not certified empty',input_sha256={str(p):file_sha(p)for p in [args.transforms,args.r0_grid,args.r1_grid,args.r1_receipt,args.r1_checkpoint,args.witnesses,args.subject,args.observations]},script_sha256=file_sha(__file__),R1_source_policy=policy1)
 (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(support=report['support_counts'],counts=records,UVmax=maxerror)))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['transforms','r0-grid','r1-grid','r1-receipt','r1-checkpoint','witnesses','subject','observations','output']:p.add_argument('--'+name,type=Path,required=True)
 main(p.parse_args())
