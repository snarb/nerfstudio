"""TRAIN-only semantic qualification of existing measured SfM rays.

Separate semantic-absence segments are an opt-in soft observation hypothesis,
never measured depth, opacity labels, visual-hull carving or grid admission.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.ndimage import distance_transform_edt
from marshawn_actor_coordinates import raw_array_indices
R=Path('/fsx/tmp/marshawn_001660_hd')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def slab(O,D,box):
 parallel=np.abs(D)<1e-12;safe=np.where(parallel,1,D);a=(box[0]-O)/safe;b=(box[1]-O)/safe;lo=np.where(parallel,-np.inf,np.minimum(a,b));hi=np.where(parallel,np.inf,np.maximum(a,b));bad=(parallel&((O<box[0])|(O>box[1]))).any(-1);return np.max(lo,axis=-1),np.min(hi,axis=-1),bad

def main(a):
 a.output.mkdir(parents=True,exist_ok=False);m=json.loads(a.mask_manifest.read_text());tf=json.loads((Path(m['data_dir'])/'transforms.json').read_text());frames={r['physical_camera']:r for r in tf['frames']};rows={r['camera']:r for r in m['rows']};z=np.load(R/'free_space_candidates/scene_loose.npz');old={k:z[k]for k in z.files};camera_names=old['camera_names'];P=old['point_idx'];C=old['camera_idx'];qualified=np.zeros(len(P),bool);pointvotes=np.zeros(9022,np.int32);selectors=[]
 for ci,cam0 in enumerate(camera_names):
  cam=str(cam0);r=rows[cam];assert r['split']=='TRAIN';assert sha(r['archive'])==r['archive_sha256'];w=np.load(r['archive']);inside=(w['hard']>=.99999)&~w['source_ineligible'];interior=distance_transform_edt(inside)>=2.;idx=np.flatnonzero(C==ci);uv=old['uv'][idx];x=np.floor(uv[:,0]).astype(int);y=np.floor(uv[:,1]).astype(int);valid=(x>=0)&(x<inside.shape[1])&(y>=0)&(y<inside.shape[0]);ok=np.zeros(len(idx),bool);ok[valid]=interior[y[valid],x[valid]];qualified[idx]=ok;np.add.at(pointvotes,P[idx[ok]],1);selectors.append({'camera':cam,'actual_track_observations':len(idx),'nominal_interior_observations':int(ok.sum())})
 keep=qualified&(pointvotes[P]>=3);boxes=json.loads(a.bounds.read_text())['boxes'];variants=[]
 for name,b in boxes.items():
  box=np.asarray(b);xyz=old['witness_xyz'];inbox=((xyz>=box[0])&(xyz<=box[1])).all(1);lo,hi,bad=slab(old['origins'],old['directions'],box);start=np.maximum(np.maximum(old['t_start'],lo),.052);end=np.minimum(old['t_end'],hi);selected=keep&inbox&~bad&(end>start);data={k:(v[selected]if len(v.shape)>0 and v.shape[0]==len(P)else v)for k,v in old.items()};data.update(t_start=start[selected],t_end=end[selected],aabb=box);p=a.output/f'actor_{name}.npz';np.savez_compressed(p,**data);variants.append({'name':'actor_'+name,'file':p.name,'sha256':sha(p),'aabb':b,'ray_count':int(selected.sum()),'point_count':len(np.unique(P[selected]))})
 oldmanifest=json.loads((R/'free_space_candidates/manifest.json').read_text());manifest={k:oldmanifest[k]for k in ['schema_version','kind','coordinate_system','t_units','calibration_sha256']};bindings=dict(oldmanifest['bindings']);bindings['transforms_sha256']=m['transforms_sha256'];manifest.update(approved_for_training=False,uses_masks=True,uses_eval_rgb=False,bindings=bindings,policy={'near':.052,'minimum_surface_margin':.01,'relative_surface_margin':.05,'same_physical_policy_all_boxes':True},variants=variants,bounds_source=str(a.bounds),bounds_sha256=sha(a.bounds),actor_semantic_selection={'manifest':str(a.mask_manifest),'sha256':sha(a.mask_manifest),'train63_only':True,'minimum_actual_track_camera_votes':3,'foreground_selection':'hardnominalconfidence>=.99999, >=2HDpx interior, source-ineligible false; same physical point qualified in >=3 original TRAIN track cameras','uv_provenance':'original calibrated reprojection at actual observed track-camera graph, NOT newly detected UV and NOT independent residual check','soft_band_is_NOT_geometry_truth':True,'original_observation_count':len(P),'selected_observation_count_before_box':int(keep.sum()),'selected_point_count_before_box':int((pointvotes>=3).sum()),'selectors':selectors},no_fullscene_MVS_or_assumed_floor=True)
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');np.savez_compressed(a.output/'point_selection.npz',point_idx=np.arange(9022),votes=pointvotes,qualified_observation=qualified,selected_observation=keep);print(json.dumps({'variants':variants,'selected_points':int((pointvotes>=3).sum())}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bounds',type=Path,required=True);p.add_argument('--mask-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
