"""CPU native-mask hull for independently reviewed final half-second samples."""
import json,argparse,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from aabb_visualization import camera_in_box_frame,box_edges
from mask_visual_hull_aabb import MaskCamera,derive_aabb,temporal_envelope
from actor_box_geometry import triangulate_actor_tracks,anchor_components
from prepare_marshawn_tuning import sha,write
from marshawn_video_prep_mask_components import qualify

def main(root):
 cv2.setNumThreads(2);tail=root/'tail_native';out=tail/'geometry';out.mkdir(exist_ok=True)
 current=json.loads((root/'frames/001660/data/box.json').read_text());T=np.asarray(current['world_to_box'])
 seed=json.loads(Path('/fsx/tmp/marshawn_001660_hd/two_actor_isolation/automatic_aabb/input_v2/manifest.json').read_text());corners,_=box_edges(seed['search_domain']);p=corners@T[:3,:3].T+T[:3,3];domain=np.stack([p.min(0),p.max(0)])
 identities={1960:{'HC':{'black':[11],'grey':[15]},'DC':{'black':[6],'grey':[4]},'KA':{'black':[6],'grey':[7]}},2020:{'HC':{'black':[13],'grey':[15]},'DC':{'black':[6],'grey':[1]},'KA':{'black':[6],'grey':[7]}}}
 extra=json.loads((tail/'extra_identity_suggestions.json').read_text())
 for frame,rows in extra.items():
  for name,groups in rows.items():identities[int(frame)][name]={k.replace('fighter_',''):v for k,v in groups.items()}
 write(tail/'reviewed_identities.json',dict(identities=identities,manually_inspected=True,note='Baldblackleatherjacketman assisting greyfighter is crew and excluded; fragmentedcrewupperbody UNKNOWN; onlystandingbraidedblackfighter andlyinggreyfighter selected'))
 extents=[np.asarray(current['aabb'])];reports=[]
 for frame,ids in identities.items():
  folder=out/str(frame);folder.mkdir(exist_ok=True);views={a:[] for a in ['black','grey']};rows=[];images=[]
  for name,groups in ids.items():
   r=json.loads((tail/'sources'/str(frame)/name/'receipt.json').read_text());row=r['camera_row'];rows.append(row);images.append(Image.open(r['image']).convert('RGB'));allmasks=np.load(tail/'candidates'/str(frame)/name/'candidates.npz')['masks'];selected=set(groups['black']+groups['grey']);others=[k for k in range(len(allmasks)) if k not in selected]
   unknown=cv2.resize(allmasks[others].any(0).astype('uint8'),(row['w'],row['h']),interpolation=cv2.INTER_AREA)>0
   selected_native={};qualification={}
   for a,g in groups.items():
    selected_native[a],discarded,qualification[a]=qualify(allmasks[g].any(0));unknown|=cv2.resize(discarded.astype('uint8'),(row['w'],row['h']),interpolation=cv2.INTER_AREA)>0
   np.savez_compressed(folder/(name+'_selected_native.npz'),**selected_native)
   write(folder/(name+'_component_qualification.json'),qualification)
   masks={a:cv2.resize(m.astype('float32'),(row['w'],row['h']),interpolation=cv2.INTER_AREA)>0 for a,m in selected_native.items()};valid=np.ones_like(unknown);c=camera_in_box_frame(row,T)
   for a in views:
    other='grey' if a=='black' else 'black';views[a].append(MaskCamera(name,np.asarray(c['transform_matrix']),c['fl_x'],c['fl_y'],c['cx'],c['cy'],masks[a],unknown|masks[other],valid,(4320/row['h'],8192/row['w'])))
  allcells=[];anchorsall=[];actors=[]
  for actor,vc in views.items():
   report,cells=derive_aabb(vc,domain,leaf_size=.008,min_support=3,max_background=0,padding_native=64,margin_world=0,snap_step=.005,max_cells=10000000)
   floor_keep=cells['upper'][:,2]>=-.02;cells={k:v[floor_keep] for k,v in cells.items()}
   report['floor_halfspace_lower_bound']=-.02;report['floor_fit_max_residual']=.004636666777014908
   xyz,observations=triangulate_actor_tracks(rows,images,[c.foreground for c in vc]);anchors=xyz@T[:3,:3].T+T[:3,3]
   qualification={'status':'NO_ANCHORS_KEEP_ALL_CONSERVATIVE_COMPONENTS'}
   if len(anchors):
    keep,qualification=anchor_components(cells['lower'],cells['upper'],anchors);cells={k:v[keep] for k,v in cells.items()}
   np.savez_compressed(folder/(actor+'_cells.npz'),**cells);np.savez_compressed(folder/(actor+'_anchors.npz'),xyz=xyz,box_xyz=anchors)
   write(folder/(actor+'_observations.json'),observations);allcells.append(cells);anchorsall.append(anchors);actors.append(dict(actor=actor,anchors=len(anchors),hull=report,component_qualification=qualification))
  lo=np.concatenate([c['lower'] for c in allcells]);hi=np.concatenate([c['upper'] for c in allcells]);anchors=np.concatenate(anchorsall);extent=np.stack([lo.min(0),hi.max(0)])
  if len(anchors):extent=np.stack([np.minimum(extent[0],anchors.min(0)),np.maximum(extent[1],anchors.max(0))])
  extent[0,2]=max(extent[0,2],-.02)
  extents.append(extent);np.savez_compressed(folder/'cells.npz',lower=lo,upper=hi);reports.append(dict(frame=frame,extent=extent.tolist(),actors=actors));write(folder/'receipt.json',reports[-1]);print(frame,extent.tolist(),flush=True)
 common=temporal_envelope(extents,minimum_margin=.025,snap_step=.01)
 # Existing approved box already has margin: avoid adding another margin to unchanged faces.
 original=np.asarray(current['aabb']);common[0]=np.minimum(original[0],np.min(extents[1:],axis=0)[0]-.025);common[1]=np.maximum(original[1],np.max(extents[1:],axis=0)[1]+.025);common[0]=np.floor(common[0]/.01)*.01;common[1]=np.ceil(common[1]/.01)*.01
 result=dict(current,aabb=common.tolist(),sample_frames=[1660,1720,1780,1840,1900,1960,2020],sample_times_seconds=[0,.5,1,1.5,2,2.5,3],third_second_reports=reports,minimum_tail_margin=.025,mat_exclusion_feasible=False,mat_reason='Greyfighter lies onmat; matintersection unavoidable',status='PROPOSED_FOR_ROOT_VISUAL_REVIEW',no_model_or_learned_grid_loaded=True,script_sha256=sha(__file__))
 write(out/'common_box.json',result);print(common.tolist(),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
