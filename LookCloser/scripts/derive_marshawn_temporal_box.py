"""Floor-frame common AABB from explicitly reviewed native actor instances."""
import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from aabb_visualization import camera_in_box_frame, box_edges
from mask_visual_hull_aabb import MaskCamera, derive_aabb, temporal_envelope, ray_box_hits
from prepare_marshawn_color_variants import sha
from actor_box_geometry import triangulate_actor_tracks, anchor_components
from PIL import Image

ROOT=Path('/fsx/tmp/marshawn_001660_hd')


def camera(row,fg,unknown,valid,frame,native_scale=(1.,1.)):
    c=camera_in_box_frame(row,frame)
    return MaskCamera(row['physical_camera'],np.array(c['transform_matrix']),
        c['fl_x'],c['fl_y'],c['cx'],c['cy'],fg,unknown,valid,native_scale)


def main(a):
    cv2.setNumThreads(2)
    O=a.output;O.mkdir(parents=True,exist_ok=False)
    identities=json.loads((a.input/'reviewed_identities.json').read_text())
    frame_meta=json.loads((a.input/'floor_frame.json').read_text())
    T=np.asarray(frame_meta['world_to_box'])
    source1660=ROOT/'two_actor_isolation/automatic_aabb/input_v2/manifest.json'
    manifest=json.loads(source1660.read_text())
    corners,_=box_edges(manifest['search_domain'])
    points=corners@T[:3,:3].T+T[:3,3]
    # Same full-scene search envelope across time, never clipped to an actor guess.
    domain=np.stack([points.min(0),points.max(0)])
    results=[]
    for f in [1660,1720,1780,1840,1900]:
        folder=O/str(f);folder.mkdir();cameras=[];bindings=[];source_rows=[];source_images=[]
        actor_cameras={'fighter_black':[],'fighter_grey':[]}
        if f!=1660 and a.reuse_completed and (a.reuse_completed/str(f)/'hull.json').exists():
            prior=a.reuse_completed/str(f);record=json.loads((prior/'hull.json').read_text())
            if not record.get('carved_per_identity') or not np.allclose(record['world_to_box'],T) or record['padding_native_pixels']!=a.padding or record['leaf_size_upper_bound']!=a.leaf_size or record['min_foreground_support']!=a.min_support:
                raise ValueError('Reused temporal hull settings differ')
            expected_bg=a.last_frame_background_allowance if f==1900 else 1
            if record['max_clear_background_contradictions']!=expected_bg or {r['camera'] for r in record['bindings']}!=set(identities[str(f)]):
                raise ValueError('Reused view set or background policy changed')
            for binding in record['bindings']:
                cam=binding['camera']
                if binding['groups']!=identities[str(f)][cam] or binding['candidates_sha256']!=sha(a.input/'candidates'/str(f)/cam/'candidates.npz'):
                    raise ValueError('Reused temporal masks or identities changed')
            if record['requires_expanded_search']:raise ValueError('Cannot reuse an unbounded hull')
            shutil.copytree(prior,folder,dirs_exist_ok=True);results.append(record);continue
        if f==1660 and a.reuse_first:
            record=json.loads((a.reuse_first/'hull.json').read_text())
            if not np.allclose(record['world_to_box'],T) or record['padding_native_pixels']!=a.padding or record['leaf_size_upper_bound']!=a.leaf_size:
                raise ValueError('Cannot reuse a differently configured first-frame hull')
            for name in ['hull.json','cells.npz']:shutil.copy2(a.reuse_first/name,folder/name)
            results.append(record);continue
        if f==1660:
            for row in manifest['cameras']:
                p=source1660.parent/row['masks']
                assert sha(p)==row['masks_sha256']
                with np.load(p) as z:
                    cameras.append(camera(row['camera_row'],z['foreground'],z['unknown'],z['valid'],T,
                        tuple(row['native_scale_yx'])))
                bindings.append(dict(camera=row['camera'],mask_sha256=sha(p)))
        else:
            for cam,groups in identities[str(f)].items():
                p=a.input/'candidates'/str(f)/cam/'candidates.npz'
                receipt=json.loads(p.with_name('result.json').read_text())
                assert sha(p)==receipt['candidates_sha256']
                row=json.loads((a.input/'sources'/str(f)/cam/'receipt.json').read_text())
                with np.load(p) as z:
                    masks=z['masks'];selected=sorted(set(sum(groups.values(),[])))
                    others=[j for j in range(len(masks)) if j not in selected]
                    foreground=masks[selected].any(0)
                    unknown=masks[others].any(0) if others else np.zeros_like(foreground)
                # Return to the FULL calibrated image, not a crop camera. The
                # manually reviewed search ROI is not the camera's field of view.
                # Treating its exterior as out-of-FOV creates far-away ghosts.
                c=row['camera_row'];x0,y0,x1,y1=row['native_crop_xyxy']
                yy,xx=np.where(foreground)
                touch=[bool(x0>0 and xx.min()<32),bool(y0>0 and yy.min()<32),
                       bool(x1<8192 and xx.max()>=foreground.shape[1]-32),
                       bool(y1<4320 and yy.max()>=foreground.shape[0]-32)]
                if any(touch):raise ValueError(f'Expand native actor search ROI: {f} {cam} {touch}')
                full=np.zeros((4320,8192),np.float32)
                full[y0:y1,x0:x1]=foreground
                fg=cv2.resize(full,(c['w'],c['h']),interpolation=cv2.INTER_AREA)>0
                full.fill(0);full[y0:y1,x0:x1]=unknown
                un=cv2.resize(full,(c['w'],c['h']),interpolation=cv2.INTER_AREA)>0
                valid=np.asarray(Image.open(ROOT/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png'/c['mask_path']))>0
                individual={}
                for identity in actor_cameras:
                    full.fill(0);full[y0:y1,x0:x1]=masks[groups[identity]].any(0)
                    individual[identity]=cv2.resize(full,(c['w'],c['h']),interpolation=cv2.INTER_AREA)>0
                for identity in actor_cameras:
                    other=next(k for k in actor_cameras if k!=identity)
                    actor_cameras[identity].append(camera(c,individual[identity],un|individual[other],valid,T,
                        (4320/c['h'],8192/c['w'])))
                cameras.append(camera(c,fg,un,valid,T,(4320/c['h'],8192/c['w'])))
                source_rows.append(c);source_images.append(Image.open(row['image']).convert('RGB'))
                np.savez_compressed(folder/(cam+'_mask.npz'),foreground=fg,unknown=un)
                bindings.append(dict(camera=cam,candidates_sha256=sha(p),groups=groups,
                    camera_row=c,native_camera=row['native_camera'],native_crop=row['native_crop_xyxy']))
        if f==1660:
            record,cells=derive_aabb(cameras,domain,leaf_size=a.leaf_size,padding_native=a.padding,
                min_support=3,max_background=1,margin_world=0.,snap_step=.005,max_cells=10000000)
            np.savez_compressed(folder/'cells.npz',**cells)
        if f!=1660:
            all_cells=[];all_xyz=[];reports=[]
            for identity,views in actor_cameras.items():
                per=folder/identity;per.mkdir()
                record,cells=derive_aabb(views,domain,leaf_size=a.leaf_size,padding_native=a.padding,
                    min_support=a.min_support,max_background=a.last_frame_background_allowance if f==1900 else 1,
                    margin_world=0.,snap_step=.005,max_cells=10000000)
                np.savez_compressed(per/'all_unqualified_cells.npz',**cells)
                xyz,observations=triangulate_actor_tracks(source_rows,source_images,[c.foreground for c in views])
                anchors=xyz@T[:3,:3].T+T[:3,3]
                (per/'anchor_observations.json').write_text(json.dumps(observations,indent=2)+'\n')
                keep,components=anchor_components(cells['lower'],cells['upper'],anchors)
                (per/'component_qualification.json').write_text(json.dumps(components,indent=2)+'\n')
                cells={k:v[keep] for k,v in cells.items()}
                all_cells.append(cells);all_xyz.append(xyz)
                reports.append(dict(identity=identity,anchors=len(anchors),components=components,
                    retained_cells=len(cells['lower'])))
            cells={k:np.concatenate([v[k] for v in all_cells]) for k in all_cells[0]}
            xyz=np.concatenate(all_xyz);anchors=xyz@T[:3,:3].T+T[:3,3]
            np.savez_compressed(folder/'actor_anchors.npz',xyz=xyz,box_xyz=anchors)
            np.savez_compressed(folder/'cells.npz',**cells)
            raw=np.stack([np.minimum(cells['lower'].min(0),anchors.min(0)),
                          np.maximum(cells['upper'].max(0),anchors.max(0))])
            record.update(retained_extent_aabb=raw.tolist(),retained_cells=len(cells['lower']),
                requires_expanded_search=bool(((raw[0]<=domain[0]+1e-9)|(raw[1]>=domain[1]-1e-9)).any()),
                carved_per_identity=True,identities=reports,independent_actor_anchors=len(anchors))
            print('qualified',f,len(anchors),raw.tolist(),flush=True)
        record.update(frame=f,world_to_box=T.tolist(),bindings=bindings)
        (folder/'hull.json').write_text(json.dumps(record,indent=2)+'\n')
        if record['requires_expanded_search']:raise ValueError('Full-scene hull remains unbounded; inspect saved cells')
        results.append(record)
    common=temporal_envelope([r['retained_extent_aabb'] for r in results],minimum_margin=a.margin,snap_step=.01)
    out=dict(aabb=common.tolist(),world_to_box=T.tolist(),box_to_world=np.linalg.inv(T).tolist(),
        coordinate_system='AABB in rigid floor-aligned frame; oriented box in original prepared world',
        original_accepted_box='accepted_1660_original_axes.json',
        record_fps=120,output_fps=24,sample_frames=[1660,1720,1780,1840,1900],
        sample_times_seconds=[0,.5,1,1.5,2],output_source_frames=list(range(1660,1900,5)),
        output_frame_count=48,end_boundary_check=1900,
        minimum_face_margin=a.margin,mask_padding_native_pixels=a.padding,
        floor_fit=frame_meta,identity_sha256=sha(a.input/'reviewed_identities.json'),
        per_frame_extents={str(r['frame']):r['retained_extent_aabb'] for r in results},
        actual_lower_margin=(np.min([r['retained_extent_aabb'][0] for r in results],axis=0)-common[0]).tolist(),
        actual_upper_margin=(common[1]-np.max([r['retained_extent_aabb'][1] for r in results],axis=0)).tolist(),
        mat_exclusion_feasible=False,reason='Grey actor lands on thick mat at 1.5s and remains there at 2s',
        temporal_scope='Five sampled poses plus explicit margin; unsampled-time containment is not mathematically certified',
        no_training=True,calibration_changed=False,optimizer_updates=0,
        script_sha256=sha(__file__),hull_script_sha256=sha(Path(__file__).with_name('mask_visual_hull_aabb.py')))
    (O/'common_box.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--leaf-size',type=float,default=.008)
    p.add_argument('--padding',type=float,default=64)
    p.add_argument('--margin',type=float,default=.025)
    p.add_argument('--min-support',type=int,default=3)
    p.add_argument('--reuse-first',type=Path)
    p.add_argument('--reuse-completed',type=Path)
    p.add_argument('--last-frame-background-allowance',type=int,default=0,
                   help='Clear-background contradictions allowed in the densely reviewed final pose')
    main(p.parse_args())
