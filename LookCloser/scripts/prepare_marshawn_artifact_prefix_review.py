"""Source-bound CPU review of five actual-artifact prefixes; never creates labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from marshawn_actor_coordinates import raw_array_indices


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def main(c, out):
    out.mkdir(parents=True, exist_ok=False)
    original = Path(__file__).resolve().parents[1] / 'experiments/assets/marshawn_artifact_sources_20261107/TRAIN_prefix_proposals_NOT_LABELS.json'
    proposals = [p for p in read(original)['rows'] if p['artifact'] != 'torso2_q50']
    assert len(proposals) == 5
    raw_request = c.parent / 'two_actor_isolation/raw_native_sources/request.json'
    raw_rows = {r['camera']:r for r in read(raw_request)['records']}
    confidence_manifest = c.parent / 'two_actor_isolation/confidence_native_hair_v2/manifest.json'
    sources = {r['camera']:r for r in read(confidence_manifest)['rows']}
    transforms = c / 'data/transforms.json'; meta = read(transforms)
    frames = {r['physical_camera']:r for r in meta['frames'] if r['file_path'] in set(meta['train_filenames'])}
    aabb_file = Path(__file__).resolve().parents[1] / 'configs/aabb_review/marshawn_1660_1900_floor.json'
    # This accepted box is fixed independently of the observed artifact points.
    bounds = np.array([[-.84,-.21,-.07],[.15,.42,.63]],np.float64)
    results = []
    for proposal in proposals:
        camera = proposal['camera']; frame = frames[camera]; source = sources[camera]; raw = raw_rows[camera]
        assert source['split'] == raw['split'] == 'TRAIN'
        assert sha(source['raw_archive']) == source['raw_archive_sha256']
        assert sha(source['archive']) == source['archive_sha256']
        assert sha(raw['image']) == raw['image_sha256']
        with np.load(source['raw_archive'],allow_pickle=False) as z:
            masks={k:z[k] for k in ['nominal','black','grey','source_ineligible']}; raw_box=z['crop_xyxy']
        assert list(raw_box) == raw['crop_xyxy']
        with np.load(c/'data/semantics'/f'{camera}.npz',allow_pickle=False) as z:
            main_masks={k:z[k] for k in ['W64','black','grey','core','band','valid']}
        with np.load(source['archive'],allow_pickle=False) as z: original_w=z['W64']
        support_file=c/'native_semantic_grid_20000/occlusion_v2/W64/archives'/f'{camera}.npz'
        with np.load(support_file,allow_pickle=False) as z: support={k:z[k] for k in ['support','unknown','valid']}
        x0,y0,x1,y1=proposal['box_xyxy_exclusive']; assert (x1-x0,y1-y0)==(7,7)
        pose=np.asarray(frame['transform_matrix'],np.float64); origin=pose[:3,3]
        rays=[]; footprints=[]
        for y in range(y0,y1):
            for x in range(x0,x1):
                local=np.array([(x+.5-frame['cx'])/frame['fl_x'],-(y+.5-frame['cy'])/frame['fl_y'],-1.])
                direction=pose[:3,:3]@local; direction/=np.linalg.norm(direction)
                hit=(bounds-origin)/direction; near=float(np.max(hit.min(0))); far=float(np.min(hit.max(0)))
                assert far>near>=0 and direction[2]<0
                guard_t=float((.025-origin[2])/direction[2]); end=min(far,guard_t-1e-6)
                start_xyz=origin+near*direction; end_xyz=origin+end*direction
                assert end>near and min(start_xyz[2],end_xyz[2])>.025
                eligible=bool(main_masks['valid'][y,x] and support['valid'][y,x] and not support['support'][y,x] and not support['unknown'][y,x] and main_masks['W64'][y,x]==0)
                assert eligible, (camera,x,y,'Existing source qualification changed')
                corners=np.array([[x,y],[x+1,y],[x+1,y+1],[x,y+1]],np.float64)
                mapped=raw_array_indices(corners,source['camera_row'])
                # Dense boundary samples and one-RAW-pixel padded rectangle are
                # only a conservative mask proximity diagnostic, not optical alpha.
                q=np.linspace(0,1,9); samples=np.concatenate([corners[i]*(1-q[:,None])+corners[(i+1)%4]*q[:,None] for i in range(4)])
                sampled=raw_array_indices(samples,source['camera_row'])-raw_box[:2]
                lo=np.floor(sampled.min(0)).astype(int)-1; hi=np.ceil(sampled.max(0)).astype(int)+2
                in_crop=bool((lo>=0).all() and hi[0]<=masks['nominal'].shape[1] and hi[1]<=masks['nominal'].shape[0])
                assert in_crop
                ys,xs=slice(lo[1],hi[1]),slice(lo[0],hi[0])
                raw_counts={k:int(v[ys,xs].sum()) for k,v in masks.items()}
                raw_counts['add_only_hair']=int((masks['nominal']&~(masks['black']|masks['grey']))[ys,xs].sum())
                rays.append(dict(camera=camera,x=x,y=y,origin=origin.tolist(),direction=direction.tolist(),near=near,far=end,
                    AABB_far=far,endpoint=end_xyz.tolist(),startpoint=start_xyz.tolist(),min_segment_floor_frame_Z=float(min(start_xyz[2],end_xyz[2])),
                    Z0_plane_t=float(-origin[2]/direction[2]),empirical_guard_Z=.025,guard_backoff_ray_units=1e-6,
                    W64=float(main_masks['W64'][y,x]),original_confidence_W64=float(original_w[y,x]),
                    core=bool(main_masks['core'][y,x]),band=bool(main_masks['band'][y,x]),
                    qualified_support=False,UNKNOWN=False,source_valid=True,main_valid=True,
                    RAW_corners_arrayindices=mapped.tolist(),RAW_padded_proximity_box_local=[*lo.tolist(),*hi.tolist()],RAW_proximity_counts=raw_counts))
                footprints.append(mapped-raw_box[:2])
        center=((x0+x1)//2,(y0+y1)//2); point=np.asarray(proposal['point_xyz'])
        central=next(r for r in rays if (r['x'],r['y'])==center); direction=np.asarray(central['direction'])
        depth=float((point-origin)@direction); perp=float(np.linalg.norm(point-origin-depth*direction))
        assert central['near']<depth<central['far']
        uv=raw_array_indices(np.array(center)+.5,source['camera_row'])-raw_box[:2]
        cx,cy=uv; detail_box=[int(cx)-240,int(cy)-180,int(cx)+240,int(cy)+180]
        raw_image=Image.open(raw['image']).convert('RGB'); native=raw_image.crop(detail_box)
        overlay=np.array(raw_image); actor=masks['black']|masks['grey']; hair=masks['nominal']&~actor
        overlay[actor]=(overlay[actor]*.55+np.array([0,90,0])).astype(np.uint8)
        overlay[hair]=(overlay[hair]*.4+np.array([130,100,0])).astype(np.uint8)
        overlay[masks['source_ineligible']]=(overlay[masks['source_ineligible']]*.5+np.array([100,0,100])).astype(np.uint8)
        marked=Image.fromarray(overlay).crop(detail_box); draw=ImageDraw.Draw(marked)
        for polygon in footprints:
            p=[tuple(v-np.array(detail_box[:2])) for v in polygon]; draw.line(p+[p[0]],fill='cyan',width=1)
        native.save(out/f'{camera}_{center[0]}_{center[1]}_RAW.png')
        marked.save(out/f'{camera}_{center[0]}_{center[1]}_RAW_footprints.png')
        image_path=c/'data'/frame['file_path']; hd=Image.open(image_path).convert('RGB')
        hd_box=[center[0]-180,center[1]-100,center[0]+180,center[1]+140]
        context=hd.crop(hd_box); d=ImageDraw.Draw(context)
        d.rectangle([x0-hd_box[0],y0-hd_box[1],x1-hd_box[0],y1-hd_box[1]],outline='cyan',width=1)
        weight=(np.clip(main_masks['W64'],0,1)*255+.5).astype(np.uint8)
        confidence=Image.fromarray(weight).convert('RGB').crop(hd_box);d=ImageDraw.Draw(confidence)
        d.rectangle([x0-hd_box[0],y0-hd_box[1],x1-hd_box[0],y1-hd_box[1]],outline='cyan',width=1)
        page=Image.new('RGB',(1000,700),'#141414');d=ImageDraw.Draw(page)
        d.text((8,4),f"{camera}{center} {proposal['artifact']} | TRAIN source review ONLY; 49 individual prefixes",fill='white')
        d.text((8,22),'RAW: original frozen display / mask overlays + calibrated7x7HD footprints (cyan)',fill='white')
        page.paste(native,(8,45)); page.paste(marked,(508,45));page.paste(context,(8,425));page.paste(confidence,(508,425))
        d.text((8,405),'Frozen HD RGB context / original MAIN RGB W64 (black=zero)',fill='white')
        d.text((8,670),f"All49: W64=0 valid ~support ~UNKNOWN; min endZ={min(r['endpoint'][2] for r in rays):.6f}; guard empirical, NOT local floor bound",fill='yellow')
        page_path=out/f'{camera}_{center[0]}_{center[1]}_review.jpg';page.save(page_path,quality=95)
        results.append(dict(artifact=proposal['artifact'],camera=camera,center_pixel=list(center),box_xyxy_exclusive=[x0,y0,x1,y1],
            source_original_checkpoint_sha256='542e97f59ec1fd41452b0a04fde6bd832e0b430773d1989eef2ec062ef3f1a34',
            diagnostic_point_xyz=point.tolist(),central_ray_point_depth=depth,central_ray_point_perpendicular=perp,
            central_ray_point_before_end=central['far']-depth,all_retained_rays=49,rays=rays,
            min_endpoint_Z=min(r['endpoint'][2] for r in rays),min_segment_Z=min(r['min_segment_floor_frame_Z'] for r in rays),
            source_bindings={str(p):sha(p) for p in [Path(raw['image']),Path(source['raw_archive']),Path(source['archive']),image_path,c/'data/semantics'/f'{camera}.npz',support_file]},
            native_RAW_source_request=raw,RAW_crop_detail_box_local=detail_box,source_page=page_path.name,source_page_sha256=sha(page_path)))
    receipt=dict(kind='FIVE_ARTIFACT_INTERSECTING_TRAIN_PREFIX_MANUAL_REVIEW_NOT_LABELS',rows=results,
        inputs_sha256={str(p):sha(p) for p in [original,raw_request,confidence_manifest,transforms,aabb_file,Path(__file__).with_name('marshawn_actor_coordinates.py')]},
        helper_sha256=sha(__file__),accepted_AABB=bounds.tolist(),empirical_floor_guard_Z=.025,
        floor_guard_limitation='Global floor-fit maxLOO .0127875 plus margin is empirical; local floor/mat/rig first-surface uncertainty remains. No calibrated floor bound is claimed.',
        coordinate_contract='HD centres x+.5,y+.5; OPENCV RAW arrayindices from existing calibrated helper. Individual pixel corners mapped with same model; RAW masks are segmentation estimates, not optical alpha.',
        excluded='AA(1195,316) positive RGB-band conflict; not included',
        requested_root_review='Confirm visible nonactor source and no physical first surface before EACH retained prefix endpoint. Masks/negative Z alone do not establish this.',
        source_or_labels_modified=False,GPU_used=False,training_authorized=False,manual_qualification='PENDING_ROOT')
    (out/'proposal.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps([dict(camera=r['camera'],center=r['center_pixel'],rays=49,minZ=r['min_segment_Z'],point_before_end=r['central_ray_point_before_end'])for r in results]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.campaign,a.output)
