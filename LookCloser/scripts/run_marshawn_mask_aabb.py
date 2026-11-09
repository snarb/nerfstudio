"""Bounded CPU64/128/256 mask-hull experiment and all-source ray coverage audit."""
from pathlib import Path
import argparse,hashlib,json,sys,time,shutil
import numpy as np
import cv2
from PIL import Image
from mask_visual_hull_aabb import load_npz_cameras,derive_aabb,ray_box_hits

R=Path('/fsx/tmp/marshawn_001660_hd')
sys.path[:0]=[str(R/'code_native'),str(Path(__file__).parent)]
from marshawn_actor_coordinates import raw_camera

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def coverage(input_manifest,boxes,output):
    source=json.loads(Path(input_manifest).read_text());m=json.loads(Path(source['source_manifest']).read_text())
    rows=[];manual=json.loads((R/'two_actor_isolation/revised_actor_bounds/source_projection_v1/proposals.json').read_text())['manual_boot_scope_observations']
    camera_rows={r['camera']:r['camera_row']for r in source['cameras']};boot=[]
    for record in manual:
        c=camera_rows[record['camera']];pose=np.array(c['transform_matrix']);x,y=record['pixel_array_xy']
        d=np.array([(x+.5-c['cx'])/c['fl_x'],-(y+.5-c['cy'])/c['fl_y'],-1.])@pose[:3,:3].T
        boot.append(dict(record,box_hits={name:bool(ray_box_hits(pose[:3,3],d,box))for name,box in boxes.items()},
            note='Manually selected source boot-scope pixel, NOT depth GT/correspondence'))
    for row in m['rows']:
        cam=row['camera'];c=row['camera_row'];pose=np.asarray(c['transform_matrix'],np.float64);K,dist=raw_camera(cam)
        if sha(row['raw_archive'])!=row['raw_archive_sha256']:raise ValueError('Native mask changed')
        with np.load(row['raw_archive'],allow_pickle=False)as z:
            nominal=z['nominal'].copy();core=z['black']|z['grey'];veto=z['source_ineligible'];crop=z['crop_xyxy'].copy()
        target=nominal&~veto;yy,xx=np.where(target);counts={name:{cat:dict(pixels=0,misses=0)for cat in ['nominal','coarse_core','accepted_hair_additions']}for name in boxes}
        examples={name:[]for name in boxes}
        for start in range(0,len(xx),200000):
            y,x=yy[start:start+200000],xx[start:start+200000]
            rawuv=np.stack([x+crop[0],y+crop[1]],-1).astype(np.float64)
            tangent=cv2.undistortPoints(rawuv[:,None,:],K,dist).reshape(-1,2)
            direction=np.column_stack([tangent[:,0],-tangent[:,1],-np.ones(len(x))])@pose[:3,:3].T
            classes={'nominal':np.ones(len(x),bool),'coarse_core':core[y,x],
                'accepted_hair_additions':~core[y,x]}
            for name,box in boxes.items():
                hit=ray_box_hits(pose[:3,3],direction,box)
                for cat,selection in classes.items():
                    counts[name][cat]['pixels']+=int(selection.sum());counts[name][cat]['misses']+=int((selection&~hit).sum())
                if len(examples[name])<16:
                    failed=np.where(~hit)[0][:16-len(examples[name])]
                    examples[name]+=[dict(raw_array_xy=rawuv[i].tolist(),HD_array_xy=[float(c['fl_x']*tangent[i,0]+c['cx']-.5),
                        float(c['fl_y']*tangent[i,1]+c['cy']-.5)],accepted_hair_addition=bool(~core[y[i],x[i]]))for i in failed]
        # Independently inspect all positive calibratedHD footprint pixels too.
        with np.load(row['archive'],allow_pickle=False)as z:hy,hx=np.where(z['nominal_coverage']>0)
        d=np.column_stack([(hx+.5-c['cx'])/c['fl_x'],-(hy+.5-c['cy'])/c['fl_y'],-np.ones(len(hx))])@pose[:3,:3].T
        hd={name:dict(pixels=len(hx),misses=int((~ray_box_hits(pose[:3,3],d,b)).sum()))for name,b in boxes.items()}
        rows.append(dict(camera=cam,native_raw_selected_counts=counts,HD_foreground_footprint=hd,miss_examples=examples,
            native_raw_mask_sha256=row['raw_archive_sha256'],camera_row=c))
        (output/'coverage_progress.json').write_text(json.dumps(dict(rows=rows,manual_boot_rays=boot),indent=2)+'\n')
        print(json.dumps(dict(camera=cam,native_nominal_misses={k:v['nominal']['misses']for k,v in counts.items()})),flush=True)
    totals={name:{cat:dict(pixels=sum(r['native_raw_selected_counts'][name][cat]['pixels']for r in rows),
        misses=sum(r['native_raw_selected_counts'][name][cat]['misses']for r in rows))for cat in ['nominal','coarse_core','accepted_hair_additions']}for name in boxes}
    result=dict(kind='ALL66_SELECTED_NATIVE_CORE_HAIR_RAY_BOX_COVERAGE_NECESSARY_NOT_DEPTH_CERTIFICATE',
        source_manifest_sha256=sha(source['source_manifest']),input_manifest_sha256=sha(input_manifest),rows=rows,totals=totals,
        manual_boot_rays=boot,boot_misses={name:sum(not b['box_hits'][name]for b in boot)for name in boxes},
        source_scope='RAWselectednominalexcludingverifiedoccluderveto; stablepersonIDs; nativeundistortion exactrig; noevalRGBfit',
        limits='A ray crossing the box does not prove the real actor surface lies inside it. Source hair/feet masks remain imperfect; manual full-source and unclipped-teacher review required.')
    (output/'coverage.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main(a):
    cv2.setNumThreads(2);a.output.mkdir(parents=True,exist_ok=False);code=a.output/'code';code.mkdir()
    for name in ['run_marshawn_mask_aabb.py','mask_visual_hull_aabb.py','marshawn_actor_coordinates.py']:
        shutil.copyfile(Path(__file__).parent/name,code/name)
    payload,cameras=load_npz_cameras(a.input);anchor=None
    if a.anchors:
        with np.load(a.anchors,allow_pickle=False)as z:
            xyz=z['witness_xyz'];ids=z['point_idx'];_,first=np.unique(ids,return_index=True);anchor=xyz[first].copy()
    source_bindings=dict(input_manifest=str(a.input),input_sha256=sha(a.input),code_sha256={p.name:sha(p)for p in code.glob('*.py')},
        anchors=str(a.anchors)if a.anchors else None,anchors_sha256=sha(a.anchors)if a.anchors else None,
        source_normalization_sha256=payload['normalization_sha256'],NO_optimizer_or_NN=True)
    (a.output/'source_bindings.json').write_text(json.dumps(source_bindings,indent=2)+'\n');boxes={};receipts={}
    for pad in [64,128,256]:
        folder=a.output/('padding%d'%pad);folder.mkdir()
        record,cells=derive_aabb(cameras,payload['search_domain'],padding_native=pad,leaf_size=a.leaf_size,
            min_support=3,max_background=1,margin_world=a.margin_world,axis_margin=a.axis_margin,bottom_margin=a.bottom_margin,
            snap_step=a.snap_step,anchors=anchor,batch_size=16384,max_cells=12000000)
        record.update(source_bindings=source_bindings,settings_frozen_before_source_results=True)
        (folder/'aabb.json').write_text(json.dumps(record,indent=2)+'\n');np.savez_compressed(folder/'retained_cells.npz',**cells)
        if record['requires_expanded_search']:raise ValueError('Actor-supported cells touch searchboundary; expand before selecting')
        name='padding%d'%pad;boxes[name]=record['aabb'];receipts[name]=record
        print(json.dumps(dict(completed=name,aabb=record['aabb'],cells=len(cells['lower']))),flush=True)
    proof=coverage(a.input,boxes,a.output)
    eligible=[name for name in boxes if proof['totals'][name]['nominal']['misses']==0 and proof['boot_misses'][name]==0]
    summary=dict(boxes=boxes,coverage_totals=proof['totals'],manual_boot_misses=proof['boot_misses'],
        smallest_zero_known_ray_miss_candidate=eligible[0]if eligible else None,selection_approval=False,
        selection_scope='Knownsource raycoverage only; root fullsource/unclippedteacher review still mandatory',
        source_bindings=source_bindings,perframe_AABB=True,temporal_window_envelope=None,
        next_temporal_scope='Union independentlycomputed1660/61/62 boxes onthissamefixedworldgrid; notyetmeasured')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--anchors',type=Path);p.add_argument('--leaf-size',type=float,default=.012)
    p.add_argument('--margin-world',type=float,default=.02);p.add_argument('--axis-margin',type=float,nargs=3)
    p.add_argument('--bottom-margin',type=float);p.add_argument('--snap-step',type=float,default=.01)
    a=p.parse_args();main(a)
