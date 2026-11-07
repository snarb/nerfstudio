"""Source review of label-independent RGB ignore within unchanged qualified bag interiors."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from marshawn_actor_coordinates import raw_array_indices,raw_mask_coverage_to_HD
from analyze_marshawn_soft_rgb_profiles import components,counts


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(c,out,profiles,levels_path,controls):
    out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(2);bag=c/'native_semantic_grid_20000/bag_qualification';proof=bag/'train_bag_qualification.json';meta=json.loads(proof.read_text());assert not meta['uses_eval_RGB']and not meta['uses_eval_metrics'];assert sha(bag/'build_train_bag_evidence.py')==meta['script_sha256']
    sources=json.loads((c.parent/'two_actor_isolation/confidence_native_hair_v2/manifest.json').read_text());source={r['camera']:r for r in sources['rows']};analysis=json.loads((profiles/'analysis.json').read_text());levels=np.load(levels_path);control=json.loads(controls.read_text());assert control['split']=='train'and len(control['pixels'])==109
    means={name:np.array(v['proposal_components']['bag6522'])for name,v in analysis['cameras'].items()};rows=[];pages=[];total_extra=0
    special={'NA':[('q50',1037,641),('peak_shell',1035,637)],'MB':[('peak_shell',1046,652)]}
    for record in meta['records']:
        name=record['camera'];assert sha(record['raw_source'])==record['raw_source_sha256'];e=np.load(record['exclusion_HD']);shell=np.load(record['unknown_HD']);assert sha(record['exclusion_HD'])==record['exclusion_sha256']and sha(record['unknown_HD'])==record['unknown_sha256'];assert not(e&shell).any()
        raster=np.zeros((4320,8192),np.uint8);cv2.fillPoly(raster,[np.array(record['polygon_full_raw'],np.int32)],1);raster=cv2.erode(raster,np.ones((5,5),np.uint8));rebuilt=raw_mask_coverage_to_HD(raster,source[name]['camera_row'])>=.99
        # The existing hash-qualified mask is authoritative. Record threshold-edge
        # replay differences in the current environment; never regenerate that mask.
        replay_diff=rebuilt^e
        with np.load(c/'data/semantics'/f'{name}.npz')as z:sem={k:z[k]for k in ['W64','core','band','valid','nominal','black','grey']}
        existing=e&sem['grey'];extra=e&~sem['grey']&(sem['W64']>0);allpositive=e&(sem['W64']>0);assert not(e&sem['black']).any()and int(e.sum())==record['excluded_pixels'];assert (sem['valid'][e]).all();total_extra+=int(extra.sum())
        original=sem['W64'];new=original.copy();new[e]=0;means[name]=components(new,sem,levels[name])
        intersections=[]
        for point in control['pixels']:
            camera=point.get('camera_name',control['camera_order'][point['camera']])
            if camera!=name:continue
            x,y=point['x'],point['y']
            if e[y,x]or shell[y,x]:intersections.append(dict(query=point,interior=bool(e[y,x]),shell=bool(shell[y,x])))
        with np.load(profiles/'profiles'/f'{name}.npz')as z:inside=z['inside_RAW_union'];dist=z['distance_HD']
        detail=[]
        for label,x,y in special.get(name,[]):
            uv=raw_array_indices(np.array([x+.5,y+.5]),source[name]['camera_row']);local=uv-np.array(record['raw_source_origin_full'])-np.array(record['native_evidence_crop_xyxy'][:2]);detail.append(dict(label=label,pixel=[x,y],raw_array_index=uv.tolist(),raw_evidence_local=local.tolist(),qualified_interior=bool(e[y,x]),protected_shell=bool(shell[y,x]),main_grey=bool(sem['grey'][y,x]),original_W64=float(original[y,x]),existing6522_W64=0. if existing[y,x]else float(original[y,x]),proposed6874_W64=0. if e[y,x]else float(original[y,x]),distance_HD=float(dist[y,x])))
        yy,xx=np.where(e);hd_box=[int(xx.min())-24,int(yy.min())-35,int(xx.max())+35,int(yy.max())+25];rgb=np.array(Image.open(c/'data/images'/f'train_{name}.png').convert('RGB'));overlay=rgb.copy();overlay[shell]=(overlay[shell]*.5+[0,110,110]).astype(np.uint8);overlay[existing]=(overlay[existing]*.4+[0,140,0]).astype(np.uint8);overlay[extra]=(overlay[extra]*.3+[180,0,130]).astype(np.uint8)
        rawim=Image.open(record['raw_source']).convert('RGB').crop(record['native_evidence_crop_xyxy']);marked=rawim.copy();d=ImageDraw.Draw(marked);poly=[tuple(p)for p in record['polygon_in_evidence_crop']];d.line(poly+[poly[0]],fill='green',width=2)
        for point in detail:
            x,y=point['raw_evidence_local'];d.ellipse((x-8,y-8,x+8,y+8),outline='red',width=2);d.text((x+10,y),point['label'],fill='red')
        page=Image.new('RGB',(900,640),'#111111');d=ImageDraw.Draw(page);d.text((5,4),f'{name} qualified bag INTERIOR; old6522 green, extra positive magenta, protected shell cyan',fill='white');d.text((5,20),'Top: original RAW / polygon + diagnostic markers. Bottom: current HD / confidence-membership overlay.',fill='white');rawim.thumbnail((430,365));marked.thumbnail((430,365));page.paste(rawim,(5,42));page.paste(marked,(455,42));im=Image.fromarray(rgb).crop(hd_box);ov=Image.fromarray(overlay).crop(hd_box);im.thumbnail((430,205));ov.thumbnail((430,205));page.paste(im,(5,425));page.paste(ov,(455,425));path=out/f'{name}_interior_review.jpg';page.save(path,quality=95);pages.append(path)
        rows.append(dict(camera=name,original_raw_source_sha256=record['raw_source_sha256'],original_semantic_sha256=sha(c/'data/semantics'/f'{name}.npz'),exclusion_sha256=record['exclusion_sha256'],shell_sha256=record['unknown_sha256'],reconstructed_RAW_erode2_area99_mismatch=int(replay_diff.sum()),replay_difference_extra_overlap=int((replay_diff&extra).sum()),OpenCV_version=cv2.__version__,authoritative_existing_mask_used=True,qualified_interior_pixels=int(e.sum()),old_grey_only_pixels=int(existing.sum()),extra_positive_pixels=int(extra.sum()),extra_weight_sum=float(original[extra].sum(dtype=np.float64)),extra_weight_minmax=[float(original[extra].min()),float(original[extra].max())]if extra.any()else None,extra_overlaps={k:int((extra&sem[k]).sum())for k in ['black','grey','nominal','core','band']},extra_unknown_shell_overlap=int((extra&shell).sum()),extra_inside_original_RAW_levelset=int((extra&inside).sum()),known109_controls_intersections=intersections,diagnostic_pixels=detail,source_page=path.name,source_page_sha256=sha(path)))
    assert total_extra==352 and sum(r['qualified_interior_pixels']for r in rows)==6874
    phases=[]
    for step in [0,4096,6144,8192,111316]:
        norm=float(np.mean(list(means.values()),axis=0)@(counts(4096,step)/4096));old=next(r for r in analysis['phases']if r['step']==step)['normalizer']['bag6522'];phases.append(dict(step=step,old6522_normalizer=old,proposed6874_normalizer=norm,additional_expected_weight_removed=old-norm))
    result=dict(kind='PROPOSED_RGB_IGNORE_ALL_EXISTING_QUALIFIED_BAG_INTERIOR_NOT_APPLIED',qualification_sha256=sha(proof),manual_review_receipt_sha256=sha(bag/'manual_review_receipt.json'),original_policy=meta['polygon_policy'],uncertainty_policy=meta['uncertainty'],existing_pixels=6522,proposed_pixels=6874,extra_positive_pixels=352,extra_weight_sum=sum(r['extra_weight_sum']for r in rows),rows=rows,normalizers=phases,controls109_sha256=sha(controls),polygon_or_shell_expanded=False,source_weights_modified=False,admission_or_alpha_targets_changed=False,qualification_class_independent=True,conclusion='Existing qualification labels visible nonactor INTERIOR independently of main grey class. Removing the grey filter covers352 existing positive RGB pixels without changing polygons or protected shells. No physical-empty/whole-ray-negative implication.')
    (out/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--profiles',type=Path,required=True);ap.add_argument('--levels',type=Path,required=True);ap.add_argument('--controls',type=Path,required=True);a=ap.parse_args();main(a.campaign,a.output,a.profiles,a.levels,a.controls)
