"""Diagnostic-only existing RAW semantic coverage at six EVAL boundary pixels.

No segmentation, matte labels, model, GPU, or evaluation-domain modification.
"""
import argparse,hashlib,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from marshawn_actor_coordinates import raw_array_indices,raw_mask_coverage_to_HD,raw_camera,RIG


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(c,out):
    cv2.setNumThreads(2);out.mkdir(parents=True,exist_ok=False)
    iso=c.parent/'two_actor_isolation';base=iso/'confidence_native_hair_v2';manifest=json.loads((base/'manifest.json').read_text());rows={r['camera']:r for r in manifest['rows']};source_request=iso/'raw_native_sources/request.json';sources={r['camera']:r for r in json.loads(source_request.read_text())['records']};frames={f['physical_camera']:f for f in json.loads((c/'data/transforms.json').read_text())['frames']}
    results=[];bindings={}
    for cam,x,y in [('KA',1028,331),('HC',1052,576)]:
        rawpath=base/'raw'/f'{cam}.npz';hdpath=base/'archives'/f'{cam}.npz';src=sources[cam];assert sha(rawpath)==rows[cam]['raw_archive_sha256'];assert sha(hdpath)==rows[cam]['archive_sha256'];assert sha(src['image'])==src['image_sha256'];raw=np.load(rawpath);box=raw['crop_xyxy'];frame=frames[cam]
        for key in ['w','h','fl_x','fl_y','cx','cy']:assert frame[key]==rows[cam]['camera_row'][key]
        K,distortion=raw_camera(cam);np.testing.assert_allclose(K,src['raw_OpenCV_K'],rtol=0,atol=1e-12);np.testing.assert_allclose(distortion,src['distortion'],rtol=0,atol=1e-12)
        with np.load(hdpath)as z:original={k:z[k]for k in ['nominal_coverage','source_ineligible','W64']}
        with np.load(c/'data/semantics'/f'{cam}.npz')as z:current={k:z[k]for k in ['grey','black','W64','core','band']}
        coordinates=np.array([[xx+.5,y+.5]for xx in [x-1,x,x+1]],float);raw_centers=raw_array_indices(coordinates,frame);per_key={};full=None
        for key in ['nominal','grey','black','source_ineligible']:
            full=np.zeros((4320,8192),np.float32);full[box[1]:box[3],box[0]:box[2]]=raw[key]
            hd=raw_mask_coverage_to_HD(full,frame);per_key[key]=[float(hd[y,xx])for xx in [x-1,x,x+1]]
            if key=='nominal':
                eligible=hd.copy();eligible[original['source_ineligible']]=0;error=float(np.max(np.abs(eligible-original['nominal_coverage'])));assert error<2e-6,error
        image=Image.open(src['image']).convert('RGB');cx,cy=np.rint(raw_centers[1]-box[:2]).astype(int);left,top=cx-60,cy-60;crop=image.crop((left,top,left+121,top+121));mask_crop=np.asarray(raw['grey'])[top:top+121,left:left+121];nom_crop=np.asarray(raw['nominal'])[top:top+121,left:left+121];assert mask_crop.shape==(121,121)
        overlay=np.array(crop).copy();edge=cv2.morphologyEx(mask_crop.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool);overlay[edge]=[0,255,100];overlay=Image.fromarray(overlay);draw=ImageDraw.Draw(overlay)
        for i,xx in enumerate([x-1,x,x+1]):
            footprint=raw_array_indices(np.array([[xx,y],[xx+1,y],[xx+1,y+1],[xx,y+1]],float),frame);local=footprint-box[:2]-[left,top];draw.polygon([tuple(v)for v in local],outline=['cyan','red','yellow'][i]);native_index=np.rint(raw_centers[i]).astype(int);localidx=native_index-box[:2]
            results.append(dict(camera=cam,HD_pixel=[xx,y],HD_camera_center=[xx+.5,y+.5],RAW_array_center=raw_centers[i].tolist(),RAW_nearest_index=native_index.tolist(),RAW_pixel_footprint_corners=footprint.tolist(),RAW_nearest_grey=bool(raw['grey'][localidx[1],localidx[0]]),RAW_nearest_nominal=bool(raw['nominal'][localidx[1],localidx[0]]),coverage={k:v[i]for k,v in per_key.items()},saved_original_nominal_coverage=float(original['nominal_coverage'][y,xx]),original_W64=float(original['W64'][y,xx]),current_GTv1_grey=bool(current['grey'][y,xx]),current_W64=float(current['W64'][y,xx]),current_core=bool(current['core'][y,xx]),recomputed_full_HD_coverage_max_error=error))
        canvas=Image.new('RGB',(121*4,145),'white');d=ImageDraw.Draw(canvas)
        for i,(label,img)in enumerate([('native RGB',crop),('existing grey mask',Image.fromarray(mask_crop.astype(np.uint8)*255).convert('RGB')),('existing nominal',Image.fromarray(nom_crop.astype(np.uint8)*255).convert('RGB')),('RGB+mask edge+HD cells',overlay)]):canvas.paste(img,(i*121,24));d.text((i*121+2,3),label,fill='black')
        canvas.resize((968,290),Image.Resampling.NEAREST).save(out/f'{cam}_RAW_edge.png');bindings[cam]=dict(raw_archive=str(rawpath),raw_archive_sha256=sha(rawpath),HD_archive=str(hdpath),HD_archive_sha256=sha(hdpath),RAW_source_tiff=src['raw_source'],RAW_source_sha256_from_sealed_request=src['raw_sha256'],native_display_crop=src['image'],native_display_crop_sha256=sha(src['image']),crop_xyxy=box.tolist(),current_semantics_sha256=sha(c/'data/semantics'/f'{cam}.npz'))
    result=dict(kind='EXISTING_RAW_SEMANTIC_EDGE_COVERAGE_DIAGNOSTIC_NOT_ALPHA_GT',records=results,source_bindings=bindings,manifest_sha256=sha(base/'manifest.json'),RAW_request_sha256=sha(source_request),transforms_sha256=sha(c/'data/transforms.json'),rig_sources_sha256={str(RIG/name):sha(RIG/name)for name in ['images.txt','cameras.txt']},coordinate_helper_sha256=sha(Path(__file__).with_name('marshawn_actor_coordinates.py')),helper_sha256=sha(__file__),no_new_segmentation=True,no_GPU=True,leaderboard_unchanged=True,coverage_policy='Exact existing AREA-prefilter RAW8192x4320→HD1920x1012 then one calibrated OpenCV linear remap; binary nominal segmentation is not calibrated optical alpha',limitation='RAW boundary segmentation is imperfect. Coverage cannot be asserted as ground-truth opacity; center label does not determine area coverage.')
    (out/'coverage.json').write_text(json.dumps(result,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.campaign,a.output)
