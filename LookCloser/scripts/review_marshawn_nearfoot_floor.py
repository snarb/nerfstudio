"""Make raw TRAIN triplets for preselected existing tracks; leaves material qualification to review."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image,ImageDraw


def main(a):
    sys.path.insert(0,str(a.helpers));sys.path.insert(0,str(a.scene/'code_native'))
    from marshawn_actor_coordinates import raw_array_indices
    from prepare_marshawn_color_variants import vendor,srgb,sha
    tfpath=a.scene/'data/transforms.json';tf=json.loads(tfpath.read_text());rows={r['physical_camera']:r for r in tf['frames']if r['file_path']in tf['train_filenames']}
    requestpath=a.scene/'two_actor_isolation/raw_native_sources/request.json';requests={r['camera']:r for r in json.loads(requestpath.read_text())['records']}
    records=json.loads(a.candidates.read_text())['records'];ids=[int(p)for p in a.ids.split(',')];selected=[next(r for r in records if r['source_point_id']==pid)for pid in ids]
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'native_patches').mkdir();raws={};sources={};result=[]
    pq=vendor();matrix=pq.rgb_to_rgb_matrix('rwg','rec709')
    for r in selected:
        obs=r['verified_feature_graph']['observations'];take=[min(obs,key=lambda o:o['fitted_error_hd_px'])]
        while len(take)<3:
            rest=[o for o in obs if o not in take]
            take.append(max(rest,key=lambda o:min(np.linalg.norm(np.array(rows[o['camera']]['transform_matrix'])[:3,3]-np.array(rows[q['camera']]['transform_matrix'])[:3,3])for q in take)))
        sheet=Image.new('RGB',(1536,850),(20,20,20));draw=ImageDraw.Draw(sheet);views=[]
        for j,ob in enumerate(take):
            cam=ob['camera'];req=requests[cam];assert req['split']=='TRAIN'and cam not in ['DC','HC','KA'];path=Path(req['raw_source'])
            if cam not in raws:
                assert sha(path)==req['raw_sha256'];raws[cam]=tifffile.memmap(path,mode='r');sources[cam]=req
            pixel=raw_array_indices(np.array(ob['detected_uv'])[None],rows[cam])[0];center=np.rint(pixel).astype(int);lo=np.maximum(center-256,0);hi=np.minimum(center+256,[8192,4320]);raw=np.asarray(raws[cam][lo[1]:hi[1],lo[0]:hi[0]])
            lin=np.maximum(np.maximum(pq.log3g10_decode(raw.astype(np.float64)/65535),0)@matrix.T,0).astype(np.float32);im=Image.fromarray(srgb(lin*4));native=a.output/'native_patches'/('p%d_%s.png'%(r['source_point_id'],cam));im.save(native)
            x,y=pixel-lo;mark=im.copy();d=ImageDraw.Draw(mark)
            for line in[(x-18,y,x-5,y),(x+5,y,x+18,y),(x,y-18,x,y-5),(x,y+5,x,y+18)]:d.line(line,fill='red',width=2)
            sheet.paste(mark,(512*j,55));zoom=im.crop((round(x)-64,round(y)-64,round(x)+64,round(y)+64)).resize((256,256));sheet.paste(zoom,(512*j+128,580))
            draw.text((j*512+5,5),'p%d %s RAW(%.3f,%.3f)'%(r['source_point_id'],cam,*pixel),fill='white');draw.text((j*512+5,22),'z=%.6f | fit %.4f HDpx'%(r['xyz_floor'][2],ob['fitted_error_hd_px']),fill='white');draw.text((j*512+5,38),'native1:1 + unmarked center128px at2x',fill='white')
            views.append(dict(ob,raw_array_xy=pixel.tolist(),native_crop_xyxy=[*lo.tolist(),*hi.tolist()],native_patch=str(native),native_patch_sha256=sha(native),raw_source=str(path),raw_sha256=req['raw_sha256']))
        triplet=a.output/('p%d_RAW_triplet.jpg'%r['source_point_id']);sheet.save(triplet,quality=97)
        result.append(dict(r,views=views,triplet=str(triplet),triplet_sha256=sha(triplet),manual_material_label='UNREVIEWED'))
        print(r['source_point_id'],flush=True)
    report=dict(records=result,source_raw=sources,bindings={str(p):sha(p)for p in[a.candidates,tfpath,requestpath,Path(__file__),a.helpers/'marshawn_actor_coordinates.py',a.helpers/'prepare_marshawn_color_variants.py']},
                no_EVAL_used=True,no_native_density_used=True,display='RAW Log3G10 RWG→Rec709 fixedgain4/Reinhard+sRGB, same existing source display. Markers leave central5pixels untouched.',
                uncertainty='Actual SIFT feature center may be scale-dependent and straddle shadow/seam/contact boundaries. No material/depth certificate until review.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['scene','helpers','candidates','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--ids',required=True);main(p.parse_args())
