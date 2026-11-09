"""CPU full-source/unclipped-teacher context; NEVER infer fit from clipped RGB."""
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw
from marshawn_aabb_overlay import overlay_box

R=Path('/fsx/tmp/marshawn_001660_hd')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(a):
    source=json.loads(a.input.read_text());summary=json.loads((a.experiment/'summary.json').read_text())
    proofpath=R/'two_actor_isolation/automatic_aabb/teacher20k_prepost_pipeline_proof.json';proof=json.loads(proofpath.read_text())
    if proof['ALL_pipeline_tensors_identical']is not True:raise ValueError('Unclipped teacher is not the same selected field/grid')
    a.output.mkdir(parents=True,exist_ok=False);rows=[];strips=[]
    teacher=R/'native_png_radialK4_COLD_R1_DEV3_STAGED/gate20000_HD_common/native_HD'
    for row in source['cameras']:
        cam=row['camera'];camera=row['camera_row']
        if sha(row['source_rgb'])!=row['source_rgb_sha256']:raise ValueError('Fullsource RGB changed')
        rgb=Image.open(row['source_rgb']).convert('RGB');panels=[('FULL actualsource; no clip',rgb)];record=dict(camera=cam,source_rgb=row['source_rgb'],source_rgb_sha256=row['source_rgb_sha256'],variants={})
        for name,bounds in summary['boxes'].items():
            out,projection=overlay_box(rgb,bounds,camera,label=cam+' '+name+' COMPUTED; FULL SOURCE, NOT CLIPPED')
            path=a.output/(cam+'_'+name+'_FULL_SOURCE.jpg');out.save(path,quality=95)
            record['variants'][name]=dict(aabb=bounds,projection=projection,full_source_overlay=str(path),full_source_overlay_sha256=sha(path))
            panels.append((name+' FULL source +12AABB edges',out))
        strip=Image.new('RGB',(2560,363),(15,15,15));draw=ImageDraw.Draw(strip)
        for i,(title,im)in enumerate(panels):
            im=im.copy();im.thumbnail((640,337));strip.paste(im,(i*640,26));draw.text((i*640+4,5),cam+' '+title,fill='white')
        strip.save(a.output/(cam+'_all_computed_FULL_SOURCE.jpg'),quality=95);strips.append(strip)
        if cam in ['DC','HC','KA']:
            full=Image.open(teacher/('eval_'+cam+'_rgb.png')).convert('RGB');full.save(a.output/(cam+'_FULL_UNCLIPPED_TEACHER.png'))
            for name,bounds in summary['boxes'].items():
                over,projection=overlay_box(full,bounds,camera,label=cam+' '+name+' SAME20K FULL TEACHER; NOT CLIPPED')
                path=a.output/(cam+'_'+name+'_FULL_UNCLIPPED_TEACHER.jpg');over.save(path,quality=95)
                record['variants'][name]['full_teacher_overlay']=str(path);record['variants'][name]['full_teacher_overlay_sha256']=sha(path)
            record['full_teacher_rgb_source']=str(teacher/('eval_'+cam+'_rgb.png'));record['full_teacher_rgb_sha256']=sha(teacher/('eval_'+cam+'_rgb.png'))
        rows.append(record)
    pages=[]
    for start in range(0,len(strips),6):
        page=Image.new('RGB',(2560,363*len(strips[start:start+6])),(15,15,15))
        for j,strip in enumerate(strips[start:start+6]):page.paste(strip,(0,363*j))
        path=a.output/('page_%02d.jpg'%(start//6));page.save(path,quality=94);pages.append(dict(path=str(path),sha256=sha(path),cameras=[r['camera']for r in rows[start:start+6]]))
    (a.output/'receipt.json').write_text(json.dumps(dict(kind='FULL_UNCLIPPED_SOURCE_AND_SAME_TEACHER_COMPUTED_BOX_CONTEXT',
        input_manifest_sha256=sha(a.input),computed_summary_sha256=sha(a.experiment/'summary.json'),rows=rows,pages=pages,
        teacher_pipeline_equality_proof=str(proofpath),teacher_pipeline_equality_proof_sha256=sha(proofpath),
        source_helper_sha256=sha(__file__),overlay_helper_sha256=sha(Path(__file__).with_name('marshawn_aabb_overlay.py')),
        no_model_or_GPU_forward=True,no_crop_or_RGBmask_compositing=True,clipped_render_separate_and_pending=True,
        limits='Rayintersection andprojected8cornerhull are not actor surface depth certification; root source and independentfoot reviews required.'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a)
