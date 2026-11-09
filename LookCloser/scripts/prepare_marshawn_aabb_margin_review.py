"""Three margin-only reviews from the SAME pad128 volume; no carving or NN."""
from pathlib import Path
from itertools import product
import argparse,hashlib,json,shutil
import numpy as np
from PIL import Image,ImageDraw
from mask_visual_hull_aabb import rebuild_margins
from marshawn_aabb_overlay import project_box

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def overlay(source,bounds,camera,label):
    image=source.copy();draw=ImageDraw.Draw(image);corners,uv,edges=project_box(bounds,camera)
    origin=np.asarray(camera['transform_matrix'])[:3,3];box=np.asarray(bounds);bits=np.asarray(list(product([0,1],repeat=3)))
    visible=[]
    for i,j in edges:
        common=np.where(bits[i]==bits[j])[0]
        front=any(origin[k]<box[0,k]if bits[i,k]==0 else origin[k]>box[1,k]for k in common)
        visible.append(front)
    for front in [False,True]:
        for (i,j),v in zip(edges,visible):
            if v!=front:continue
            a,b=uv[i],uv[j];length=np.linalg.norm(b-a)
            if not length:continue
            direction=(b-a)/length
            for offset in np.arange(0,length,24):
                xy=[tuple(a+direction*offset),tuple(a+direction*min(offset+14,length))]
                draw.line(xy,fill=(5,5,5),width=5 if front else 4)
                draw.line(xy,fill=(255,35,35)if front else(155,155,155),width=3 if front else 2)
    draw.rectangle((5,4,1100,31),fill=(10,10,10));draw.text((10,9),label,fill='white')
    return image,dict(corners_array_xy=uv.tolist(),edges=edges,edge_camera_facing=visible,
        legend='RED camera-facing box edges; GRAY rear edges; no actor-depth occlusion; FULL image, not clipped')

def main(a):
    a.output.mkdir(parents=True,exist_ok=False);source=json.loads(a.input.read_text());variants={}
    for name,common,bottom in [('common002',.02,None),('common004',.04,None),('common004_bottom006',.04,.06)]:
        folder=a.output/name;folder.mkdir();r=rebuild_margins(a.receipt,a.cells,minimum_margin=common,bottom_margin=bottom,snap_step=.01)
        r.update(uses_eval_segmentation_for_AABB=True,eval_photometric_or_gain_or_NN_fit=False,
            margin_source='SAMEpad128retainedfullcells+364TRAINanchors;no_recarving',prepared_source_manifest_sha256=source['source_manifest_sha256'])
        path=folder/'aabb.json';path.write_text(json.dumps(r,indent=2)+'\n');path.chmod(0o444)
        variants[name]=dict(receipt=str(path),receipt_sha256=sha(path),aabb=r['aabb'])
    rows=[]
    teacher=a.input.parent.parent/'full_context_v2' # Filled from explicit root below.
    root=Path('/fsx/tmp/marshawn_001660_hd');teacher=root/'native_png_radialK4_COLD_R1_DEV3_STAGED/gate20000_HD_common/native_HD'
    for row in source['cameras']:
        if row['camera']not in ['DC','HC','KA']:continue
        cam=row['camera'];rgb=Image.open(row['source_rgb']).convert('RGB');full=Image.open(teacher/('eval_'+cam+'_rgb.png')).convert('RGB')
        contact=Image.new('RGB',(1920,556),(10,10,10));draw=ImageDraw.Draw(contact);entry=dict(camera=cam,variants={})
        for j,(name,variant)in enumerate(variants.items()):
            entry['variants'][name]={}
            for k,(kind,image)in enumerate([('FULL_SOURCE',rgb),('FULL_UNCLIPPED_TEACHER',full)]):
                out,projection=overlay(image,variant['aabb'],row['camera_row'],cam+' '+name+' '+kind+' | red-facing / gray-rear box edges')
                p=a.output/(cam+'_'+name+'_'+kind+'.jpg');out.save(p,quality=96)
                entry['variants'][name][kind]=dict(path=str(p),sha256=sha(p),projection=projection)
                out.thumbnail((640,253));contact.paste(out,(j*640,k*278+25));draw.text((j*640+5,k*278+5),cam+' '+name+' '+kind,fill='white')
        p=a.output/(cam+'_margin_contact.jpg');contact.save(p,quality=96);entry['contact']=str(p);entry['contact_sha256']=sha(p);rows.append(entry)
    selection=dict(kind='ROOT_REQUESTED_FIRST_EXPERIMENTAL_SAFETY_AABB_NOT_FINAL_MINIMUM',
        selected_variant='common004_bottom006',**variants['common004_bottom006'],
        root_instruction='Freezepad128 common.04 lowerZ.06 forfirstexperimentalpipeline; nooptimizer/trainingauthorization',
        source_volume_receipt=str(a.receipt),source_volume_receipt_sha256=sha(a.receipt),
        source_manifest=str(a.input),source_manifest_sha256=sha(a.input),
        uses_eval_segmentation_for_AABB=True,eval_photometric_or_gain_or_NN_fit=False,
        NN_training_authorized=False,not_final_minimal_sufficient=True)
    (a.output/'experimental_aabb_freeze.json').write_text(json.dumps(selection,indent=2)+'\n')
    (a.output/'experimental_aabb_freeze.json').chmod(0o444)
    (a.output/'review_receipt.json').write_text(json.dumps(dict(variants=variants,rows=rows,
        helper_sha256=sha(__file__),source_input_sha256=sha(a.input),source_cells_sha256=sha(a.cells),
        NO_recarving_or_NN=True,experimental_freeze_sha256=sha(a.output/'experimental_aabb_freeze.json')),indent=2)+'\n')
    print(json.dumps(selection),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--cells',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
