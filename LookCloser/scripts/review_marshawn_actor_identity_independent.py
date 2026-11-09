"""Build source-bound selected-ID contacts for a separate manual identity review."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from PIL import Image,ImageDraw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build(master,request,output,expected_sha):
    if sha(master)!=expected_sha:raise ValueError('Identity master changed before export')
    records=json.loads(Path(master).read_text())['rows'];sources={r['camera']:r for r in json.loads(Path(request).read_text())['records']}
    output=Path(output);output.mkdir(parents=True,exist_ok=False);rows=[];tiles=[]
    for r in records:
        s=sources[r['camera']];npz=Path(r['candidates_npz'])
        if sha(npz)!=r['source_candidate_npz_sha256']or sha(s['image'])!=s['image_sha256']:
            raise ValueError('Bound candidate/source changed')
        with np.load(npz,allow_pickle=False)as z:
            black=z['masks'][r['black_actor_instance']].copy();grey=z['masks'][r['grey_actor_instance']].copy()
        with Image.open(s['image'])as image:source=image.convert('RGB');assert source.size==black.shape[::-1]
        rgb=np.asarray(source).copy();blend=rgb.astype(np.float32)
        blend[black]=.55*blend[black]+.45*np.array([0,255,255]);blend[grey]=.55*blend[grey]+.45*np.array([255,0,180])
        for m,color in [(black,[0,255,255]),(grey,[255,0,180])]:
            edge=m&(~np.roll(m,1,0)|~np.roll(m,-1,0)|~np.roll(m,1,1)|~np.roll(m,-1,1));blend[edge]=color
        overlay=Image.fromarray(blend.astype(np.uint8));tile=Image.new('RGB',(720,520),(20,20,20));draw=ImageDraw.Draw(tile)
        title=f"{r['camera']} BLACK ID {r['black_actor_instance']} cyan | GREY ID {r['grey_actor_instance']} magenta"
        draw.text((5,5),title,fill='white');draw.text((5,23),'Original RGB | selected instance overlays; not pixel-perfect hair certification',fill='white')
        for x,im in [(0,source),(360,overlay)]:
            im=im.copy();im.thumbnail((356,470));tile.paste(im,(x+(356-im.width)//2,45+(470-im.height)//2))
        out=output/(r['camera']+'.jpg');tile.save(out,quality=93)
        row=dict(camera=r['camera'],black_actor_instance=r['black_actor_instance'],grey_actor_instance=r['grey_actor_instance'],
            verified_other_person_ids=r['verified_other_person_ids'],candidate_npz=str(npz),candidate_sha256=sha(npz),
            original_source=str(s['image']),original_source_sha256=sha(s['image']),selected_contact=str(out),selected_contact_sha256=sha(out),
            black_pixels=int(black.sum()),grey_pixels=int(grey.sum()),overlap_pixels=int((black&grey).sum()))
        rows.append(row);tiles.append(tile)
        print(r['camera'],flush=True)
    pages=[]
    for i in range(0,len(tiles),4):
        page=Image.new('RGB',(1440,1040),(25,25,25))
        for j,tile in enumerate(tiles[i:i+4]):page.paste(tile,((j%2)*720,(j//2)*520))
        out=output/('page_%02d.jpg'%(i//4));page.save(out,quality=94)
        pages.append(dict(path=str(out),sha256=sha(out),cameras=[r['camera']for r in rows[i:i+4]]))
    if sha(master)!=expected_sha:raise ValueError('Identity master changed during export; review invalid')
    (output/'contact_manifest.json').write_text(json.dumps(dict(master=str(master),master_sha256=expected_sha,
        raw_request=str(request),raw_request_sha256=sha(request),rows=rows,pages=pages,builder_sha256=sha(__file__),
        manual_review_complete=False,scope='Identity and major body/occlusion only; not hair matte or alpha GT'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--master',type=Path,required=True);p.add_argument('--request',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-sha',required=True);a=p.parse_args()
    build(a.master,a.request,a.output,a.expected_sha)
