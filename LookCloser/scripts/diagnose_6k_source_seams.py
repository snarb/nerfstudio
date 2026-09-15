"""Post-hoc native source-label diagnostics; never edits production renders."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label
from joint_temporal_texture import read, sha, atomic_json

ROOT=Path('/mnt/data/dec5_6k_source_seam_diagnosis')
NEW=Path('/mnt/data/dec5_cinematic_wide_spiral_6k_output_v1')
OLD=Path('/mnt/data/dec5_cinematic_wide_spiral_6k_v2')


def boundaries(ids):
    result=np.zeros(ids.shape,bool)
    result[:,1:]|=ids[:,1:]!=ids[:,:-1]
    result[:,:-1]|=ids[:,1:]!=ids[:,:-1]
    result[1:]|=ids[1:]!=ids[:-1]
    result[:-1]|=ids[1:]!=ids[:-1]
    return result


def main():
    ROOT.mkdir(exist_ok=False)
    frame='001083';folder=NEW/'frames'/frame;old=OLD/'frames'/frame
    bindings={}
    for base in [NEW,OLD]:
        d=base/'frames'/frame;complete=read(d/'complete.json')
        assert sha(base/'request.json')==complete['request_sha256']
        bindings[str(base/'request.json')]=sha(base/'request.json')
        bindings[str(d/'complete.json')]=sha(d/'complete.json')
        for name in ['frame.png','source_ids.png']:
            assert sha(d/name)==complete['hashes'][name]
            bindings[str(d/name)]=sha(d/name)
    im=Image.open(folder/'frame.png');assert im.size==(3456,6144)
    ids=np.rot90(np.asarray(Image.open(folder/'source_ids.png')))
    low=np.rot90(np.asarray(Image.open(old/'source_ids.png')))
    reference=np.asarray(Image.fromarray(low).resize(im.size,Image.Resampling.NEAREST))
    old_image=Image.open(old/'frame.png').resize(im.size,Image.Resampling.BICUBIC)
    palette=np.random.default_rng(41).integers(25,240,(256,3),dtype=np.uint8);palette[255]=0
    camera_names=read(Path(read(NEW/'request.json')['parent'])/'frames'/frame/'result.json')['source_cameras']
    crops=read(NEW/'review/canary_crops.json')
    records=[]
    for row in crops['records']:
        if row['frame']!=frame:continue
        x0,y0,x1,y1=row['box'];crop=ids[y0:y1,x0:x1];control=reference[y0:y1,x0:x1]
        rgb=np.asarray(im.crop(row['box']));edges=boundaries(crop)
        overlay=rgb.copy();overlay[edges]=[255,30,20]
        changed=crop!=control;diff=rgb.copy();diff[changed]=[255,255,0]
        parts=[('HD RGB enlarged control',old_image.crop(row['box'])),('Native 6K RGB',Image.fromarray(rgb)),
            ('6K source-ID boundaries',Image.fromarray(overlay)),('Native source IDs',Image.fromarray(palette[crop])),
            ('Changed vs enlarged HD IDs',Image.fromarray(diff))]
        w,h=rgb.shape[1],rgb.shape[0];canvas=Image.new('RGB',(w*len(parts),h+28));draw=ImageDraw.Draw(canvas)
        for i,(title,panel) in enumerate(parts):canvas.paste(panel,(i*w,28));draw.text((i*w+4,6),title,fill='white')
        path=ROOT/f'{frame}_{row["region"]}.png';canvas.save(path)
        sources=[]
        for ci in np.unique(crop):
            cc,n=label(crop==ci);areas=np.bincount(cc.ravel())[1:]
            sources.append(dict(index=int(ci),camera=camera_names[ci] if ci!=255 else 'unsupported',
                pixels=int((crop==ci).sum()),components=n,component_areas_descending=sorted(areas.tolist(),reverse=True)))
        records.append(dict(region=row['region'],box=row['box'],sources=sources,
            changed_source_pixels=int(changed.sum()),pixels=crop.size,source_boundary_pixels=int(edges.sum()),
            image=str(path),image_sha256=sha(path)))
    atomic_json(ROOT/'result.json',dict(frame=frame,records=records,input_hashes=bindings,script_sha256=sha(__file__),
        diagnostic_only=True,production_modified=False,source_boundary_does_not_alone_prove_rgb_error=True))
    print([(r['region'],r['changed_source_pixels'],len(r['sources'])) for r in records],flush=True)


if __name__=='__main__':main()
