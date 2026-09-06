#!/usr/bin/env python3
"""Native train-only crops for visual validation of temporal feature identities."""
from __future__ import annotations
import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from audit_fixed_camera_feature_geometry import farthest_points


def main():
    from nerfstudio.data.utils.data_utils import load_exr_image
    from convert_exr_nerfstudio_to_jpeg import tone_map
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Preserve previous review')
    audit=json.loads(a.audit.read_text());receipts={(r['frame_id'],r['physical_camera']):r for r in audit['ingest_receipts']}
    images={};hashes={str(a.audit):sha256(a.audit)}
    def image(frame_id,name):
        key=(frame_id,name)
        if key not in images:
            receipt=receipts[key];path=Path(receipt['source']);digest=sha256(path)
            if digest!=audit['input_hashes'][str(path)]:raise ValueError('Changed EXR source')
            hashes[str(path)]=digest
            rgb=tone_map(load_exr_image(path),receipt['gain']);memory=BytesIO()
            Image.fromarray(rgb).save(memory,format='JPEG',quality=98,subsampling=0)
            if hashlib.sha256(memory.getvalue()).hexdigest()!=receipt['jpeg_sha256']:
                raise ValueError('In-memory JPEG no longer reproduces feature audit input')
            images[key]=Image.open(BytesIO(memory.getvalue())).convert('RGB')
        return images[key]
    a.output.mkdir(parents=True);rows=[]
    for row in audit['results']:
        records=[r for r in row['records'] if r['group']=='moving' and len(r['matches'])==5]
        chosen=farthest_points(np.array([r['reference_xy'] for r in records]),min(len(records),6)) if records else []
        if not chosen:continue
        canvas=Image.new('RGB',(6*104,len(chosen)*124))
        for i,index in enumerate(chosen):
            r=records[index]
            panels=[(row['frame_id'],audit['reference_camera'],r['reference_xy'],'reference')]+[
                (row['window'][str(t)],row['secondary_camera'],r['matches'][str(t)]['point'],
                 f'{t:+d}: e={r["matches"][str(t)]["residual"]:.2f}') for t in range(-2,3)]
            for j,(frame_id,name,point,label) in enumerate(panels):
                x,y=np.rint(point).astype(int);crop=image(frame_id,name).crop((x-48,y-48,x+48,y+48)).rotate(90)
                ImageDraw.Draw(crop).ellipse((45,45,51,51),outline='red')
                canvas.paste(crop,(j*104,i*124+26));ImageDraw.Draw(canvas).text((j*104+1,i*124+3),label,fill='white')
        path=a.output/f'{row["frame_id"]}_{row["secondary_camera"]}.png';canvas.save(path)
        rows.append(dict(path=str(path),sha256=sha256(path),frame_id=row['frame_id'],secondary_camera=row['secondary_camera'],
                         reference_indices=[records[i]['reference_index'] for i in chosen]))
        images.clear()
    atomic_json(a.output/'review_manifest.json',dict(uses_eval_rgb=False,changes_prediction=False,
        selection='Spatially spread moving reference features matched at all five offsets; not selected by favorable timing result',
        crop_size=96,native_scale=True,rotation=90,rows=rows,input_hashes=hashes,script_sha256=sha256(Path(__file__))))
    print(json.dumps({'sheets':len(rows),'correspondences':sum(len(r['reference_indices']) for r in rows)}))


if __name__=='__main__':main()
