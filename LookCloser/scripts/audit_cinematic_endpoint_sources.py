"""Read-only train RGB endpoint study for cinematic dynamic push-ins.

Actual native pixels, fixed frozen display response. This does not render a
candidate, fit geometry, pick held-out data or assert novel-view quality.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw
from joint_temporal_texture import cameras, exr, display, read, sha, atomic_json, ROOT as COLOR

ROOT=Path('/mnt/data/dec5_cinematic_endpoint_sources')
FRAMES=['001151','001173','001185','001197']
PREFIXES=['G004_C005','H004_C005','I004_C005','J004_C005','K004_C005']
BOX=(100,500,1000,1350)


def run():
    ROOT.mkdir(exist_ok=False)
    log_gain=np.load(COLOR/'parameters.npz')['log_gain']
    gain=np.exp(log_gain-log_gain.mean(0,keepdims=True))
    exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']
    records=[]
    for frame in FRAMES:
        rows,_,_=cameras(frame); chosen=[]
        for prefix in PREFIXES:
            matches=[(i,r) for i,r in enumerate(rows) if r['physical_camera'].startswith(prefix)]
            assert len(matches)==1; chosen.append(matches[0])
        def load(pair):
            i,row=pair; rgb=np.rint(display(exr(row['file_path'])*gain[i],exposure)*255).clip(0,255).astype(np.uint8)
            return row,np.rot90(rgb)
        with ThreadPoolExecutor(max_workers=4) as pool:images=list(pool.map(load,chosen))
        sheet=Image.new('RGB',(5*360,364));draw=ImageDraw.Draw(sheet)
        for j,(row,array) in enumerate(images):
            name=row['physical_camera'];folder=ROOT/frame/name;folder.mkdir(parents=True)
            path=folder/'train_native.png';Image.fromarray(array).save(path)
            crop=Image.fromarray(array).crop(BOX);crop.save(folder/'head_native.png')
            sheet.paste(crop.resize((360,340)),(j*360,24));draw.text((j*360+3,3),frame+' '+name,fill='white')
            records.append(dict(frame=frame,camera=name,source=row['file_path'],source_sha256=sha(row['file_path']),
                rendered=False,box=BOX,crop_resampled=False,
                files={str(p):sha(p) for p in folder.iterdir() if p.is_file()}))
        sheet.save(ROOT/(frame+'_contact.png'))
    atomic_json(ROOT/'request.json',dict(frames=FRAMES,train_prefixes=PREFIXES,script_sha256=sha(__file__),
        parameters_sha256=sha(COLOR/'parameters.npz'),exposure_sha256=sha(COLOR/'exposure.json'),
        quality_metrics=False,heldout_used=False,generated_pixels=False,records=records))
    print('Prepared20 native train endpoint witnesses; not candidate renders',flush=True)


if __name__=='__main__':run()
