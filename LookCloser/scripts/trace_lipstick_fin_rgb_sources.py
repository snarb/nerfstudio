"""Trace actual hard RGB selections on the diagnostic fin to measured layers."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from diagnose_lipstick_fin_depth import ROOT,FRAME,CAMERA,POLYGON
from review_full_block_transfer import ROOT as DEPTH_ROOT
from study_confidence_depth_prior import load_real,unproject,project_integer
from calibrated_depth_witness import load_images
from review_jaw_repair_transfer import verified_image


def run():
    out=ROOT/'rgb_trace';out.mkdir(exist_ok=False)
    folder=DEPTH_ROOT/FRAME/'rgb'/CAMERA/'production'
    _,receipt=verified_image(folder,FRAME);rows,depths,depth_receipt=load_real(DEPTH_ROOT,FRAME)
    assert [r['physical_camera'] for r in rows]==receipt['source_cameras']
    target=receipt['camera'];root=folder/'frames'/FRAME
    complete=read(root/'complete.json')
    for name in ['source_ids.png','target_depth.npz']:assert sha(root/name)==complete['hashes'][name]
    depth=np.load(root/'target_depth.npz')['depth'];sources=np.asarray(Image.open(root/'source_ids.png'))
    mask=Image.new('L',(1080,1920));ImageDraw.Draw(mask).polygon(POLYGON,fill=1)
    mask=np.rot90(np.asarray(mask,dtype=bool),-1)
    y,x=np.nonzero(mask&(depth>0)&(sources<62))
    points=unproject(target,x,y,depth[y,x],offset=.5)
    chosen=sources[y,x];valid=np.zeros(len(x),bool);deltas=np.zeros(len(x));uvs=np.zeros((len(x),2))
    for ci,(row,d) in enumerate(zip(rows,depths)):
        j=np.flatnonzero(chosen==ci)
        if not len(j):continue
        uv,z=project_integer(row,points[j]);uvs[j]=uv
        xy=np.rint(uv).astype(int);ok=(z>0)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
        jj=np.flatnonzero(ok);obs=d[xy[jj,1],xy[jj,0]];fine=np.isfinite(obs)&(obs>0)
        valid[j[jj[fine]]]=True;deltas[j[jj[fine]]]=obs[fine]-z[jj[fine]]
    images,_,rgb_receipt=load_images(FRAME);pred=np.asarray(Image.open(root/'prediction_native.png'))
    records=[];examples=[]
    for ci in np.unique(chosen):
        j=np.flatnonzero(chosen==ci);far=j[valid[j]&(deltas[j]>.005)]
        record=dict(camera=rows[ci]['physical_camera'],chosen_pixels=len(j),
            measured_available=int(valid[j].sum()),farther_measured_depth_pixels=len(far),
            near_measured_depth_pixels=int((valid[j]&(np.abs(deltas[j])<=.0015)).sum()))
        if len(far):
            k=int(far[np.argsort(deltas[far])[len(far)//2]])
            u,v=np.rint(uvs[k]).astype(int);tx,ty=int(x[k]),int(y[k])
            sheet=Image.new('RGB',(400,240));draw=ImageDraw.Draw(sheet)
            for col,(image,cx,cy,title) in enumerate([(pred,tx,ty,'prediction target pixel'),
                (images[rows[ci]['physical_camera']],u,v,'actual selected source RGB')]):
                crop=Image.fromarray(image).crop((cx-90,cy-90,cx+90,cy+90))
                ImageDraw.Draw(crop).ellipse((87,87,93,93),outline='red',width=1)
                sheet.paste(crop,(col*200,50));draw.text((col*200+2,28),title,fill='white')
            draw.text((2,5),f'{rows[ci]["physical_camera"]} measured z - mesh z = {deltas[k]:+.5f}',fill='white')
            path=out/f'case_{len(examples):02d}.png';sheet.save(path)
            examples.append(dict(camera=rows[ci]['physical_camera'],target_xy=[tx,ty],
                source_uv=uvs[k].tolist(),delta=float(deltas[k]),image=path.name,sha256=sha(path)))
        records.append(record)
    atomic_json(out/'result.json',dict(frame=FRAME,source_complete_sha256=sha(root/'complete.json'),
        diagnostic_request_sha256=sha(ROOT/'request.json'),depth_receipt=depth_receipt,
        rgb_receipt=rgb_receipt,script_sha256=sha(__file__),
        pixels=len(x),per_actual_selected_camera=records,examples=examples,
        rgb_averaging=False,geometry_changed=False,visual_status='pending'))
    print('Actual selected-source attribution',records,flush=True)


if __name__=='__main__':run()
