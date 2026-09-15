"""Read-only endpoint comparison with the real train image through the virtual lens.

The calibrated train extrinsic must coincide. Reprojecting a train image with
only intrinsics changed requires no mesh or inferred depth. These comparison
images never enter rendering; no held-out or quality metric is involved.
"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw
from joint_temporal_texture import cameras, exr, display, read, sha, atomic_json, ROOT as COLOR
from audit_cinematic_path_requests import raw_matrix, CALIBRATION
from review_jaw_repair_transfer import verified_image


def run(base, output, variant, frame):
    folder=base/variant; prediction,receipt=verified_image(folder,frame)
    request=read(folder/'request.json'); entry=next(r for r in request['inventory'] if r['frame_id']==frame)
    target=entry['camera']; rows,_,_=cameras(frame); name=request['camera_path_report']['endpoint_train_camera']
    ci=next(i for i,r in enumerate(rows) if r['physical_camera']==name); source=rows[ci]
    calibration=read(CALIBRATION)
    actual=next(r for r in calibration['frames'] if r['physical_camera']==name)
    pose=raw_matrix(target['transform_matrix'],read(entry['metadata']),calibration)
    np.testing.assert_allclose(pose,actual['transform_matrix'],atol=1e-10,rtol=0)
    yy,xx=np.mgrid[:1080,:1920]
    u=(xx+.5-target['cx'])/target['fl_x']*source['fl_x']+source['cx']-.5
    v=(yy+.5-target['cy'])/target['fl_y']*source['fl_y']+source['cy']-.5
    assert u.min()>=0 and v.min()>=0 and u.max()<1919 and v.max()<1079
    ix,iy=np.floor(u).astype(int),np.floor(v).astype(int);fu,fv=u-ix,v-iy
    linear=exr(source['file_path']);rgb=np.zeros((1080,1920,3),np.float32)
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        weight=(fu if dx else 1-fu)*(fv if dy else 1-fv)
        rgb+=linear[iy+dy,ix+dx]*weight[...,None]
    log_gain=np.load(COLOR/'parameters.npz')['log_gain'];gain=np.exp(log_gain-log_gain.mean(0,keepdims=True))[ci]
    exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']
    gt=np.rot90(np.rint(display(rgb*gain,exposure)*255).clip(0,255).astype(np.uint8))
    data=folder/'frames'/frame
    depth=np.rot90(np.load(data/'target_depth.npz')['depth'])
    ids=np.rot90(np.asarray(Image.open(data/'source_ids.png')))
    assert receipt['source_cameras'][ci]==name
    # Mutually exclusive raw ray/source categories. No semantic interpretation.
    labels=np.zeros(depth.shape,np.uint8)
    labels[depth==0]=1
    labels[(depth>0)&(ids==255)]=2
    labels[(depth>0)&(ids<62)&(ids!=ci)]=3
    colors=np.array([[20,150,70],[255,30,30],[30,100,255],[255,190,20]],np.uint8)
    palette=colors[labels]
    out=output/variant/frame;out.mkdir(parents=True,exist_ok=False)
    for n,im in [('train_gt_virtual_lens.png',gt),('prediction.png',prediction),('source_categories.png',palette)]:
        Image.fromarray(im).save(out/n)
    sheet=Image.new('RGB',(3*432,800));draw=ImageDraw.Draw(sheet)
    for i,(im,label) in enumerate([(gt,'real H/C, same virtual lens'),(prediction,'mesh prediction'),(palette,'green own / yellow other / red no mesh')]):
        sheet.paste(Image.fromarray(im).resize((432,768)),(432*i,32));draw.text((432*i+3,5),label,fill='white')
    sheet.save(out/'comparison.png')
    np.savez_compressed(out/'evidence.npz',source_id=ids,depth=depth,ray_category=labels,
        source_u=u,source_v=v)
    atomic_json(out/'result.json',dict(frame=frame,variant=variant,endpoint_train_camera=name,own_source_id=ci,
        request_sha256=sha(folder/'request.json'),source_exr=source['file_path'],source_exr_sha256=sha(source['file_path']),
        profile_sha256=sha(COLOR/'parameters.npz'),exposure_sha256=sha(COLOR/'exposure.json'),script_sha256=sha(__file__),
        actual_extrinsic_max_error=float(np.abs(pose-np.asarray(actual['transform_matrix'])).max()),
        gt_generated_without_mesh=True,gt_is_real_train_rgb_not_diffusion=True,heldout_used=False,
        quality_metrics=False,renderer_changed=False,semantic_roi_not_yet_defined=True,
        hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',required=True);p.add_argument('--frame',required=True)
    a=p.parse_args();run(a.base,a.output,a.variant,a.frame)
