"""Save native train-image audits of measured-only head-band mask additions."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from joint_temporal_texture import read,sha,atomic_json,exr,display,ROOT as CAL
from refine_measured_head_masks import ROOT
from transfer_close_boundary_completion import FRAMES
from study_confidence_depth_prior import REGIONS
from review_jaw_repair_transfer import panel


def run():
    gains=read(CAL/'camera_profiles.json');gainmap=dict(zip(gains['physical_cameras'],gains['rgb_gain']))
    exposure=read(CAL/'exposure.json')['fixed_exposure_gain'];records=[]
    for frame in FRAMES:
        root=ROOT/frame;q=read(root/'request.json');r=read(root/'result.json')
        assert r['request_sha256']==sha(root/'request.json')
        for p,h in r['hashes'].items():
            assert sha(root/p)==h
        names=read(root/'cameras.json');old=np.load(Path(q['original_mask_root'])/'masks.npz')['masks'].astype(bool)
        updated=np.load(root/'masks.npz')['masks'];rows={row['physical_camera']:row for row in q['rows']}
        chosen=sorted(set([REGIONS[frame]['camera'],'D004_D005_1210LZ']+
            [v['camera'] for v in sorted(r['records'],key=lambda v:-v['added_mask_pixels'])[:2]]))
        for name in chosen:
            ci=names.index(name);folder=root/'cameras'/name;cr=read(folder/'result.json')
            for p,h in cr['hashes'].items():
                assert sha(folder/p)==h
            e=np.load(folder/'evidence.npz');seeds=e['seeds'];added=updated[ci]&~old[ci]
            row=rows[name];assert sha(row['file_path'])==q['rgb_receipt']['source_rgb_hashes'][row['file_path']]
            rgb=np.rint(display(exr(row['file_path'])*gainmap[name],exposure)*255).clip(0,255).astype(np.uint8)
            overlay=rgb.copy();overlay[old[ci]&~binary_erosion(old[ci])]=[255,255,0]
            overlay[added]=[255,40,220];overlay[seeds]=[30,240,240]
            ims=[np.rot90(rgb).copy(),np.rot90(overlay).copy()]
            overview=[np.array(Image.fromarray(im).resize((360,640),Image.Resampling.LANCZOS)) for im in ims]
            out=root/'review';out.mkdir(exist_ok=True)
            panel(out/(name+'_overview.png'),overview,['train RGB','yellow old / magenta added / cyan seeds'],(0,0,360,640))
            yy,xx=np.nonzero(np.rot90(seeds));assert len(xx)
            # Review framing only; never trims the saved mask or accepted seeds.
            x0,y0=np.floor([np.quantile(xx,.01)-20,np.quantile(yy,.01)-20]).astype(int)
            x1,y1=np.ceil([np.quantile(xx,.99)+21,np.quantile(yy,.99)+21]).astype(int)
            box=(max(0,int(x0)),max(0,int(y0)),min(1080,int(x1)),min(1920,int(y1)))
            panel(out/(name+'_native.png'),ims,['train RGB',name+' measured mask expansion'],box)
            records.append(dict(frame=frame,camera=name,box=box,seeds=int(seeds.sum()),added=int(added.sum()),
                source_rgb_sha256=sha(row['file_path']),images={str(out/(name+'_'+part+'.png')):sha(out/(name+'_'+part+'.png')) for part in ['overview','native']}))
    atomic_json(ROOT/'review_manifest.json',dict(records=records,visual_status='pending',
        selection='two largest added counts plus fixed D/D and prior hair-reference camera',
        all_masks_saved=True,script_sha256=sha(__file__),production_updated=False))
    print('Wrote',len(records),'camera review pairs',flush=True)


if __name__=='__main__':
    run()
