"""GPU-sequential model preservation, numerical-repeat and dense depth audits."""
import argparse
from pathlib import Path
import hashlib
import time
import numpy as np
import torch
from PIL import Image
from luster_hybrid_common import ORIGIN,TARGETS,geometry,image,log,native_rows,read,write
from render_luster_hybrid import load,packed_quantiles,state_hashes


def audit(root):
    cfg,m=load(root,128)
    from render_luster_quality import camera_for,restrict_to_checkpoint_occupancy,median_depth,measured_plate
    old=read(ORIGIN/'evaluations/cold_body80_fas50_threshold095_zero/000106316/full_review/receipt.json')
    grids={k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in
           [('occupancy',m.occupancy_grid.binaries),('frequency',m.freq_grid.grid)]}
    assert grids==old['grids_before']
    before=state_hashes(m);rows=native_rows();out=root/'diagnostics';out.mkdir(exist_ok=True)
    report=dict(grids=grids,rows=[],repeat_max_abs=0.)
    # These boxes are diagnostics only, never admission masks for prediction.
    boxes={7:dict(skin=[1550,650,1678,778],eyes=[1570,520,1698,648]),
           95:dict(lace=[1740,2600,1868,2728]),150:dict(cuff=[930,3820,1058,3948])}
    with torch.inference_mode():
        for c,regions in boxes.items():
            row=rows[c];camera=camera_for(row,m.device);plate,_=measured_plate(cfg,row)
            for name,box in regions.items():
                x0,y0,x1,y1=box;yy,xx=np.mgrid[y0:y1,x0:x1];coords=torch.tensor(np.stack([yy,xx],-1).reshape(-1,2),device=m.device,dtype=torch.float32)+.5
                dense=[];qvalues=[];repeat=[]
                for mode in ['arm','arm','arm',8192,16384]:
                    colors=[];depths=[]
                    for start in range(0,len(coords),128):
                        cc=coords[start:start+128];rays=m.collider(camera.generate_rays(camera_indices=0,coords=cc))
                        if mode=='arm':
                            result=m.get_outputs(rays);q=packed_quantiles(result,rays)[:,1]
                        else:
                            m.config.fixed_num_samples_per_ray=mode
                            with restrict_to_checkpoint_occupancy(m,True):result=m.fixed_ray_marching(rays)
                            q,_=median_depth(result['loss_weights'],rays.nears,rays.fars)
                        z=q*(-(rays.directions@camera.camera_to_worlds[0,:3,:3])[:,2])
                        rgb=result['rgb'];a=result['accumulation']
                        ij=cc.long().cpu().numpy()
                        if plate is not None:rgb=rgb+(1-a)*torch.from_numpy(plate[ij[:,0],ij[:,1]]).to(rgb)
                        colors.append(rgb.float().cpu().numpy());depths.append(z.cpu().numpy())
                    rgb=np.concatenate(colors).reshape(y1-y0,x1-x0,3);z=np.concatenate(depths).reshape(y1-y0,x1-x0)
                    if mode=='arm':repeat.append(rgb)
                    else:dense.append(rgb);qvalues.append(z)
                delta=max(float(np.abs(repeat[i]-repeat[0]).max()) for i in [1,2])
                report['repeat_max_abs']=max(report['repeat_max_abs'],delta)
                base=geometry(root,c)
                image(out/f'cam{c:03d}_{name}_ARM_8192_16384.png',np.concatenate([repeat[0],*dense],axis=1))
                zbase=base['depth_z_q50'][y0:y1,x0:x1]
                row_result=dict(camera=c,region=name,box=box,repeat_max_abs=delta,
                      cache_rgb_max_abs=float(np.abs(repeat[0]-base['rgb'][y0:y1,x0:x1]).max()),
                      dense_rgb_mae=float(np.abs(dense[0]-dense[1]).mean()),
                      arm_dense_depth_median_abs=float(np.nanmedian(np.abs(zbase-qvalues[1]))),
                      dense_depth_median_abs=float(np.nanmedian(np.abs(qvalues[0]-qvalues[1]))))
                report['rows'].append(row_result);log(root,'dense_audit',**row_result)
    assert before==state_hashes(m)
    report['state_unchanged']=True
    # Numerical floor only; not a quality tolerance.
    report['lpips_tolerance']=1e-6
    write(root/'preflight/numerical_audit.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    audit(p.parse_args().root)
