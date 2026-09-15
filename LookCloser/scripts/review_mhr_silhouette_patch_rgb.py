"""Native RGB panels and observed rendering side effects, not held-out metrics."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image,ImageDraw
from admit_mhr_silhouette_patch import OUT
from study_multiview_face_prior import read,save,sha
from study_mhr_local_head_prior import RGB,FRAME


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--views',nargs='+',required=True);args=parser.parse_args()
    dest=OUT/'rgb_review';dest.mkdir(exist_ok=True)
    spots=Path('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')
    box=next(r['bbox_inclusive'] for r in read(spots)['selected_components'] if r['frame_id']==FRAME)
    predictions={r['camera']:r for r in read(RGB/'inference.json')['records'] if r['frame']==FRAME and r['detected']==1}
    for view in args.views:
        receipt=dest/(view+'.json');assert not receipt.exists()
        if view=='old_moving':x0,y0,x1,y1=box;crop=(max(0,x0-90),max(0,y0-150),min(1080,x1+91),min(1920,y1+130))
        else:
            name=next(n for n in predictions if n.startswith(view));x0,y0,x1,y1=predictions[name]['native_review_box'];crop=(x0,y0,x1,min(1550,y1+200))
        frames={};bindings={str(spots):sha(spots),str(RGB/'inference.json'):sha(RGB/'inference.json')}
        for variant in ['baseline','strict','interpolated']:
            root=OUT/'rgb'/view/variant;folder=root/'frames'/FRAME;c=read(folder/'complete.json');assert c['request_sha256']==sha(root/'request.json')
            for name,digest in c['hashes'].items():assert sha(folder/name)==digest
            image=np.array(Image.open(folder/'frame.png'));depth=np.rot90(np.load(folder/'target_depth.npz')['depth']);source=np.rot90(np.array(Image.open(folder/'source_ids.png')))
            frames[variant]=(image,depth,source);bindings[str(folder/'complete.json')]=sha(folder/'complete.json');bindings[str(root/'request.json')]=sha(root/'request.json')
        w,h=crop[2]-crop[0],crop[3]-crop[1];panel=Image.new('RGB',(w*3,h+24));draw=ImageDraw.Draw(panel)
        stats=[];base,bd,bs=frames['baseline'];bh=bd>0
        for index,(variant,(im,d,s)) in enumerate(frames.items()):
            panel.paste(Image.fromarray(im).crop(crop),(index*w,24));draw.text((index*w+2,4),variant,fill='white')
            new=(d>0)&~bh;common=(d>0)&bh;difference=np.zeros(d.shape);difference[common]=abs(d[common]-bd[common])
            color_changed=np.any(im!=base,axis=2)
            record=dict(variant=variant,new_geometry_pixels_full=int(new.sum()),new_geometry_pixels_with_rgb=int((new&(im.max(2)>0)).sum()),
                common_depth_changed_pixels=int((difference>1e-6).sum()),maximum_common_depth_change=float(difference.max()),
                source_label_changes_at_stable_geometry=int((common&(difference<=1e-6)&(s!=bs)).sum()),rgb_changed_pixels_full=int(color_changed.sum()))
            if view=='old_moving':
                bx0,by0,bx1,by1=box;sl=np.s_[by0:by1+1,bx0:bx1+1];missing=~bh[sl]
                record.update(fixed_original_missing=int(missing.sum()),fixed_remaining_missing=int((missing&~(d[sl]>0)).sum()),
                    fixed_new_geometry_with_rgb=int((missing&(d[sl]>0)&(im[sl].max(2)>0)).sum()),
                    fixed_new_source_fallback=int((missing&(d[sl]>0)&(s[sl]==255)).sum()))
            stats.append(record)
        path=dest/(view+'_native.png');panel.save(path)
        save(receipt,dict(statistics=stats,native_crop=list(crop),panel_path=str(path),panel_sha256=sha(path),
            input_hashes=bindings,script_sha256=sha(__file__),same_camera_calibration_profiles=True,
            cpu_only=True,heldout_metrics_not_claimed=True,production_accepted=False))
        print(view,stats,flush=True)


if __name__=='__main__':main()
