"""Verify same geometry/cameras and inspect exact native RGB footprint changes."""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
from study_native_texture_footprint import OUT,BASE
from review_jaw_repair_transfer import verified_image,panel


def run():
    frame='001193';dest=OUT/'review';dest.mkdir(exist_ok=False);records=[]
    for view in ['moving','F004_E005_1210FP','heldout']:
        baseline=BASE/'heldout' if view=='heldout' else BASE/'rgb'/frame/view/'repaired'
        roots=[baseline,OUT/view];images=[];depth=[];metadata=[]
        for root in roots:
            im,r=verified_image(root,frame);images.append(im);metadata.append(r)
            depth.append(np.load(root/'frames'/frame/'target_depth.npz')['depth'])
        for k in ['mesh_sha256','camera','source_cameras','fixed_exposure']:
            assert metadata[0][k]==metadata[1][k]
        np.testing.assert_array_equal(depth[0],depth[1])
        record=dict(view=view,same_mesh_camera_sources_exposure=True,depth_arrays_exact=True,
            changed_pixels=int(np.any(images[0]!=images[1],axis=2).sum()),
            input_hashes={str(p/'frames'/frame/'frame.png'):sha(p/'frames'/frame/'frame.png') for p in roots})
        labels=['previous footprint','zero-weight footprint']
        if view=='F004_E005_1210FP':
            gtpath=Path('/mnt/data/dec5_jaw_repair_transfer/review')/frame/(view+'_gt.png');gt=np.array(Image.open(gtpath))
            poly=[[610,1090],[655,1120],[693,1134],[710,1146],[719,1152],[711,1215],[705,1230],[690,1200],[668,1175],[635,1145],[610,1125]]
            mask=np.zeros(gt.shape[:2],np.uint8);cv2.fillPoly(mask,[np.array(poly,np.int32)],1);mask=mask.astype(bool)
            d=np.rot90(depth[0]);black=[(im.max(2)==0)&mask for im in images]
            record.update(gt_sha256=sha(gtpath),polygon=poly,depth_misses=int((mask&(d==0)).sum()),
                black_rgb=[int(b.sum()) for b in black],black_with_geometry=[int((b&(d>0)).sum()) for b in black])
            previously_missing=black[0]&(d>0);new_ids=np.rot90(np.array(Image.open(OUT/view/'frames'/frame/'source_ids.png')))
            record['recovered_source_ids']=new_ids[previously_missing].tolist()
            record['recovered_vs_GT_rgb_absolute_error_mean_8bit']=float(np.abs(images[1].astype(float)-gt)[previously_missing].mean())
            images=[gt,*images];labels=['real train GT',*labels]
        panel(dest/(view+'_head.png'),images,labels,(170,450,970,1350))
        panel(dest/(view+'_detail.png'),images,labels,(570,1070,780,1240) if view=='F004_E005_1210FP' else (370,1080,670,1320))
        records.append(record)
    atomic_json(dest/'result.json',dict(records=records,script_sha256=sha(__file__),visual_status='pending',
        scope='source-footprint/coverage diagnostics, not full-frame image quality scores'))
    print(records,flush=True)


if __name__=='__main__':run()
