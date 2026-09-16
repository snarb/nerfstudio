"""Independent count replay, exact splice audit, and every recovered component."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label,find_objects
from study_multiview_face_prior import read,save,sha
from recover_supported_front_surface import ROOT,CULLED,PARENT,FRAME
from review_measured_free_surface import VIEWS
from study_confidence_depth_prior import load_real
from review_full_block_transfer import ROOT as DEPTH_ROOT
from review_jaw_repair_transfer import panel


def count_near(points,rows,depths):
    count=np.zeros(len(points),int)
    for row,depth in zip(rows,depths):
        pose=np.array(row['transform_matrix']);p=(points-pose[:3,3])@pose[:3,:3];z=-p[:,2]
        uv=np.c_[row['fl_x']*p[:,0]/z+row['cx'],-row['fl_y']*p[:,1]/z+row['cy']].astype(np.float32)
        xy=np.rint(uv).astype(int);x,y=xy.T
        ok=(z>0)&(x>=0)&(x<1920)&(y>=0)&(y<1080)&np.isfinite(uv).all(1)
        idx=np.flatnonzero(ok);d=depth[y[idx],x[idx]]
        count[idx]+=(d>0)&np.isfinite(d)&(abs(d-z[idx])<=.0015)
    return count


def main():
    dest=ROOT/FRAME/'review';assert not dest.exists();rows,depths,receipt=load_real(DEPTH_ROOT,FRAME)
    records=[];bindings={}
    for view in VIEWS:
        folder=ROOT/FRAME/view;r=read(folder/'result.json');e=np.load(folder/'evidence.npz')
        assert r['depth_receipt']==receipt
        assert r['script_sha256']==sha(Path(__file__).with_name('recover_supported_front_surface.py'))
        for p,h in r['input_hashes'].items():assert sha(p)==h;bindings[p]=h
        for p,h in r['hashes'].items():assert sha(folder/p)==h;bindings[str(folder/p)]=h
        old=PARENT/'carved'/FRAME/'rgb'/view/'frames'/FRAME;raw=CULLED/FRAME/view/'frames'/FRAME
        images=[np.array(Image.open(p/'prediction_native.png')) for p in [old,raw,folder]]
        sources=[np.array(Image.open(p/'source_ids.png')) for p in [old,raw,folder]]
        depths_saved=[np.load(p/'target_depth.npz')['depth'] for p in [old,raw,folder]]
        old_near=count_near(e['original_points'],rows,depths);new_near=count_near(e['candidate_points'],rows,depths)
        np.testing.assert_array_equal(old_near,e['original_near']);np.testing.assert_array_equal(new_near,e['candidate_near'])
        x,y=e['candidate_xy'].T;selected=e['accepted'];mask=e['mask']
        expected=(old_near==0)&(new_near>=3)&e['original_backface']&e['behind']
        np.testing.assert_array_equal(selected,expected)
        expectedmask=np.zeros(mask.shape,bool);expectedmask[y[selected],x[selected]]=True
        np.testing.assert_array_equal(mask,expectedmask)
        for array in [images,sources,depths_saved]:
            np.testing.assert_array_equal(array[2][~mask],array[0][~mask])
            np.testing.assert_array_equal(array[2][mask],array[1][mask])
        assert (images[0][mask].max(1)==0).all() and (sources[0][mask]==255).all()
        assert (images[2][mask].max(1)>0).all() and (sources[2][mask]<62).all()
        assert not ((images[0].max(2)>0)&(images[2].max(2)==0)).any()
        portrait=[np.rot90(a) for a in images];pm=np.rot90(mask);lab,n=label(pm);components=[]
        for i,sl in enumerate(find_objects(lab),1):
            yy,xx=sl;box=(max(0,xx.start-22),max(0,yy.start-22),min(1080,xx.stop+22),min(1920,yy.stop+22))
            panels=portrait.copy();names=['baseline','global culling','guarded fallback']
            if view!='moving':
                p=DEPTH_ROOT/FRAME/'review'/view/'train_gt.png';panels.insert(0,np.array(Image.open(p)));names.insert(0,'actual train GT');bindings[str(p)]=sha(p)
            path=dest/view/f'recovered_{i:02d}.png';panel(path,panels,names,box)
            components.append(dict(path=str(path),pixels=int((lab==i).sum()),box=box))
        records.append(dict(view=view,recovered_pixels=int(mask.sum()),components=components,
            original_colored_pixels_exact=True,all_other_depth_rgb_source_exact=True,
            original_support_replayed=old_near.tolist(),new_support_replayed=new_near.tolist()))
        bindings[str(folder/'result.json')]=sha(folder/'result.json')
    dest.mkdir(parents=True,exist_ok=True)
    save(dest/'result.json',dict(records=records,input_hashes=bindings,
        images={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*.png')},
        independent_projection_near_count_replay=True,script_sha256=sha(__file__),
        visual_status='pending',production_promoted=False))
    print([(r['view'],r['recovered_pixels'],len(r['components'])) for r in records],flush=True)


if __name__=='__main__':main()
