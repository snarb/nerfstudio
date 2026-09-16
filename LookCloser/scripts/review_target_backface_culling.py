"""Matched backface-culling review with source labels and unchanged mesh checks."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import label,find_objects
from study_multiview_face_prior import read,save,sha
from study_target_backface_culling import ROOT,PARENT,FRAME
from review_measured_free_surface import VIEWS
from review_subface_free_space import BOXES,HEADS
from review_jaw_repair_transfer import verified_image,panel
from bake_joint_temporal_mesh import camera_depth
from diffusion_mesh_repair import scene_for


def main():
    dest=ROOT/FRAME/'review';assert not dest.exists();records=[];bindings={}
    for view in VIEWS:
        old=PARENT/'carved'/FRAME/'rgb'/view;new=ROOT/FRAME/view
        a,ar=verified_image(old,FRAME);b,br=verified_image(new,FRAME)
        aq,bq=read(old/'request.json'),read(new/'request.json')
        for key in ['camera','source_cameras','mesh_sha256','fixed_exposure']:assert ar[key]==br[key]
        for key in ['recipe','profiles_sha256','exposure_sha256','calibration_sha256']:assert aq[key]==bq[key]
        assert bq['target_culling_script_sha256']==sha(Path(__file__).with_name('study_target_backface_culling.py'))
        labs=[np.load(p/'frames'/FRAME/'face_source_labels.npy') for p in [old,new]]
        np.testing.assert_array_equal(*labs)
        d0,d1=[np.load(p/'frames'/FRAME/'target_depth.npz')['depth'] for p in [old,new]]
        assert not ((d0==0)&(d1>0)).any()
        assert not ((d0>0)&(d1>0)&(d1<d0-1e-6)).any()
        mesh=o3d.io.read_triangle_mesh(ar['mesh_path']);v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles,np.uint32)
        d,ids,bary=camera_depth(scene_for(v,t),ar['camera']);hit=np.isfinite(d)
        np.testing.assert_array_equal(np.where(hit,d,0),d0)
        tv=v[t];normal=np.cross(tv[:,1]-tv[:,0],tv[:,2]-tv[:,0]);center=np.array(ar['camera']['transform_matrix'])[:3,3]
        facing=np.sum(normal*(center-tv[:,0]),axis=1)>0
        retain=np.load(new/'target_retained_faces.npy');np.testing.assert_array_equal(retain,np.flatnonzero(facing))
        culled=np.zeros(d0.shape,bool);culled[hit]=~facing[ids[hit]]
        unchanged=hit&~culled
        np.testing.assert_allclose(d0[unchanged],d1[unchanged],rtol=0,atol=1e-6)
        raw_a=np.rot90(a,-1);raw_b=np.rot90(b,-1);outside_changes=int((np.any(raw_a!=raw_b,2)&unchanged).sum())
        folder=dest/view;panel(folder/'lipstick_native.png',[a,b],['double-sided','target backface culling'],BOXES[view])
        panel(folder/'head_native.png',[a,b],['double-sided','target backface culling'],HEADS[view])
        bad=(a.max(2)>0)&(b.max(2)==0);lab,n=label(bad);patches=[];components=[]
        for i,sl in enumerate(find_objects(lab),1):
            yy,xx=sl;box=(max(0,xx.start-12),max(0,yy.start-12),min(1080,xx.stop+12),min(1920,yy.stop+12))
            count=int((lab==i).sum());p=folder/f'black_{i:03d}.png';panel(p,[a,b],[f'before / {count}px','after'],box)
            patches.append(Image.open(p).copy());components.append(dict(pixels=count,box=box,path=str(p)))
        width=max(1000,max((p.width for p in patches),default=1));x=y=h=0;positions=[]
        for p in patches:
            if x+p.width>width:y+=h+8;x=h=0
            positions.append((x,y));x+=p.width+8;h=max(h,p.height)
        sheet=Image.new('RGB',(width,max(1,y+h)),(35,35,35))
        for p,xy in zip(patches,positions):sheet.paste(p,xy)
        sheet.save(folder/'new_black_native.png')
        for p in [old,new]:
            bindings[str(p/'request.json')]=sha(p/'request.json');bindings[str(p/'frames'/FRAME/'complete.json')]=sha(p/'frames'/FRAME/'complete.json')
            for name,h in read(p/'frames'/FRAME/'complete.json')['hashes'].items():bindings[str(p/'frames'/FRAME/name)]=h
        item=dict(view=view,original_backface_hit_pixels=int(culled.sum()),changed_rgb=int(np.any(a!=b,2).sum()),
            newly_black=int(bad.sum()),newly_colored=int(((a.max(2)==0)&(b.max(2)>0)).sum()),
            lost_depth=int(((d0>0)&(d1==0)).sum()),rgb_changes_outside_culled_first_hits=outside_changes,
            source_face_labels_exact=True,mesh_exact=True,components=components)
        if view=='moving':
            e=np.load('/mnt/data/dec5_gap_texture_admission/evidence.npz');x,y=e['landscape_xy'].T
            item['five_diagnostic_rgb']=raw_b[y,x].tolist();item['five_diagnostic_depth']=d1[y,x].tolist()
        records.append(item)
    save(dest/'result.json',dict(records=records,input_hashes=bindings,
        images={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*.png')},visual_status='pending',
        production_changed=False,counts_not_quality_metrics=True,script_sha256=sha(__file__)))
    print([{k:v for k,v in r.items() if k!='components'} for r in records],flush=True)


if __name__=='__main__':main()
