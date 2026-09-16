"""Post-hoc diagnosis of remaining ray hits and every new black component."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from scipy.ndimage import label,find_objects,distance_transform_edt
from study_multiview_face_prior import read,save,sha
from study_train_gap_subfaces import ROOT,FRAME,COARSE,classify
from study_train_gap_carving import MASKS,HAND
from review_measured_free_surface import VIEWS
from review_jaw_repair_transfer import panel
from bake_joint_temporal_mesh import camera_depth
from admit_mhr_local_patch_depth import Scene2


def main():
    root=ROOT/FRAME;dest=root/'remaining';assert not dest.exists();dest.mkdir()
    q=read(root/'request.json');e=np.load(root/'evidence.npz');sources={}
    records=[]
    for view in VIEWS:
        old=COARSE/FRAME/'rgb'/view/'frames'/FRAME
        new=ROOT/'carved'/FRAME/'rgb'/view/'frames'/FRAME
        a=np.array(Image.open(old/'frame.png'));b=np.array(Image.open(new/'frame.png'))
        black=(a.max(2)>0)&(b.max(2)==0);lab,n=label(black);patches=[];components=[]
        for i,sl in enumerate(find_objects(lab),1):
            yy,xx=sl;box=(max(0,xx.start-15),max(0,yy.start-15),min(1080,xx.stop+15),min(1920,yy.stop+15))
            count=int((lab==i).sum());p=dest/f'{view}_{i:02d}.png'
            panel(p,[a,b],[f'coarse / {count}px','refined carved'],box)
            patches.append(Image.open(p).copy());components.append(dict(pixels=count,box=box,path=str(p)))
        width=max(800,max((p.width for p in patches),default=1));x=y=h=0;locations=[]
        for p in patches:
            if x+p.width>width:y+=h+8;x=h=0
            locations.append((x,y));x+=p.width+8;h=max(h,p.height)
        sheet=Image.new('RGB',(width,max(1,y+h)),(35,35,35))
        for p,xy in zip(patches,locations):sheet.paste(p,xy)
        sheet.save(dest/(view+'_new_black_native.png'))
        record=dict(view=view,components=components,new_black=int(black.sum()))
        if view!='moving':
            s=next(s for s in q['views'] if s['camera']==view);x0,y0,x1,y1=s['crop'];bc=black[y0:y1,x0:x1]
            record['inside_tube_hand']=[];record['inside_eroded_tube_hand']=[]
            for folder,run in [(MASKS,'sam_v2'),(HAND,'sam_v1')]:
                rv=read(folder/'mask_review.json');p=folder/run/view/f'mask_{rv["selected"][view]}.png'
                m=np.array(Image.open(p))>0;record['inside_tube_hand'].append(int((bc&m).sum()))
                record['inside_eroded_tube_hand'].append(int((bc&(distance_transform_edt(m)>2)).sum()));sources[str(p)]=sha(p)
        for folder in [old,new]:
            for name in ['frame.png','target_depth.npz','result.json','complete.json']:sources[str(folder/name)]=sha(folder/name)
        records.append(record)
    view='K004_B005_1210DS';new=ROOT/'carved'/FRAME/'rgb'/view/'frames'/FRAME
    result=read(new/'result.json');mesh=o3d.io.read_triangle_mesh(result['mesh_path'])
    assert sha(result['mesh_path'])==result['mesh_sha256']
    v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles)
    d,ids,bary=camera_depth(Scene2(v,t),result['camera'])
    actual=np.load(new/'target_depth.npz')['depth'];np.testing.assert_array_equal(np.where(np.isfinite(d),d,0),actual)
    polygon=Image.new('1',(1080,1920));ImageDraw.Draw(polygon).polygon([(156,1230),(164,1230),(164,1248),(156,1248)],fill=1)
    core=np.rot90(np.array(polygon,bool),-1);take=core&(actual>0)
    weights=np.c_[1-bary[take].sum(1),bary[take]];points=(v[t[ids[take]]]*weights[:,:,None]).sum(1)
    negative,positive=classify(q,points,np.load(COARSE/FRAME/'positive_masks.npz'))
    eligible=(negative.sum(0)>=3)&~positive.any(0)
    olddepth=np.load(COARSE/FRAME/'rgb'/view/'frames'/FRAME/'target_depth.npz')['depth']
    change=actual[take]-olddepth[take]
    reasons=dict(core_pixels=int(core.sum()),remaining_hits=int(take.sum()),point_rule_eligible=int(eligible.sum()),
        positive_protected=int(positive.any(0).sum()),fewer_than_three_negative=int((negative.sum(0)<3).sum()),
        depth_delta_min=float(change.min()) if len(change) else None,depth_delta_max=float(change.max()) if len(change) else None)
    np.savez_compressed(dest/'core.npz',points=points,negative=negative,positive=positive,eligible=eligible,depth_delta=change)
    sources[str(root/'request.json')]=sha(root/'request.json');sources[str(Path(__file__))]=sha(__file__)
    save(dest/'result.json',dict(records=records,core=reasons,input_hashes=sources,
        outputs={p.name:sha(p) for p in dest.iterdir() if p.is_file()},posthoc_only=True,production_changed=False))
    print(reasons,flush=True);print([{k:v for k,v in r.items() if k!='components'} for r in records],flush=True)


if __name__=='__main__':main()
