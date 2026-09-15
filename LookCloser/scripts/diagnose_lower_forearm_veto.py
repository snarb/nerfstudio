"""Inspect real RGB at corroborated depth vetoes of the lower-forearm prior."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from calibrated_depth_witness import load_images
from study_confidence_depth_prior import support,unproject
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from study_foundation_lower_forearm import ROOT,FRAME


def run():
    import study_forearm_plane_transfer_v3 as real
    real.configure();rows,depths,hashes=real.v2.v1.load_real(FRAME)
    request=read(ROOT/'foreground/request.json');q=np.load(ROOT/'foreground/proposal.npz')
    old=o3d.io.read_triangle_mesh(request['source_mesh']);v=np.concatenate((np.asarray(old.vertices),q['added_vertices']))
    t=np.concatenate((np.asarray(old.triangles),q['added_triangles']+len(old.vertices)))
    scene=scene_for(v,t);images,_,receipt=load_images(FRAME);records=[];dest=ROOT/'veto_diagnosis';dest.mkdir(exist_ok=False)
    for name in ['B004_E005_1210VE','E004_E005_1210WX','F004_E005_1210FP']:
        index=next(i for i,r in enumerate(rows) if r['physical_camera']==name);row=rows[index]
        actual=dict(row);actual['cx']+=.5;actual['cy']+=.5
        d,ids,_=camera_depth(scene,actual);observed=depths[index]
        y,x=np.nonzero(np.isfinite(d)&(ids>=len(old.triangles))&(ids<len(t))&(observed>0)&(observed>d+.003))
        votes,_=support(unproject(row,x,y,observed[y,x]),row,rows,depths);trusted=votes>=3;x,y=x[trusted],y[trusted]
        im=np.rot90(images[name]).copy();px=y;py=1919-x;colors=images[name][y,x]
        warm=colors[:,0].astype(float)-colors[:,2].astype(float)>8
        # Evenly spaced native vertical quantiles, independent of RGB diagnosis.
        order=np.argsort(py);chosen=order[np.linspace(0,len(order)-1,min(5,len(order))).astype(int)]
        canvas=Image.new('RGB',(5*220,260));draw=ImageDraw.Draw(canvas)
        samples=[]
        for col,ci in enumerate(chosen):
            u,w=int(px[ci]),int(py[ci]);crop=Image.fromarray(im).crop((u-110,w-110,u+110,w+110))
            canvas.paste(crop,(col*220,40));draw.ellipse((col*220+106,146,col*220+114,154),outline='red',width=2)
            delta=float(observed[y[ci],x[ci]]-d[y[ci],x[ci]])
            draw.text((col*220+3,4),name,fill='white');draw.text((col*220+3,20),f'delta={delta:.5f} RGB={colors[ci].tolist()}',fill='white')
            samples.append(dict(native_xy=[int(x[ci]),int(y[ci])],portrait_xy=[u,w],
                proposed_depth=float(d[y[ci],x[ci]]),observed_depth=float(observed[y[ci],x[ci]]),
                rgb=colors[ci].tolist(),warm=bool(warm[ci]),triangle=int(ids[y[ci],x[ci]])))
        path=dest/(name+'.png');canvas.save(path)
        records.append(dict(camera=name,trusted_veto_pixels=len(x),warm_color_pixels=int(warm.sum()),
            warm_color_fraction=float(warm.mean()),samples=samples,panel=str(path),panel_sha256=sha(path)))
        print(records[-1],flush=True)
    atomic_json(dest/'result.json',dict(records=records,script_sha256=sha(__file__),rgb_receipt=receipt,
        source_depth_hashes=hashes,raw_proposal_sha256=sha(ROOT/'foreground/proposal.npz'),
        warm_color_is_not_segmentation_or_correctness_label=True,diagnosis_only=True,visual_status='pending'))


if __name__=='__main__':run()
