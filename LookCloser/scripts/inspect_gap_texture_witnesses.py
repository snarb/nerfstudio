"""Actual train RGB/mask witnesses and continuous ray visibility for gap points."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from study_multiview_face_prior import read,save,sha
from diagnose_gap_texture_admission import ROOT
from study_train_gap_subfaces import ROOT as STUDY,FRAME
from joint_temporal_texture import ROOT as COLOR,cameras,exr,display,project
from calibrated_depth_witness import response_gains
from diffusion_mesh_repair import scene_for
from study_lipstick_instance_masks import portrait_xy


def main():
    out=ROOT/'witnesses';assert not out.exists()
    record=read(ROOT/'result.json');assert sha(ROOT/'evidence.npz')==record['evidence_sha256']
    e=np.load(ROOT/'evidence.npz');points=e['points'];rows,_,_=cameras(FRAME)
    q=read(STUDY/'carved'/FRAME/'rgb/moving/request.json');r=read(STUDY/'carved'/FRAME/'rgb/moving/frames'/FRAME/'result.json')
    mesh=o3d.io.read_triangle_mesh(r['mesh_path']);assert sha(r['mesh_path'])==r['mesh_sha256']
    scene=scene_for(np.asarray(mesh.vertices,np.float32),np.asarray(mesh.triangles,np.uint32))
    direct=[]
    for row in rows:
        center=np.array(row['transform_matrix'],np.float32)[:3,3]
        rays=np.c_[np.broadcast_to(center,points.shape),points-center].astype(np.float32)
        direct.append(scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy())
    direct=np.stack(direct);visible=np.isfinite(direct)&(abs(direct-1)<1e-5)
    chosen=sorted(set(np.flatnonzero(e['raw_final'].any(1)).tolist()+
        [i for i,row in enumerate(rows) if row['physical_camera'].startswith(('H004_C','K004_B'))]))
    spec=record['source_mask_spec'];maskroot=Path(spec['root'])
    masks=np.load(maskroot/'masks.npz')['masks'];names=read(maskroot/'cameras.json')
    gains=response_gains(np.load(COLOR/'parameters.npz')['log_gain']);gain=read(COLOR/'exposure.json')['fixed_exposure_gain']
    expected={x['physical_camera']:x['sha256'] for x in q['source_rows'][0]['source_images']}
    bindings={str(ROOT/'result.json'):sha(ROOT/'result.json'),str(ROOT/'evidence.npz'):sha(ROOT/'evidence.npz')}
    out.mkdir();records=[]
    for i in chosen:
        row=rows[i];name=row['physical_camera'];path=Path(row['file_path']);assert sha(path)==expected[name];bindings[str(path)]=sha(path)
        rgb=np.rint(display(exr(path)*gains[i],gain)*255).clip(0,255).astype(np.uint8)
        portrait=np.rot90(rgb).copy();mask=np.rot90(masks[names.index(name)])>0
        uv,z=project(points,[row]);xy=portrait_xy(uv[0],1920)
        low=np.maximum(np.floor(xy.min(0)).astype(int)-30,[0,0]);hi=np.minimum(np.ceil(xy.max(0)).astype(int)+31,[1080,1920]);box=(*low,*hi)
        plain=Image.fromarray(portrait).crop(box);marked=plain.copy();overlay=portrait.copy()
        overlay[~mask]=(overlay[~mask]*.35+[120,0,120]).clip(0,255).astype(np.uint8)
        overlay=Image.fromarray(overlay).crop(box)
        for im in [marked,overlay]:
            draw=ImageDraw.Draw(im)
            for j,p in enumerate(xy-low):
                x,y=map(float,p);draw.ellipse((x-2,y-2,x+2,y+2),outline='cyan')
                if j in [0,4]:draw.text((x+3,y-7),str(j),fill='cyan')
        canvas=Image.new('RGB',(plain.width*3,plain.height+38))
        for k,im in enumerate([plain,marked,overlay]):canvas.paste(im,(k*plain.width,38))
        draw=ImageDraw.Draw(canvas);draw.text((2,2),name+' / RGB, projections, magenta=masked out',fill='white')
        draw.text((2,17),'direct t: '+', '.join(f'{x:.4f}' for x in direct[i]),fill='white')
        canvas.save(out/(name+'.png'))
        records.append(dict(camera=name,source_index=i,crop=list(map(int,box)),
            direct_t=direct[i].tolist(),direct_visible=visible[i].tolist(),
            raw_footprint_visible=e['raw_final'][i].tolist(),masked_footprint_visible=e['masked_final'][i].tolist()))
    for p in [COLOR/'parameters.npz',COLOR/'exposure.json',maskroot/'masks.npz',maskroot/'cameras.json',Path(__file__)]:bindings[str(p)]=sha(p)
    np.savez_compressed(out/'direct_rays.npz',t_hit=direct,visible=visible)
    save(out/'result.json',dict(records=records,direct_visible_camera_counts=visible.sum(0).tolist(),
        input_hashes=bindings,outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        uses_real_train_rgb=True,production_changed=False,visual_status='pending'))
    print('directly visible views',visible.sum(0).tolist(),flush=True)


if __name__=='__main__':main()
