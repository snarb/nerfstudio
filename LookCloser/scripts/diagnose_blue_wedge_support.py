"""Actual ray-hit depth support in a visually identified blue-wedge core.

Post-hoc only: neither this rectangle nor its RGB enters mesh/source selection.
Evaluates exact rendered surface points, not nearby observed-point corroboration.
"""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from study_multiview_face_prior import read,save,sha
from study_query_support_quorum import ROOT,FRAME
from review_full_block_transfer import ROOT as DEPTH_ROOT
from study_confidence_depth_prior import load_real,project_integer
from diagnose_jaw_measured_depth import observed_at
from prune_measured_free_surface import near_tap_evidence
from admit_mhr_local_patch_depth import Scene2
from bake_joint_temporal_mesh import camera_depth
from combine_query_quorum_texture_guard import CORE_POLYGON
from diagnose_lipstick_fin_depth import CAMERA,BOX,POLYGON as OLD_POLYGON


def main():
    root=ROOT/FRAME;out=root/'blue_wedge_support';assert not out.exists()
    folder=root/'rgb'/CAMERA/'frames'/FRAME; r=read(folder/'result.json');camera=r['camera']
    assert sha(r['mesh_path'])==r['mesh_sha256']
    mesh=o3d.io.read_triangle_mesh(r['mesh_path']);v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles)
    d,ids,bary=camera_depth(Scene2(v,t),camera)
    rendered=np.load(folder/'target_depth.npz')['depth'];np.testing.assert_array_equal(np.where(np.isfinite(d),d,0),rendered)
    mask=Image.new('L',(1080,1920));ImageDraw.Draw(mask).polygon(CORE_POLYGON,fill=1)
    take=np.rot90(np.asarray(mask,bool),-1)&(rendered>0)
    weights=np.c_[1-bary[take].sum(1),bary[take]];points=(v[t[ids[take]]]*weights[:,:,None]).sum(1)
    selected=np.asarray(Image.open(folder/'source_ids.png'))[take]
    rows,depths,receipt=load_real(DEPTH_ROOT,FRAME);assert [x['physical_camera'] for x in rows]==r['source_cameras']
    near=[];delta=[];available=[]
    for row,depth in zip(rows,depths):
        uv,z=project_integer(row,points);near.append(near_tap_evidence(depth,uv,z,tolerance=.0015,radius=0))
        xy,z,obs,ok=observed_at(points,row,depth);delta.append(np.where(ok,obs-z,np.nan));available.append(ok)
    near=np.asarray(near);delta=np.asarray(delta);available=np.asarray(available)
    valid_source=selected<62;chosen_delta=np.full(len(points),np.nan)
    chosen_delta[valid_source]=delta[selected[valid_source],np.flatnonzero(valid_source)]
    summary={}
    for ci in np.unique(selected[valid_source]):
        values=chosen_delta[selected==ci];finite=values[np.isfinite(values)]
        summary[rows[ci]['physical_camera']]=dict(pixels=int((selected==ci).sum()),valid_depths=len(finite),
            depth_delta_min=float(finite.min()) if len(finite) else None,
            depth_delta_median=float(np.median(finite)) if len(finite) else None,
            depth_delta_max=float(finite.max()) if len(finite) else None)
    oldmask=Image.new('L',(1080,1920));ImageDraw.Draw(oldmask).polygon(OLD_POLYGON,fill=1)
    overlap=int((np.asarray(mask,bool)&np.asarray(oldmask,bool)).sum())
    out.mkdir();np.savez_compressed(out/'evidence.npz',points=points,near_by_camera=near,
        available=available,depth_deltas=delta,selected_sources=selected,selected_depth_delta=chosen_delta,
        native_pixel_mask=take,face_ids=ids[take])
    image=Image.open(folder/'frame.png').convert('RGB');draw=ImageDraw.Draw(image)
    draw.polygon(OLD_POLYGON,outline='orange',width=1);draw.polygon(CORE_POLYGON,outline='cyan',width=1)
    image.crop(BOX).save(out/'regions.png')
    sources=[folder/'result.json',folder/'complete.json',folder/'source_ids.png',folder/'target_depth.npz',Path(__file__),Path(r['mesh_path'])]
    n=near.sum(0)
    result=dict(core_pixels=len(points),core_polygon=CORE_POLYGON,overlap_with_old_polygon=overlap,
        actual_query_near_views=dict(min=int(n.min()),median=float(np.median(n)),max=int(n.max())),
        chosen_source_depth_deltas=summary,depth_receipt=receipt,posthoc_only=True,production_changed=False,
        input_hashes={str(p):sha(p) for p in sources},outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    save(out/'result.json',result);print({k:result[k] for k in ['core_pixels','overlap_with_old_polygon','actual_query_near_views','chosen_source_depth_deltas']},flush=True)


if __name__=='__main__':main()
