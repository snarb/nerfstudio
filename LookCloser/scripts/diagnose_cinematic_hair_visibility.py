"""Replay source visibility at frozen hair diagnostic points, without rendering."""
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
from PIL import Image, ImageDraw
from joint_temporal_texture import cameras, read, sha, atomic_json, exr, display, ROOT as COLOR
from bake_joint_temporal_mesh import camera_depth
from diffusion_mesh_repair import scene_for
from native_texture_footprint import snap_centers, sample_native, relevant_tap
from diagnose_cinematic_hair_rim import BASE, OUT


def run():
    torch.set_num_threads(2)
    frame = '001083'; root = OUT/frame
    dest = root/'visibility'; dest.mkdir(exist_ok=False)
    result = read(root/'result.json')
    assert sha(root/'projections.npz') == result['projection_sha256']
    assert sha(result['mesh_path']) == result['mesh_sha256']
    q = read(BASE/'request.json')
    row = next(r for r in q['inventory'] if r['frame_id'] == frame)
    rows, _, _ = cameras(frame)
    mroot = Path(row['source_masks']['root'])
    assert sha(mroot/'masks.npz') == result['masks_sha256']
    masks = np.load(mroot/'masks.npz')['masks']; names = read(mroot/'cameras.json')
    mesh = o3d.io.read_triangle_mesh(result['mesh_path'])
    scene = scene_for(np.asarray(mesh.vertices,np.float32), np.asarray(mesh.triangles,np.uint32))
    samples = [s for s in result['samples'] if s['surface']]
    a = np.load(root/'projections.npz'); uv = snap_centers(torch.from_numpy(a['uv'][:,None])); z = torch.from_numpy(a['depth'])
    visible = []; tap_depths = []; interpolated = []
    for ci, row in enumerate(rows):
        d = camera_depth(scene,row)[0]
        d[masks[names.index(row['physical_camera'])]==0] = np.inf
        image = torch.from_numpy(np.where(np.isfinite(d),d,0)[None,None])
        point_uv = uv[ci:ci+1]; zz = z[ci:ci+1]
        sd = sample_native(image,point_uv)[:,0,0]
        valid = (zz>0)&(sd>0)&((sd-zz).abs()<.0015*zz)
        valid &= (point_uv[:,0,:,0]>2)&(point_uv[:,0,:,0]<1917)&(point_uv[:,0,:,1]>2)&(point_uv[:,0,:,1]<1077)
        taps = []
        for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
            tap = sample_native(image,point_uv.floor()+point_uv.new_tensor([dx,dy]))[:,0,0]
            valid &= ((tap>0)&((tap-zz).abs()<.003*zz))|~relevant_tap(point_uv,dx,dy)
            taps.append(tap[0].numpy())
        visible.append(valid[0].numpy()); tap_depths.append(np.stack(taps,-1)); interpolated.append(sd[0].numpy())
    visible = np.array(visible)
    gains = np.load(COLOR/'parameters.npz')['log_gain']; gains = np.exp(gains-gains.mean(0,keepdims=True))
    exposure = read(COLOR/'exposure.json')['fixed_exposure_gain']
    output = []; rgb_hashes = {}; cache = {}
    for si, sample in enumerate(samples):
        chosen = sample['source_id']; assert visible[chosen,si], ('Chosen source fails visibility',si)
        eligible = np.flatnonzero(visible[:,si])
        order = eligible[np.argsort(a['original_signed_distance'][eligible,si])[::-1]]
        selected = list(dict.fromkeys([chosen,*order[:3].tolist()]))
        panel = Image.new('RGB',(192*len(selected),245),(25,25,25));draw=ImageDraw.Draw(panel)
        details = []
        for wi, ci in enumerate(selected):
            source = rows[ci]['file_path']
            if ci not in cache:
                cache[ci] = np.rint(display(exr(source)*gains[ci],exposure)*255).clip(0,255).astype(np.uint8)
                rgb_hashes[source] = sha(source)
            u,v = uv[ci,0,si].numpy(); ix,iy = np.rint([u,v]).astype(int)
            crop = np.rot90(np.asarray(Image.fromarray(cache[ci]).crop((ix-90,iy-90,ix+90,iy+90))))
            offset=wi*192;panel.paste(Image.fromarray(crop),(offset+6,45))
            dist=float(a['original_signed_distance'][ci,si])
            draw.text((offset+5,5),rows[ci]['physical_camera'],fill='white')
            draw.text((offset+5,19),f'{"CHOSEN" if ci==chosen else "visible"} d={dist:.1f}',fill='white')
            px,py=offset+96+v-iy,134-u+ix
            draw.ellipse((px-3,py-3,px+3,py+3),outline='red')
            details.append(dict(camera=rows[ci]['physical_camera'],original_mask_distance=dist))
        path=dest/f'point_{si:02d}.png';panel.save(path)
        output.append(dict(portrait_xy=sample['portrait_xy'],visible_sources=len(eligible),
            chosen_camera=rows[chosen]['physical_camera'],witnesses=details,panel=path.name,panel_sha256=sha(path)))
    np.savez_compressed(dest/'evidence.npz',visible=visible,interpolated=np.array(interpolated),tap_depths=np.array(tap_depths))
    atomic_json(dest/'result.json',dict(parent_result_sha256=sha(root/'result.json'),script_sha256=sha(__file__),
        samples=output,source_rgb_hashes=rgb_hashes,geometry_changed=False,source_selection_changed=False,
        all_selected_sources_pass_exact_visibility=True,evidence_sha256=sha(dest/'evidence.npz'),visual_status='pending'))
    print(output,flush=True)


if __name__ == '__main__':run()
