"""Opt-in real-train silhouette carving and RGB foreground eligibility.

Semantic masks are fallible silhouette evidence, NOT measured depth. A retained
mesh can still be wrong; no synthetic RGB or cross-camera RGB blend is used.
The original renderer, datasets and meshes remain unchanged.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
from joint_temporal_texture import ROOT, read, sha, atomic_json, cameras, exr, display, project

MODEL = None
SETTINGS = dict(source_count=62, native_dilation=4, outside_votes=12, grabcut_iterations=3,
                semantic_seed_threshold=.5, mask_is_independent_depth=False)


def carve_candidates(vertices, triangles, rows, masks, votes=6):
    uv, z = project(vertices, rows); outside = np.zeros(len(vertices), np.uint16)
    for i, mask in enumerate(masks):
        xy = np.rint(uv[i]).astype(np.int32)
        valid = (z[i] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
        hit = np.ones(len(vertices), bool)
        hit[valid] = mask[xy[valid, 1], xy[valid, 0]] > 0
        outside += valid & ~hit
    return (outside[triangles] >= votes).all(1), outside


def mask_from_probability(rgb, probability):
    """Do not force an unrecognized held object to background near the hand."""
    small = cv2.resize(rgb, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    prob = cv2.resize(probability, (small.shape[1], small.shape[0]))
    person = (prob > .5).astype(np.uint8)
    near = cv2.dilate(person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0
    seed = np.full(prob.shape, cv2.GC_PR_BGD, np.uint8)
    seed[~near] = cv2.GC_BGD
    seed[cv2.dilate(person, np.ones((5, 5), np.uint8)) > 0] = cv2.GC_PR_FGD
    seed[cv2.erode((prob > .9).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0] = cv2.GC_FGD
    if not (seed == cv2.GC_FGD).any() or not (seed == cv2.GC_BGD).any():
        raise ValueError('Unusable semantic seeds')
    cv2.grabCut(small, seed, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
    mask = binary_fill_holes(np.isin(seed, [cv2.GC_FGD, cv2.GC_PR_FGD])).astype(np.uint8)
    mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    diameter=2*SETTINGS['native_dilation']+1
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)))


def prepare(output, record, source_manifest):
    global MODEL
    root = output/'foreground_guard'/record['frame_id']; root.mkdir(parents=True, exist_ok=True)
    model_file = Path(torch.hub.get_dir())/'checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth'
    request = dict(parent_request_sha256=sha(output/'request.json'), mesh_sha256=record['mesh_sha256'],
                   source_manifest=source_manifest, settings=SETTINGS, script_sha256=sha(__file__),
                   model_sha256=sha(model_file))
    if (root/'complete.json').exists():
        receipt = read(root/'complete.json')
        if receipt['request'] != request: raise ValueError('Foreground cache request mismatch')
        for name, digest in receipt['hashes'].items():
            if sha(root/name) != digest: raise ValueError('Foreground cache checksum mismatch')
        return root
    rows, _, _ = cameras(record['frame_id'])
    expected = {r['physical_camera']: r['sha256'] for r in source_manifest['source_images']}
    profiles = read(ROOT/'camera_profiles.json')
    response = dict(zip(profiles['physical_cameras'], profiles['rgb_gain']))
    exposure = read(ROOT/'exposure.json')['fixed_exposure_gain']
    def load(row):
        if sha(row['file_path']) != expected[row['physical_camera']]: raise ValueError('Source checksum mismatch')
        rgb = np.rint(display(exr(row['file_path'])*np.array(response[row['physical_camera']]), exposure)*255).clip(0,255).astype(np.uint8)
        return np.rot90(rgb).copy()
    with ThreadPoolExecutor(max_workers=4) as pool: images = list(pool.map(load, rows))
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    if MODEL is None: MODEL = deeplabv3_resnet50(weights=weights).cuda().eval()
    preprocessing = weights.transforms(); person = weights.meta['categories'].index('person'); probabilities=[]
    for rgb in images:
        x = preprocessing(Image.fromarray(rgb)).unsqueeze(0).cuda()
        with torch.inference_mode():
            p = MODEL(x)['out'].softmax(1)[:, person:person+1]
            probabilities.append(F.interpolate(p, size=rgb.shape[:2], mode='bilinear', align_corners=False)[0,0].cpu().numpy())
    cv2.setNumThreads(1)
    # GrabCut uses process-global random state: sequential fixed-seed fits make
    # masks repeatable regardless of worker scheduling across different frames.
    portrait_masks=[]
    for rgb, p in zip(images, probabilities):
        cv2.setRNGSeed(17); portrait_masks.append(mask_from_probability(rgb, p))
    masks=np.stack([np.rot90(m,-1) for m in portrait_masks]).astype(np.uint8)
    np.savez_compressed(root/'masks.npz', masks=masks)
    atomic_json(root/'cameras.json', [r['physical_camera'] for r in rows])
    mesh=o3d.io.read_triangle_mesh(record['mesh']); v=np.asarray(mesh.vertices); t=np.asarray(mesh.triangles)
    remove, outside=carve_candidates(v,t,rows,masks,SETTINGS['outside_votes'])
    # Independent native-depth evidence, when present and bound to this exact
    # face inventory, overrides semantic proposals which would cut real fingers.
    evidence_root=Path('/mnt/data/lookcloser_dec5_5a3_surface_repair/supported_shell_control/mesh')
    evidence_receipt=evidence_root/'carving_request.json'; protected=0; evidence_hash=None
    if evidence_receipt.exists() and read(evidence_receipt)['mesh_sha256']==record['mesh_sha256']:
        evidence_path=evidence_root/'triangle_evidence.npz'; near=np.load(evidence_path)['near_counts']
        if len(near)!=len(t):raise ValueError('Independent depth face inventory mismatch')
        protected=int((remove&(near>=2)).sum());remove&=near<2; evidence_hash=sha(evidence_path)
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    before,ids,_=camera_depth(scene_for(v,t),record['camera']);hit=np.isfinite(before)
    # A silhouette prior is not allowed to punch a new enclosed hole or expose
    # a deeper backing surface. Restore proposals monotonically until stable.
    restoration_rounds=0
    while remove.any():
        after,_,_=camera_depth(scene_for(v,t[~remove]),record['camera']);after_hit=np.isfinite(after)
        removed_hit=hit & remove[np.minimum(ids,len(t)-1)]
        enclosed=binary_fill_holes(after_hit)&~after_hit
        protect=np.unique(ids[removed_hit&(after_hit|enclosed)])
        if not len(protect):break
        remove[protect]=False;restoration_rounds+=1
        if restoration_rounds>100:raise ValueError('Silhouette restoration did not converge')
    if remove.mean()>.15: raise ValueError('Excessive semantic carving; inspect masks before continuing')
    result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t[~remove]))
    result.remove_unreferenced_vertices();result.compute_vertex_normals();o3d.io.write_triangle_mesh(str(root/'mesh.ply'),result)
    np.savez_compressed(root/'evidence.npz',removed=remove, vertex_outside_votes=outside)
    centers=np.array([r['transform_matrix'] for r in rows])[:,:3,3]
    selected=np.argsort(np.linalg.norm(centers-np.array(record['camera']['transform_matrix'])[:3,3],axis=1))[:6]
    panel=Image.new('RGB',(960,1176)); draw=ImageDraw.Draw(panel)
    for j,i in enumerate(selected):
        rgb=images[i].copy();edge=cv2.morphologyEx(portrait_masks[i],cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
        rgb[edge]=[255,0,255];x,y=j%3*320,j//3*588
        panel.paste(Image.fromarray(rgb).resize((320,568)),(x,y+20));draw.text((x+2,y+2),rows[i]['physical_camera'],fill='white')
        Image.fromarray(rgb).crop((200,900,750,1480)).save(root/f'lipstick_mask_{j}.png')
    panel.save(root/'mask_review.png')
    atomic_json(root/'result.json',dict(frame_id=record['frame_id'],source_camera_count=62,removed_faces=int(remove.sum()),
        original_triangles=len(t),protected_by_independent_depth=protected,independent_depth_evidence_sha256=evidence_hash,
        conservative_restoration_rounds=restoration_rounds,no_new_enclosed_geometry_holes=True,
        mesh_sha256=sha(root/'mesh.ply'),geometry_is_semantic_hull_not_certified_depth=True,visual_status='pending'))
    atomic_json(root/'complete.json',dict(request=request,hashes={p.name:sha(p) for p in root.iterdir() if p.is_file() and p.name!='complete.json'}))
    print(f'guard={record["frame_id"]} removed={remove.sum()}/{len(t)} depth_protected={protected}',flush=True)
    return root


def install(module):
    original_render=module.render_one; original_depth=module.camera_depth
    def render(output,record,manifest):
        root=prepare(output,record,manifest)
        guarded=deepcopy(record);guarded['mesh']=str(root/'mesh.ply');guarded['mesh_sha256']=sha(root/'mesh.ply')
        masks=np.load(root/'masks.npz')['masks']; lookup=dict(zip(read(root/'cameras.json'),masks))
        def depth(scene,row):
            d,ids,b=original_depth(scene,row)
            if row['physical_camera'] in lookup:
                d=d.copy();d[lookup[row['physical_camera']]==0]=np.inf
            return d,ids,b
        module.camera_depth=depth
        try: return original_render(output,guarded,manifest)
        finally: module.camera_depth=original_depth
    module.render_one=render
