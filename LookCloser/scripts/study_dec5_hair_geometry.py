#!/usr/bin/env python3
"""One-frame, train-only raw-MVS/TSDF/depth-ranking hair controls.

All methods use one fixed hard-source RGB transfer. Source RGB remains linear
EXR until the frozen display transform. Novel-view metrics are diagnostics;
PSNR/SSIM/LPIPS require a real held-out camera and are scored separately.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter, minimum_filter

from dec5_hair import (VALIDATION, TEST, cv_pose, unproject, project, orientations,
                      orientation_planes, orientation_cost, resize_pinhole, diverse_witnesses, integer_center_pinhole)
from import_colmap_mvs_depth_dataset import read_colmap_dense_array, load_binary_pinhole_calibration
from run_dec5_hair_campaign import atomic_json, sha


def display(x):
    x = np.maximum(np.asarray(x), 0)*10.570320292648677
    x = x/(1+x)
    return np.clip(np.where(x <= .0031308, 12.92*x, 1.055*np.power(x, 1/2.4)-.055), 0, 1)


class Scene:
    def __init__(self, root, refinement=None, load_rgb=True, mvs=None, mesh=None):
        self.root = root
        self.data = root/'dataset'
        payload = json.loads((self.data/'transforms.json').read_text())
        self.validation = [f for f in payload['frames'] if f['physical_camera'] in VALIDATION]
        self.frames = [f for f in payload['frames'] if f['file_path'] in payload['train_filenames']]
        if len(self.frames) != 59 or any(f['physical_camera'] in VALIDATION+TEST for f in self.frames):
            raise ValueError('Unexpected fitting split')
        self.mvs = mvs if mvs is not None else root/'pm960'
        self.mesh = mesh if mesh is not None else root/'tsdf/mesh.ply'
        self.has_refinement = refinement is not None
        dense = self.mvs/'dense'
        calibration = load_binary_pinhole_calibration(dense/'sparse')
        self.cameras, self.depth, self.observed_depth, self.photo, self.semantic, self.rgb = [], [], [], [], [], []
        sources = {r['camera']:r for r in json.loads((root/'prepare.json').read_text())['records']}
        for frame in self.frames:
            name = frame['physical_camera']
            camera = dict(frame, **calibration[frame['file_path']])
            depth = read_colmap_dense_array(dense/'stereo/depth_maps'/(frame['file_path']+'.geometric.bin'))[..., 0]
            photo = read_colmap_dense_array(dense/'stereo/depth_maps'/(frame['file_path']+'.photometric.bin'))[..., 0]
            actual_image = cv2.imread(str(dense/'images'/frame['file_path']))
            if actual_image is None or actual_image.shape[:2] != depth.shape:
                raise ValueError('MVS image/depth raster size mismatch')
            # COLMAP's sparse camera can retain its pre-undistorter dimensions.
            camera = resize_pinhole(camera, depth.shape[1], depth.shape[0])
            depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
            self.observed_depth.append(depth.copy())
            self.photo.append(np.where(np.isfinite(photo) & (photo > 0), photo, 0).astype(np.float32))
            if refinement is not None:
                record = refinement/(name+'.npz')
                if record.exists():
                    proposed = np.load(record)
                    if str(proposed['base_sha256']) != sha(dense/'stereo/depth_maps'/(frame['file_path']+'.geometric.bin')):
                        raise ValueError('Depth override belongs to a different source map')
                    candidate = proposed['depth'].astype(np.float32)
                    if candidate.shape != depth.shape or not np.isfinite(candidate).all() or (candidate < 0).any():
                        raise ValueError('Invalid proposed source depths')
                    changed = candidate != depth
                    if not np.array_equal(changed, proposed['changed']) or np.any(changed & (proposed['confidence'] < 3/8)):
                        raise ValueError('Unqualified source depth override')
                    depth = candidate
            probability = np.load(root/'semantics'/(name+'.npz'))['probability'].astype(np.float32)/65535
            probability = cv2.resize(probability, (camera['w'], camera['h']), interpolation=cv2.INTER_LINEAR)
            if refinement is not None and np.any((depth != self.observed_depth[-1]) & (probability[..., 0] <= .5)):
                raise ValueError('Source depth override escaped the hair domain')
            if depth.shape != (camera['h'], camera['w']):
                raise ValueError('MVS calibration/depth size mismatch')
            self.cameras.append(camera)
            self.depth.append(depth)
            self.semantic.append(probability)
            if load_rgb:
                # Use the immutable local EXR copy, not the historical remote path.
                basename = Path(sources[name]['original_path']).name
                x = cv2.imread(str(root/'source/images'/basename), cv2.IMREAD_UNCHANGED)
                if x is None or not np.isfinite(x).all():
                    raise ValueError('Unreadable linear source EXR')
                self.rgb.append(np.ascontiguousarray(x[..., ::-1]))
        self.centers = np.stack([cv_pose(f)[:3, 3] for f in self.cameras])

    def head_bounds(self):
        points = []
        for camera, depth, semantic in zip(self.cameras, self.depth, self.semantic):
            yy, xx = np.nonzero((semantic[..., 0] > .8) & (depth > 0))
            yy, xx = yy[::8], xx[::8]
            if len(xx):
                points.append(unproject(np.stack((xx+.5, yy+.5), -1), depth[yy, xx], camera))
        xyz = np.concatenate(points)
        # Spatial crop is train-derived. It is NOT a strand/background classifier.
        return np.quantile(xyz, [.02, .98], axis=0)+np.array([[-.25]*3, [.25]*3])


def fuse(scene, destination, voxel_length=.005):
    import open3d as o3d
    destination.mkdir(parents=True, exist_ok=False)
    bounds = scene.head_bounds()
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length, sdf_trunc=voxel_length*8,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for index, (camera, source) in enumerate(zip(scene.cameras, scene.depth)):
        yy, xx = np.indices(source.shape)
        xyz = unproject(np.stack((xx+.5, yy+.5), -1), source, camera)
        inside = ((xyz >= bounds[0]) & (xyz <= bounds[1])).all(-1)
        depth = np.where(inside, source, 0).astype(np.float32)
        color = cv2.imread(str(scene.mvs/'dense/images'/camera['file_path']))[..., ::-1].copy()
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(depth), depth_scale=1., depth_trunc=20.,
            convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(camera['w'], camera['h'], *integer_center_pinhole(camera))
        volume.integrate(rgbd, intrinsic, np.linalg.inv(cv_pose(camera)))
        atomic_json(destination/'progress.json', dict(status='integrating', completed=index+1, total=59))
    mesh = volume.extract_triangle_mesh()
    if len(mesh.triangles) == 0:
        raise RuntimeError('TSDF produced an empty head mesh')
    o3d.io.write_triangle_mesh(str(destination/'mesh.ply'), mesh)
    atomic_json(destination/'mesh.json', dict(bounds=bounds.tolist(), voxel_length=voxel_length, sdf_trunc=voxel_length*8,
        train_only=True, cameras=59, depth_convention='OpenCV camera-z', units='calibration units',
        mesh_sha256=sha(destination/'mesh.ply'), triangles=len(mesh.triangles),
        component_pruning=False, unknown_depth_not_background=True))
    atomic_json(destination/'progress.json', dict(status='complete', triangles=len(mesh.triangles)))


def sample(image, uv, mode='bilinear'):
    """CHW tensor, continuous pixel-center coordinates (first center=.5)."""
    _, h, w = image.shape
    grid = uv*uv.new_tensor([2/w, 2/h])-1
    return F.grid_sample(image[None], grid[None, None], mode=mode,
                         padding_mode='zeros', align_corners=False)[0, :, 0].T


def tensor_project(points, camera):
    pose = torch.as_tensor(cv_pose(camera), device=points.device, dtype=points.dtype)
    xyz = (points-pose[:3, 3]) @ pose[:3, :3]
    z = xyz[:, 2]
    uv = xyz[:, :2]/z.clamp_min(1e-8)[:, None]
    uv = uv*uv.new_tensor([camera['fl_x'], camera['fl_y']])+uv.new_tensor([camera['cx'], camera['cy']])
    return uv, z


def tensor_unproject(uv, depth, camera):
    xy = (uv-uv.new_tensor([camera['cx'], camera['cy']]))/uv.new_tensor([camera['fl_x'], camera['fl_y']])
    xyz = torch.cat((xy*depth[:, None], depth[:, None]), -1)
    pose = torch.as_tensor(cv_pose(camera), device=uv.device, dtype=uv.dtype)
    return xyz @ pose[:3, :3].T+pose[:3, 3]


class Renderer:
    def __init__(self, scene):
        self.scene = scene
        self.visibility_depth_sampling = getattr(scene, 'visibility_depth_sampling', 'bilinear')
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(24*(1 << 30)/torch.cuda.get_device_properties(0).total_memory)
        # The explicit neural-visibility ablation changes only the source z test.
        visibility=getattr(scene,'visibility_depth',scene.observed_depth)
        self.depth = [torch.as_tensor(d, device='cuda')[None] for d in visibility]
        self.semantic = [torch.as_tensor(s, device='cuda').permute(2, 0, 1) for s in scene.semantic]
        self.rgb = [torch.as_tensor(x, device='cuda').permute(2, 0, 1) for x in scene.rgb]
        points, hair, background, changed = [], [], [], []
        for camera, array, original, sem in zip(scene.cameras, scene.depth, scene.observed_depth, self.semantic):
            depth = torch.as_tensor(array, device='cuda')[None]
            h, w = depth.shape[-2:]
            yy, xx = torch.meshgrid(torch.arange(h, device='cuda'), torch.arange(w, device='cuda'), indexing='ij')
            valid = depth[0] > 0
            uv = torch.stack((xx[valid]+.5, yy[valid]+.5), -1)
            points.append(tensor_unproject(uv, depth[0][valid], camera))
            hair.append(sem[0][valid] > .25)
            background.append(sem[2][valid] > .9)
            changed.append(torch.as_tensor(array != original, device='cuda')[valid])
        self.points = torch.cat(points)
        self.hair = torch.cat(hair)
        self.background = torch.cat(background)
        self.changed = torch.cat(changed)

    def source_order(self,camera):
        center=getattr(self.scene,'source_reference_center',cv_pose(camera)[:3,3])
        order=np.argsort(np.linalg.norm(self.scene.centers-center,axis=-1),kind='stable')[:8]
        allowed=getattr(self.scene,'allowed_visibility_sources',None)
        if allowed is not None and not {self.scene.frames[i]['physical_camera'] for i in order}<=allowed:
            raise ValueError('Frozen neural source subset does not cover requested source order')
        return order

    def load_mattes(self, directory):
        self.alpha, self.foreground, self.known = [], [], []
        for index,frame in enumerate(self.scene.frames):
            data = np.load(directory/(frame['physical_camera']+'.npz'))
            if not np.isfinite(data['alpha']).all() or not np.isfinite(data['foreground']).all():raise ValueError('Nonfinite source matte')
            height,width=self.rgb[index].shape[1:]
            if tuple(data['original_shape'][:2])!=(height,width):raise ValueError('Matte/native RGB raster mismatch')
            alpha = np.zeros((width, height), np.float32)
            foreground = np.zeros((width, height, 3), np.float32)
            known = np.zeros((width, height), np.float32)
            x0, y0, x1, y1 = [int(x) for x in data['portrait_crop']]
            alpha[y0:y1, x0:x1] = data['alpha'].astype(np.float32)
            foreground[y0:y1, x0:x1] = data['foreground'].astype(np.float32)
            known[y0:y1, x0:x1] = 1
            self.alpha.append(torch.as_tensor(np.ascontiguousarray(np.rot90(alpha, -1)), device='cuda')[None])
            self.foreground.append(torch.as_tensor(np.ascontiguousarray(np.rot90(foreground, -1)), device='cuda').permute(2, 0, 1))
            self.known.append(torch.as_tensor(np.ascontiguousarray(np.rot90(known, -1)), device='cuda')[None])

    @torch.inference_mode()
    def warp_layer(self, camera, depth, foreground=False, decontaminate=False, premultiply=False, allow_zero_alpha=False):
        h, w = depth.shape
        yy, xx = torch.meshgrid(torch.arange(h, device='cuda'), torch.arange(w, device='cuda'), indexing='ij')
        xyz = tensor_unproject(torch.stack((xx.flatten()+.5, yy.flatten()+.5), -1), depth.flatten(), camera)
        order = self.source_order(camera)
        rgb = torch.zeros((h*w, 3), device='cuda')
        alpha = torch.zeros(h*w, device='cuda')
        ids = torch.full((h*w,), -1, device='cuda', dtype=torch.int16)
        support = torch.zeros(h*w, device='cuda', dtype=torch.int16)
        zero_support=torch.zeros_like(support) if allow_zero_alpha else None
        zero_ids=torch.full_like(ids,-1) if allow_zero_alpha else None
        for begin in range(0, h*w, 65536):
            end = begin+65536
            for index in order:
                source = self.scene.cameras[index]
                uv, z = tensor_project(xyz[begin:end], source)
                observed = sample(self.depth[index], uv, mode=getattr(self,'visibility_depth_sampling','bilinear'))[:, 0]
                valid = ((depth.flatten()[begin:end] > 0) & (z > 0) & (observed > 0)
                         & ((observed-z).abs() <= .01*z+.002) & (uv[:, 0] >= .5) & (uv[:, 0] <= source['w']-.5)
                         & (uv[:, 1] >= .5) & (uv[:, 1] <= source['h']-.5))
                semantics = sample(self.semantic[index], uv)
                native_uv = uv*uv.new_tensor([self.rgb[index].shape[2]/source['w'], self.rgb[index].shape[1]/source['h']])
                if foreground:
                    a = sample(self.alpha[index], native_uv)[:, 0]
                    known = sample(self.known[index], native_uv)[:, 0] > .99
                    hair_domain=semantics[:,0]>.1
                    if hasattr(self,'matte_hair_domain'):hair_domain=sample(self.matte_hair_domain[index],uv)[:,0]>.5
                    domain=valid&known&hair_domain
                    if allow_zero_alpha:
                        zero=valid&known&(hair_domain|(semantics[:,2]>.9))&(a<=.001);zero_support[begin:end]+=zero.to(torch.int16)
                        first_zero=zero&(zero_ids[begin:end]<0);zero_ids[begin:end][first_zero]=int(index)
                    valid=domain&(a>.001)
                    source_rgb = self.foreground[index] if decontaminate else self.rgb[index]
                    if premultiply:source_rgb=source_rgb*self.alpha[index]
                    colors = sample(source_rgb, native_uv)
                else:
                    a = torch.ones_like(z)
                    valid &= semantics[:, 2] > .8
                    colors = sample(self.rgb[index], native_uv)
                support[begin:end] += valid.to(torch.int16)
                take = valid & (ids[begin:end] < 0)
                rgb[begin:end][take] = colors[take]
                alpha[begin:end][take] = a[take]
                ids[begin:end][take] = int(index)
        if allow_zero_alpha:
            zero=(ids<0)&(zero_support>=3);ids[zero]=zero_ids[zero];support[zero]=zero_support[zero]
        return dict(rgb=rgb.reshape(h,w,3),alpha=alpha.reshape(h,w),source_id=ids.reshape(h,w),support=support.reshape(h,w))

    @torch.inference_mode()
    def rasterize(self, camera):
        h, w = camera['h'], camera['w']
        buffers = [torch.full((h*w,), float('inf'), device='cuda') for _ in range(4)]
        # Four-pixel footprint for half-resolution MVS, identical in every arm.
        # This is raster coverage, not a candidate-dependent hole fill.
        for begin in range(0, len(self.points), 262144):
            end = begin+262144
            uv, z = tensor_project(self.points[begin:end], camera)
            xy = torch.floor(uv-.5).long()
            for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
                x, y = xy[:, 0]+dx, xy[:, 1]+dy
                valid = (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
                for buffer, mask in zip(buffers, (valid, valid & self.hair[begin:end], valid & self.background[begin:end],
                                                 valid & self.changed[begin:end])):
                    indices = y[mask]*w+x[mask]
                    buffer.scatter_reduce_(0, indices, z[mask], reduce='amin', include_self=True)
        return [torch.where(torch.isfinite(d), d, 0).reshape(h, w) for d in buffers]

    @torch.inference_mode()
    def warp(self, camera, depth):
        h, w = depth.shape
        yy, xx = torch.meshgrid(torch.arange(h, device='cuda'), torch.arange(w, device='cuda'), indexing='ij')
        xyz = tensor_unproject(torch.stack((xx.flatten()+.5, yy.flatten()+.5), -1), depth.flatten(), camera)
        order = self.source_order(camera)
        rgb = torch.zeros((h*w, 3), device='cuda')
        ids = torch.full((h*w,), -1, device='cuda', dtype=torch.int16)
        support = torch.zeros(h*w, device='cuda', dtype=torch.int16)
        hair_vote = torch.zeros(h*w, device='cuda')
        skin_vote = torch.zeros(h*w, device='cuda')
        reprojection = torch.zeros(h*w, device='cuda')
        for begin in range(0, h*w, 65536):
            end = begin+65536
            point = xyz[begin:end]
            valid_target = depth.flatten()[begin:end] > 0
            for index in order:
                source = self.scene.cameras[index]
                uv, z = tensor_project(point, source)
                sd = sample(self.depth[index], uv, mode=getattr(self,'visibility_depth_sampling','bilinear'))[:, 0]
                valid = (valid_target & (z > 0) & (sd > 0) & (uv[:, 0] >= .5) & (uv[:, 0] <= source['w']-.5)
                         & (uv[:, 1] >= .5) & (uv[:, 1] <= source['h']-.5) & ((sd-z).abs() <= .01*z+.002))
                semantics = sample(self.semantic[index], uv)
                resampler = getattr(self, 'visibility_depth_resampler', None)
                if resampler is not None:
                    eligible = (valid_target & ~valid & (z > 0)
                                & (uv[:, 0] >= .5) & (uv[:, 0] <= source['w']-.5)
                                & (uv[:, 1] >= .5) & (uv[:, 1] <= source['h']-.5))
                    domain = getattr(self, 'visibility_resampling_domain', None)
                    if domain is not None:
                        eligible &= domain.flatten()[begin:end]
                    sd = resampler(index, uv, sd, semantics, eligible)
                    valid = (valid_target & (z > 0) & (sd > 0)
                             & (uv[:, 0] >= .5) & (uv[:, 0] <= source['w']-.5)
                             & (uv[:, 1] >= .5) & (uv[:, 1] <= source['h']-.5)
                             & ((sd-z).abs() <= .01*z+.002))
                support[begin:end] += valid.to(torch.int16)
                back_uv, _ = tensor_project(tensor_unproject(uv, sd, source), camera)
                original_uv = torch.stack((xx.flatten()[begin:end]+.5, yy.flatten()[begin:end]+.5), -1)
                error = torch.linalg.norm(back_uv-original_uv, dim=-1).clamp_max(20)
                reprojection[begin:end] += torch.where(valid, error, 0.)
                hair_vote[begin:end] += valid*semantics[:, 0]
                skin_vote[begin:end] += valid*semantics[:, 1]
                take = valid & (ids[begin:end] < 0)
                # Normalized grids address matching native EXR pixels with scaled intrinsics.
                native_uv = uv*uv.new_tensor([self.rgb[index].shape[2]/source['w'], self.rgb[index].shape[1]/source['h']])
                colors = sample(self.rgb[index], native_uv)
                rgb[begin:end][take] = colors[take]
                ids[begin:end][take] = int(index)
        return dict(rgb=rgb.reshape(h, w, 3), source_id=ids.reshape(h, w),
                    support=support.reshape(h, w), hair_probability=(hair_vote/support.clamp_min(1)).reshape(h, w),
                    skin_probability=(skin_vote/support.clamp_min(1)).reshape(h, w),
                    reprojection_error=torch.where(support > 0, reprojection/support.clamp_min(1), 20.).reshape(h,w))


def mesh_depth(mesh_path, camera):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    yy, xx = np.indices((camera['h'], camera['w']))
    uv = np.stack((xx+.5, yy+.5), -1)
    points = unproject(uv, np.ones(uv.shape[:2]), camera)
    origin = cv_pose(camera)[:3, 3]
    directions = points-origin
    rays = np.concatenate((np.broadcast_to(origin, directions.shape), directions), -1).astype(np.float32)
    # Directions have camera-z=1, not unit length: t_hit IS camera-z.
    depth = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    return np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)


def render(scene, output, method='raw', baseline=None, cameras=None, layer_style=None, matting=None,
           target_depth=None, target_depth_key='expected', target_depth_gate=False,reuse_baseline_depth=False):
    output.mkdir(parents=True, exist_ok=False)
    renderer = Renderer(scene)
    if layer_style:
        if baseline is None or matting is None:
            raise ValueError('Layer ablations require a frozen baseline and train matting')
        renderer.load_mattes(matting)
    cameras = scene.validation if cameras is None else cameras
    frozen_records=({row['camera']:row for row in json.loads((baseline/'complete.json').read_text())['records']}
                    if reuse_baseline_depth else {})
    records = []
    for index, camera in enumerate(cameras):
        started = time.time()
        name = camera['physical_camera']
        if reuse_baseline_depth:
            if sha(baseline/(name+'.npz'))!=frozen_records[name]['npz_sha256']:raise ValueError('Changed frozen receiver archive')
            parent=np.load(baseline/(name+'.npz'))
            front,hair_depth,back=[torch.as_tensor(parent[key],device='cuda') for key in ('front_depth','projected_hair_depth','back_depth')]
            changed_depth=torch.zeros_like(front)
        else:front, hair_depth, back, changed_depth = renderer.rasterize(camera)
        qualified_depth = None
        if scene.has_refinement:
            if baseline is None:
                raise ValueError('Local source-depth refinement requires a frozen target baseline')
            parent = np.load(baseline/(name+'.npz'))
            base_depth = torch.as_tensor(parent['front_depth'], device='cuda')
            potential = (torch.as_tensor(parent['roi'], device='cuda') & (changed_depth > 0)
                         & (changed_depth <= front+.002) & ((changed_depth-base_depth).abs() > 1e-5))
            front = torch.where(potential, changed_depth, base_depth)
        measured = None
        if method == 'mesh':
            measured = torch.as_tensor(mesh_depth(scene.mesh, camera), device='cuda')
            # Keep raw room/background geometry; the TSDF experiment changes only head receivers.
            front = torch.where(measured > 0, measured, front)
        if target_depth is not None:
            if baseline is None:
                raise ValueError('Target depth controls require a baseline ROI')
            parent = np.load(baseline/(name+'.npz'))
            proposal = np.load(target_depth/(name+'.npz'))[target_depth_key]
            if proposal.shape != front.shape or not np.isfinite(proposal).all():
                raise ValueError('Invalid neural target depth')
            domain = parent['roi'] & (proposal > 0)
            front = torch.as_tensor(parent['front_depth'].copy(), device='cuda')
            front[torch.as_tensor(domain, device='cuda')] = torch.as_tensor(proposal, device='cuda')[torch.as_tensor(domain, device='cuda')]
        result = renderer.warp(camera, front)
        if scene.has_refinement:
            qualified_depth = (potential & (result['support'] >= 3) & (result['hair_probability'] >= .5)
                               & (result['skin_probability'] < .7)).cpu().numpy()
        if target_depth is not None and target_depth_gate:
            qualified_depth = (torch.as_tensor(domain, device='cuda') & (result['support'] >= 3)
                               & (result['hair_probability'] >= .5) & (result['skin_probability'] < .7)
                               & (result['reprojection_error'] < torch.as_tensor(parent['reprojection_error'], device='cuda')*.99)).cpu().numpy()
        arrays = {k:v.cpu().numpy() for k, v in result.items()}
        arrays.update(front_depth=front.cpu().numpy(), back_depth=back.cpu().numpy(),
                      projected_hair_depth=hair_depth.cpu().numpy())
        if qualified_depth is not None:
            arrays['depth_override_gate'] = qualified_depth
        if measured is not None:
            arrays['mesh_depth'] = measured.cpu().numpy()
        roi = cv2.dilate((arrays['projected_hair_depth'] > 0).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        # Reject projected hair hidden behind an independently nearer surface.
        roi &= ((arrays['front_depth'] == 0) | (arrays['projected_hair_depth'] <= arrays['front_depth']*1.02+.002))
        roi &= arrays['skin_probability'] < .7
        arrays['roi'] = roi
        if baseline is not None:
            parent = np.load(baseline/(name+'.npz'))
            # Freeze the train-derived edit domain before comparing candidates.
            roi = parent['roi'].copy()
            if not reuse_baseline_depth:roi &= parent['skin_probability'] < .7
            arrays['roi'] = roi
            keep = roi if qualified_depth is None else (roi & qualified_depth)
            for key in ('rgb', 'front_depth', 'source_id', 'support', 'hair_probability',
                        'skin_probability', 'projected_hair_depth', 'back_depth', 'reprojection_error'):
                arrays[key][~keep] = parent[key][~keep]
            if not np.array_equal(arrays['rgb'][~keep], parent['rgb'][~keep]):
                raise ValueError('Non-hair RGB changed')
        if layer_style:
            parent = np.load(baseline/(name+'.npz'))
            df = torch.as_tensor(parent['projected_hair_depth'] if layer_style == 'two-depth' else parent['front_depth'], device='cuda')
            db = torch.as_tensor(parent['back_depth'], device='cuda')
            fg = renderer.warp_layer(camera, df, foreground=True, decontaminate=layer_style in ('fb', 'two-depth'))
            bg = renderer.warp_layer(camera, db)
            alpha = fg['alpha']
            # Compare the same uncertain edge band in every alpha ablation.
            band = (alpha > .001) & (alpha < .995)
            gate = (torch.as_tensor(parent['roi'], device='cuda') & band & (df > 0) & (db > df+.002)
                    & (fg['support'] >= 2) & (bg['support'] >= 2))
            actual_alpha = (alpha >= .5).float() if layer_style == 'binary' else alpha
            mixed = fg['rgb']*actual_alpha[..., None]+bg['rgb']*(1-actual_alpha[..., None])
            arrays['rgb'] = parent['rgb'].copy()
            arrays['rgb'][gate.cpu().numpy()] = mixed.cpu().numpy()[gate.cpu().numpy()]
            arrays['front_depth'] = parent['front_depth'].copy()
            arrays['front_depth'][gate.cpu().numpy()] = df.cpu().numpy()[gate.cpu().numpy()]
            arrays['hair_coverage'] = parent['hair_probability'].copy()
            arrays['hair_coverage'][gate.cpu().numpy()] = actual_alpha.cpu().numpy()[gate.cpu().numpy()]
            arrays['layer_gate'] = gate.cpu().numpy()
            arrays['layer_alpha'] = actual_alpha.cpu().numpy()
            arrays['foreground_premultiplied'] = (fg['rgb']*actual_alpha[..., None]).cpu().numpy()
            arrays['background_rgb'] = bg['rgb'].cpu().numpy()
            arrays['foreground_source_id'] = fg['source_id'].cpu().numpy()
            arrays['background_source_id'] = bg['source_id'].cpu().numpy()
            arrays['layer_front_depth'] = df.cpu().numpy()
            arrays['layer_back_depth'] = db.cpu().numpy()
            # Preserve all baseline geometry diagnostics outside the actual layer gate.
            arrays['source_id'] = parent['source_id'].copy()
            arrays['support'] = parent['support'].copy()
            arrays['reprojection_error'] = parent['reprojection_error'].copy()
            if layer_style == 'two-depth':
                independent = renderer.warp(camera, df)['reprojection_error'].cpu().numpy()
                arrays['reprojection_error'][gate.cpu().numpy()] = independent[gate.cpu().numpy()]
            arrays['source_id'][gate.cpu().numpy()] = fg['source_id'].cpu().numpy()[gate.cpu().numpy()]
            arrays['support'][gate.cpu().numpy()] = torch.minimum(fg['support'], bg['support']).cpu().numpy()[gate.cpu().numpy()]
        np.savez_compressed(output/(name+'.npz'), **arrays)
        if reuse_baseline_depth:
            for key in ('front_depth','projected_hair_depth','back_depth','roi'):
                if not np.array_equal(arrays[key],parent[key]):raise ValueError('Frozen receiver geometry changed')
        rgb = np.rint(display(arrays['rgb'])*255).astype(np.uint8)
        cv2.imwrite(str(output/(name+'.png')), rgb[..., ::-1])
        preview = np.rot90(rgb)
        preview = cv2.resize(preview, (360, 640), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output/(name+'_review.jpg')), preview[..., ::-1])
        records.append(dict(camera=name, seconds=time.time()-started, missing_source_pixels=int((arrays['source_id'] < 0).sum()),
                            npz_sha256=sha(output/(name+'.npz')), png_sha256=sha(output/(name+'.png'))))
        atomic_json(output/'progress.json', dict(status='rendering', completed=index+1, total=len(cameras)))
    atomic_json(output/'complete.json', dict(method=method, train_sources=[f['physical_camera'] for f in scene.frames],
        uses_validation_rgb=False, uses_test_rgb=False, records=records, source_count=8,
        source_selection=('nearest to path-local reference' if hasattr(scene,'source_reference_center') else 'nearest to target camera')+'; first depth-valid source; no RGB averaging',
        source_order_calibration_sha256=getattr(scene, 'source_order_calibration_sha256', None),
        source_reference_camera_sha256=getattr(scene,'source_reference_camera_sha256',None),
        reuse_baseline_depth=reuse_baseline_depth,
        visibility_depth_sampling=renderer.visibility_depth_sampling,
        depth_convention='OpenCV camera-z', units='calibration units',
        input_geometry_longest_edge=scene.cameras[0]['w'], output_resolution=[1920, 1080],
        mesh_miss_policy='raw MVS fallback; cannot certify TSDF coverage recovery', layer_style=layer_style,
        layer_minimum_support=2 if layer_style else None,
        baseline_sha256=sha(baseline/'complete.json') if baseline else None,
        source_maps_request_sha256=sha(scene.mvs/'request.json'),
        implementation_sha256=sha(__file__),
        geometry_mesh_sha256=sha(scene.mesh) if method == 'mesh' else None,
        target_depth_key=target_depth_key if target_depth else None,
        target_depth_gate=target_depth_gate,
        target_depth_receipt_sha256=sha(target_depth/'complete.json') if target_depth else None,
        uses_original_measured_depth_for_visibility=not hasattr(scene,'visibility_depth'),
        neural_visibility_depth_key=getattr(scene,'visibility_depth_key',None),
        neural_visibility_source_count=len(scene.allowed_visibility_sources) if hasattr(scene,'allowed_visibility_sources') else None,
        neural_visibility_receipt_sha256=getattr(scene,'visibility_receipt_sha256',None)))
    atomic_json(output/'progress.json', dict(status='complete', completed=len(cameras), total=len(cameras)))


def rerank(scene, destination, orientation_weight=1., neural_depth=None, neural_candidate='front'):
    """Small STITCH-inspired source-depth candidate experiment, not full STITCH."""
    from scipy.ndimage import distance_transform_edt
    destination.mkdir(parents=True, exist_ok=False)
    if neural_depth is not None:
        volume = json.loads((neural_depth/'request.json').read_text())
        training = json.loads((scene.root/'nerf_v2/request.json').read_text())
        if (volume['checkpoint_sha256'] != sha(scene.root/'nerf_v2/best.pt')
                or training['data_sha256'] != sha(scene.data/'transforms.json')):
            raise ValueError('Neural depth provenance differs from campaign inputs')
    maps, confidence, gray_images = [], [], []
    focus = scene.head_bounds().mean(0)
    for camera in scene.cameras:
        image = cv2.imread(str(scene.mvs/'dense/images'/camera['file_path']))[..., ::-1].astype(np.float32)/255
        o, q = orientations(image)
        maps.append(o)
        confidence.append(q)
        gray_images.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    records = []
    for index, (camera, base, sem) in enumerate(zip(scene.cameras, scene.depth, scene.semantic)):
        tolerance = np.maximum(2*base/min(camera['fl_x'],camera['fl_y']), .002)
        spread = maximum_filter(base, size=3)-minimum_filter(base, size=3)
        uncertain = ((base <= 0) | (scene.photo[index] <= 0)
                     | (np.abs(scene.photo[index]-base) > tolerance) | (spread > 4*tolerance))
        domain = (sem[..., 0] > .5) & (confidence[index] > .3) & uncertain
        # Hole proposals borrow at most three pixels; acceptance still requires 3 witnesses.
        distance, nearest = distance_transform_edt(base <= 0, return_indices=True)
        seed = np.where((base > 0) | (distance <= 3), base[tuple(nearest)], 0)
        domain &= seed > 0
        yy, xx = np.nonzero(domain)
        refined = base.copy()
        changed = np.zeros_like(base, bool)
        output_confidence = np.zeros_like(base)
        witnesses = diverse_witnesses(scene.centers, index, focus)
        if len(witnesses) < 3:
            yy, xx = yy[:0], xx[:0]  # Unconstrained view: retain the base, never invent evidence.
        for begin in range(0, len(xx), 8192):
            y, x = yy[begin:begin+8192], xx[begin:begin+8192]
            uv = np.stack((x+.5, y+.5), -1)
            d = seed[y, x]
            delta = np.maximum(2*d/min(camera['fl_x'], camera['fl_y']), .002)
            candidates = np.stack((base[y, x], scene.photo[index][y, x], d,
                                    d-delta, d+delta, d-2*delta, d+2*delta), -1)
            if neural_depth is not None:
                neural = np.load(neural_depth/(camera['physical_camera']+'.npz'))[neural_candidate]
                if neural.shape != base.shape:
                    neural = cv2.resize(neural, (base.shape[1],base.shape[0]), interpolation=cv2.INTER_NEAREST)
                candidates = np.concatenate((candidates, neural[y, x, None]), -1)
            xyz = unproject(uv[:, None, :], candidates, camera)
            normals, qs, residuals, counts, photo_errors = [], [], [], [], []
            target_image = gray_images[index]
            target_color = target_image[y, x]
            for j in witnesses:
                other = scene.cameras[j]
                p, z = project(xyz, other)
                mapx, mapy = (p[..., 0]-.5).astype(np.float32), (p[..., 1]-.5).astype(np.float32)
                observed = cv2.remap(scene.depth[j], mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                hair = cv2.remap(scene.semantic[j][..., 0], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                orientation = cv2.remap(maps[j], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                q = cv2.remap(confidence[j], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                tolerance = np.maximum(2*z/min(other['fl_x'], other['fl_y']), .002)
                relative = np.abs(observed-z)/np.maximum(tolerance, 1e-8)
                a, b = xyz-scene.centers[index], xyz-scene.centers[j]
                cosine = (a*b).sum(-1)/(np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)).clip(1e-12)
                parallax = np.arccos(np.clip(cosine, -1, 1)) >= np.deg2rad(1.)
                visible = ((observed > 0) & (z > 0) & (hair > .5) & parallax
                           & (relative <= 2) & (candidates > 0))
                normals.append(orientation_planes(p, orientation, other))
                qs.append(q*hair*visible)
                counts.append(visible)
                residuals.append(np.minimum(relative, 2))
                other_image = gray_images[j]
                color = cv2.remap(other_image, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                photo_errors.append(np.minimum(np.abs(color-target_color[:, None])/.1, 2))
            support = np.stack(counts).sum(0)
            ocost, _ = orientation_cost(np.stack(normals), np.stack(qs))
            residual = (np.stack(residuals)*np.stack(counts)).sum(0)/np.maximum(support, 1)
            photo = (np.stack(photo_errors)*np.stack(counts)).sum(0)/np.maximum(support, 1)
            prior = np.abs(candidates-d[:, None])/delta[:, None]
            cost = residual+.25*photo+.05*prior
            if orientation_weight:
                cost += orientation_weight*10*ocost
            cost[(support < 3) | (candidates <= 0)] = np.inf
            best = cost.argmin(-1)
            row = np.arange(len(y))
            score = cost[row, best]
            baseline_cost = cost[:, 0]
            take = np.isfinite(score) & ((score < baseline_cost*.9) | ((base[y, x] <= 0) & (support[row, best] >= 4)))
            # A different candidate label with equal numerical depth is not a change.
            take &= np.abs(candidates[row, best]-base[y, x]) > 1e-6
            refined[y[take], x[take]] = candidates[row[take], best[take]]
            changed[y[take], x[take]] = True
            output_confidence[y[take], x[take]] = np.minimum(support[row[take], best[take]]/8, 1)
        path = destination/(camera['physical_camera']+'.npz')
        source = scene.mvs/'dense/stereo/depth_maps'/(camera['file_path']+'.geometric.bin')
        np.savez_compressed(path, depth=refined, changed=changed, confidence=output_confidence, base_sha256=sha(source))
        records.append(dict(camera=camera['physical_camera'], changed_pixels=int(changed.sum()),
                            new_depth_pixels=int(((base == 0) & (refined > 0)).sum()), sha256=sha(path)))
        atomic_json(destination/'progress.json', dict(status='ranking', completed=index+1, total=59))
    atomic_json(destination/'complete.json', dict(records=records, orientation_weight=orientation_weight,
        independent_witnesses=8, minimum_support=3, parallax_min_degrees=1., candidates=8 if neural_depth else 7,
        neural_candidate=neural_candidate if neural_depth else None,
        domain='train hair, orientation confidence, and measured/photo depth ambiguity',
        target_override='only changed hypotheses with >=3 original measured witnesses',
        implementation_sha256=sha(__file__), source_request_sha256=sha(scene.mvs/'request.json'),
        low_confidence_fallback='unchanged base', full_stitch=False, uses_eval_rgb=False))
    atomic_json(destination/'progress.json', dict(status='complete', completed=59, total=59))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('fuse', 'render', 'rerank'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--method', choices=('raw', 'mesh'), default='raw')
    p.add_argument('--refinement', type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--orientation-weight', type=float, default=1.)
    p.add_argument('--layer-style', choices=('binary', 'soft', 'fb', 'two-depth'))
    p.add_argument('--matting', type=Path)
    p.add_argument('--neural-depth', type=Path)
    p.add_argument('--neural-candidate', choices=('front', 'peak1', 'peak2'), default='front')
    p.add_argument('--mvs', type=Path)
    p.add_argument('--mesh', type=Path)
    p.add_argument('--voxel-length', type=float, default=.005)
    p.add_argument('--target-depth', type=Path)
    p.add_argument('--target-field-training',type=Path,
                   help='Bind target depth to a new field training directory; defaults to the frozen original nerf_v2.')
    p.add_argument('--target-depth-key', choices=('expected','median','front','peak1','peak1_subpixel'), default='expected')
    p.add_argument('--target-field-checkpoint',type=Path,
                   help='Explicit frozen checkpoint when assessing a saved gate rather than the PSNR-selected best.')
    p.add_argument('--source-field-depth',type=Path,
                   help='Separate ablation: frozen train-only neural z maps for source visibility inside their ROI.')
    p.add_argument('--source-depth-key',choices=('expected','median'),default='expected')
    p.add_argument('--visibility-depth-sampling',choices=('bilinear','nearest'),default='bilinear',
                   help='Visibility-only control: nearest avoids averaging across depth discontinuities; RGB remains bilinear.')
    p.add_argument('--source-reference-camera',type=Path,
                   help='Path-local fixed source ordering; a JSON camera without RGB. Default keeps per-target ordering.')
    p.add_argument('--reuse-baseline-depth',action='store_true',
                   help='Counterfactual renderer test: reuse exact baseline depth buffers and ROI.')
    p.add_argument('--target-depth-gate', action='store_true',
                   help='Accept only >=3 original witnesses and improved measured roundtrip, else exact baseline.')
    p.add_argument('--camera-manifest', type=Path)
    p.add_argument('--source-order-calibration', type=Path,
                   help='Freeze nearest-source order to another train59 transforms.json during calibration screens.')
    args = p.parse_args()
    if args.reuse_baseline_depth and (args.baseline is None or args.stage!='render' or args.target_depth is not None or args.refinement is not None or args.method!='raw'):
        p.error('Frozen receiver-depth test requires a baseline raw render without geometry overrides')
    if args.target_depth_gate and args.target_depth is None:
        p.error('--target-depth-gate requires --target-depth')
    if args.target_depth is not None and args.refinement is not None:
        p.error('Compare source refinement and target profiles as separate controls')
    if args.target_depth is not None:
        provenance = json.loads((args.target_depth/'complete.json').read_text())
        field_training=args.target_field_training or args.root/'nerf_v2'
        field_checkpoint=args.target_field_checkpoint or field_training/'best.pt'
        if (provenance['checkpoint_sha256'] != sha(field_checkpoint)
                or not provenance.get('frozen_state_unchanged', provenance.get('unchanged', False))):
            raise ValueError('Target depths do not belong to the frozen campaign field')
        if args.target_field_training is not None:
            field_request=json.loads((field_training/'request.json').read_text())
            if field_request['data_sha256']!=sha(args.root/'dataset/transforms.json'):
                raise ValueError('New target field belongs to a different source split/calibration')
        if args.camera_manifest is not None:
            if (args.baseline is None
                    or provenance.get('camera_manifest_sha256') != sha(args.camera_manifest)
                    or provenance.get('target_roi_parent_receipt_sha256') != sha(args.baseline/'complete.json')):
                raise ValueError('Target profiles are bound to different poses or baseline ROI')
        for record in provenance.get('records', []):
            if sha(args.target_depth/(record['camera']+'.npz')) != record['sha256']:
                raise ValueError('Changed target profile archive')
    scene = Scene(args.root, args.refinement, load_rgb=args.stage == 'render', mvs=args.mvs, mesh=args.mesh)
    scene.visibility_depth_sampling=args.visibility_depth_sampling
    if args.source_reference_camera is not None:
        if args.stage!='render':p.error('Source reference is a rendering-only control')
        reference=json.loads(args.source_reference_camera.read_text())
        scene.source_reference_center=cv_pose(reference)[:3,3]
        scene.source_reference_camera_sha256=sha(args.source_reference_camera)
    if args.source_field_depth is not None:
        if args.target_depth is None or args.stage!='render':p.error('Neural visibility requires a frozen neural target depth render')
        receipt=json.loads((args.source_field_depth/'complete.json').read_text())
        if (not receipt.get('source_visibility_only') or not receipt.get('frozen_state_unchanged')
                or receipt['checkpoint_sha256']!=provenance['checkpoint_sha256']
                or receipt['dataset_sha256']!=sha(scene.data/'transforms.json')
                or receipt.get('uses_validation_rgb') is not False or receipt.get('uses_test_rgb') is not False):
            raise ValueError('Source visibility must come from the same frozen train-only field')
        records={r['camera']:r for r in receipt['records']}
        if not records or not set(records)<={f['physical_camera'] for f in scene.frames}:raise ValueError('Neural source visibility inventory mismatch')
        scene.allowed_visibility_sources=set(records)
        scene.visibility_depth=[]
        for frame,base in zip(scene.frames,scene.observed_depth):
            name=frame['physical_camera'];path=args.source_field_depth/(name+'.npz')
            if name not in records:
                scene.visibility_depth.append(base)
                continue
            if sha(path)!=records[name]['sha256']:raise ValueError('Changed source visibility depth')
            source=np.load(path);proposal=source[args.source_depth_key];roi=source['roi']
            if proposal.shape!=base.shape or roi.shape!=base.shape or not np.isfinite(proposal).all() or (proposal<0).any():
                raise ValueError('Invalid neural source visibility map')
            # Unknown native depth stays unknown inside its domain; never label
            # a missing neural hit as visible by falling back to another surface.
            scene.visibility_depth.append(np.where(roi,proposal,base).astype(np.float32))
        scene.visibility_receipt_sha256=sha(args.source_field_depth/'complete.json')
        scene.visibility_depth_key=args.source_depth_key
    if args.source_order_calibration is not None:
        if args.stage != 'render': p.error('--source-order-calibration is only a rendering control')
        ordering = json.loads(args.source_order_calibration.read_text())
        by_camera = {f['physical_camera']:f for f in ordering['frames'] if f['file_path'] in ordering['train_filenames']}
        if len(by_camera) != 59 or set(by_camera) != {f['physical_camera'] for f in scene.frames}:
            raise ValueError('Frozen source ordering belongs to a different train split')
        scene.centers = np.stack([cv_pose(by_camera[f['physical_camera']])[:3,3] for f in scene.frames])
        scene.source_order_calibration_sha256 = sha(args.source_order_calibration)
    if args.stage == 'fuse':
        fuse(scene, args.output, args.voxel_length)
    elif args.stage == 'render':
        cameras = json.loads(args.camera_manifest.read_text())['frames'] if args.camera_manifest else None
        render(scene, args.output, args.method, args.baseline, cameras=cameras, layer_style=args.layer_style, matting=args.matting,
               target_depth=args.target_depth, target_depth_key=args.target_depth_key, target_depth_gate=args.target_depth_gate,
               reuse_baseline_depth=args.reuse_baseline_depth)
    else:
        rerank(scene, args.output, args.orientation_weight, args.neural_depth, args.neural_candidate)


if __name__ == '__main__':
    main()
