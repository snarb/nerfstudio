"""Calibrated pinhole AABB overlays with scene-depth occlusion.

Coordinates: OpenGL camera-to-world, -Z forward, half-integer pixel centers.
Depth is distance along a UNIT camera ray, not camera-axis Z. Hidden lines are
omitted by default, including a box bottom buried underneath the scene floor.
"""
from itertools import product

import numpy as np
from PIL import Image, ImageDraw


def validate_rigid_frame(world_to_box=None):
    """A box may be axis aligned in a declared rigid frame (e.g. the floor)."""
    frame = np.eye(4) if world_to_box is None else np.asarray(world_to_box,dtype=np.float64)
    if (frame.shape != (4,4) or not np.isfinite(frame).all()
            or not np.allclose(frame[3],[0,0,0,1],atol=1e-9)
            or not np.allclose(frame[:3,:3].T @ frame[:3,:3],np.eye(3),atol=1e-6)
            or not np.isclose(np.linalg.det(frame[:3,:3]),1,atol=1e-6)):
        raise ValueError('world_to_box must be a proper rigid transform, without scale')
    return frame


def camera_in_box_frame(camera, world_to_box=None):
    frame = validate_rigid_frame(world_to_box)
    pose = np.asarray(camera['transform_matrix'],dtype=np.float64)
    if pose.shape == (3,4):
        pose = np.vstack([pose,[0,0,0,1]])
    return dict(camera,transform_matrix=(frame @ pose).tolist())


def validate_bounds(bounds):
    box = np.asarray(bounds, dtype=np.float64)
    if box.shape != (2, 3) or not np.isfinite(box).all() or (box[1] <= box[0]).any():
        raise ValueError('AABB must contain finite ordered min/max XYZ')
    return box


def box_edges(bounds):
    box = validate_bounds(bounds)
    bits = np.asarray(list(product([0, 1], repeat=3)))
    corners = np.where(bits, box[1], box[0])
    edges = [(i, j) for i in range(8) for j in range(i+1, 8)
             if np.count_nonzero(bits[i] != bits[j]) == 1]
    return corners, edges


def camera_local(points, camera):
    pose = np.asarray(camera['transform_matrix'], dtype=np.float64)
    if pose.shape not in [(3, 4), (4, 4)] or not np.isfinite(pose).all():
        raise ValueError('Expected finite OpenGL camera-to-world pose')
    if not np.allclose(pose[:3,:3].T @ pose[:3,:3], np.eye(3), atol=1e-5):
        raise ValueError('Camera rotation must be orthonormal; no pose scaling')
    if any(abs(camera.get(k, 0)) > 1e-12 for k in ['k1','k2','k3','k4','p1','p2']):
        raise ValueError('Use an undistorted pinhole raster for this overlay')
    return (np.asarray(points)-pose[:3,3]) @ pose[:3,:3]


def project_local(local, camera):
    return np.stack([camera['fl_x']*local[...,0]/-local[...,2]+camera['cx']-.5,
                     -camera['fl_y']*local[...,1]/-local[...,2]+camera['cy']-.5], -1)


def edge_samples(bounds, camera, near=1e-5):
    """Perspective-correct samples at <=0.5 image pixels, with near clipping."""
    corners, edges = box_edges(bounds)
    local = camera_local(corners, camera)
    result = []
    for i, j in edges:
        a, b = local[i].copy(), local[j].copy()
        if -a[2] < near and -b[2] < near:
            continue
        if -a[2] < near:
            a += ((-near-a[2])/(b[2]-a[2]))*(b-a)
        if -b[2] < near:
            b += ((-near-b[2])/(a[2]-b[2]))*(a-b)
        uv = project_local(np.stack([a,b]), camera)
        # Clip in screen space before sampling, so an edge across the near plane
        # cannot allocate an enormous line. Reciprocal depth interpolates linearly.
        delta = uv[1]-uv[0]
        lo, hi = 0., 1.
        for axis, upper in [(0,camera['w']-1),(1,camera['h']-1)]:
            if abs(delta[axis]) < 1e-15:
                if not 0 <= uv[0,axis] <= upper:
                    lo, hi = 1., 0.
            else:
                t0, t1 = (0-uv[0,axis])/delta[axis], (upper-uv[0,axis])/delta[axis]
                lo, hi = max(lo,min(t0,t1)), min(hi,max(t0,t1))
        if hi < lo:
            continue
        length = np.linalg.norm(delta)
        s = np.linspace(lo,hi,max(2,int(np.ceil((hi-lo)*length*2))+1))
        depth = 1/((1-s)/(-a[2])+s/(-b[2]))
        positions = (((1-s)[:,None]*a/(-a[2])+s[:,None]*b/(-b[2]))*depth[:,None])
        result.append(dict(edge=(i,j), uv=uv[0]+s[:,None]*delta,
                           distance=np.linalg.norm(positions,axis=1), arclength=s*length))
    return result


def depth_visibility(distance, surface_distance, tolerance=0.):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Depth tolerance must be finite and nonnegative')
    surface = np.asarray(surface_distance)
    # NaN means unknown (hide), +inf means a valid ray with no opaque surface.
    return ((np.asarray(distance) <= surface+tolerance) & (surface > 0))


def overlay_depth_aware(rgb, bounds, camera, surface_depth, *, tolerance=0.,
                        dash=14., gap=10., width=3, hidden='omit'):
    """Draw independently depth-tested line pixels; never bridge an occluder.

    surface_depth is the distance at accumulated opacity 0.5 (or another declared
    surface estimate). This is an approximate surface overlay, not a depth oracle.
    hidden='ghost' is an explicitly requested x-ray diagnostic, not the default.
    """
    if hidden not in ['omit','ghost'] or dash <= 0 or gap < 0 or width < 1:
        raise ValueError('Invalid line style')
    im = rgb.convert('RGB').copy() if isinstance(rgb,Image.Image) else Image.fromarray(rgb)
    w,h = camera['w'],camera['h']
    depth = np.asarray(surface_depth)
    if im.size != (w,h) or depth.shape != (h,w):
        raise ValueError('Image/depth/camera rasters must match exactly')
    image = np.array(im)
    radius = width//2
    counts = dict(visible_samples=0,hidden_samples=0)
    for line in edge_samples(bounds,camera):
        xy = np.rint(line['uv']).astype(int)
        on_dash = np.mod(line['arclength'],dash+gap)<dash
        for dy in range(-radius,radius+1):
            for dx in range(-radius,radius+1):
                x,y = xy[:,0]+dx,xy[:,1]+dy
                valid=(x>=0)&(x<w)&(y>=0)&(y<h)&on_dash
                x,y,d = x[valid],y[valid],line['distance'][valid]
                visible=depth_visibility(d,depth[y,x],tolerance)
                image[y[visible],x[visible]]=[255,35,35]
                if hidden=='ghost':
                    image[y[~visible],x[~visible]]=(image[y[~visible],x[~visible]]*.8+np.array([120,120,120])*.2).astype(np.uint8)
                if dx==0 and dy==0:
                    counts['visible_samples']+=int(visible.sum())
                    counts['hidden_samples']+=int((~visible).sum())
    return Image.fromarray(image), dict(**counts,hidden_edges=hidden,
        depth_convention='unit-ray distance',depth_tolerance_world=tolerance,
        limitation='Visibility is conditional on the supplied rendered surface depth')
