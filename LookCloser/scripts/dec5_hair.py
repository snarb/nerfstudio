"""Small, explicit geometry/compositing primitives for the DEC5 hair experiment.

Depth is positive OpenCV camera-z in calibration units, never normalized ray
distance. Unknown depth is zero; semantic confidence is NOT alpha or depth GT.
This module has no training, filesystem or production-default side effects.
"""
from __future__ import annotations

import numpy as np


VALIDATION = ('D004_A005_12103C', 'E004_B005_1210I7', 'I004_D005_1210Q7')
TEST = ('F004_B005_1210O9', 'J004_D005_1210TA', 'L004_B005_12106A')


def camera_split(frames):
    names = [f['physical_camera'] for f in frames]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate physical cameras')
    if not set(VALIDATION + TEST).issubset(names):
        raise ValueError('Missing validation/test cameras')
    return {name: ('validation' if name in VALIDATION else 'test' if name in TEST else 'train')
            for name in names}


def mvs_camera_model(frames, payload=None):
    """Avoid OPENCV's unnecessary undistortion/crop for already pinhole RGB."""
    payload = payload or {}
    if any(f.get('camera_model', payload.get('camera_model', 'OPENCV')) not in ('OPENCV', 'PINHOLE') for f in frames):
        raise ValueError('This campaign requires perspective OPENCV/PINHOLE cameras')
    zero = all(abs(float(f.get(k, payload.get(k, 0)))) <= 1e-12
               for f in frames for k in ('k1','k2','k3','k4','p1','p2'))
    return 'PINHOLE' if zero else 'OPENCV'


def cv_pose(frame):
    """OpenCV c2w from the Nerfstudio/OpenGL c2w stored in transforms.json."""
    pose = np.asarray(frame['transform_matrix'], dtype=np.float64).copy()
    pose[:3, 1:3] *= -1
    return pose


def integer_center_pinhole(frame):
    """COLMAP/NS centers at .5 -> Open3D RGBD centers at integer indices."""
    return (frame['fl_x'], frame['fl_y'], frame['cx']-.5, frame['cy']-.5)


def resize_pinhole(frame, width, height):
    """COLMAP keeps full-size sparse cameras while MVS images may be resized."""
    if width <= 0 or height <= 0 or frame['w'] <= 0 or frame['h'] <= 0:
        raise ValueError('Invalid image dimensions')
    sx, sy = width/frame['w'], height/frame['h']
    if abs(sx-sy) > 1e-6:
        raise ValueError('Unexplained anisotropic image resize/crop')
    result = dict(frame, w=width, h=height)
    for key in ('fl_x', 'cx'):
        result[key] *= sx
    for key in ('fl_y', 'cy'):
        result[key] *= sy
    return result


def diverse_witnesses(centers, reference, focus, count=8, min_degrees=1., max_degrees=25.):
    """A nearby seed plus pose-diverse witnesses; no RGB/held-out selection."""
    centers = np.asarray(centers)
    directions = centers-np.asarray(focus)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True).clip(1e-12)
    angles = np.arccos(np.clip(directions @ directions[reference], -1, 1))
    eligible = [int(i) for i in np.argsort(np.linalg.norm(centers-centers[reference],axis=-1),kind='stable')
                if i != reference and np.deg2rad(min_degrees) <= angles[i] <= np.deg2rad(max_degrees)]
    if not eligible:
        return []
    selected = [eligible.pop(0)]
    while eligible and len(selected) < count:
        separation = np.arccos(np.clip(directions[eligible] @ directions[selected].T, -1, 1)).min(-1)
        score = separation-.01*angles[eligible]
        selected.append(eligible.pop(int(score.argmax())))
    return selected


def unproject(uv, depth, frame):
    """Undistorted pixel coordinates; z depth. Distortion must be removed first."""
    if any(abs(float(frame.get(k, 0))) > 1e-10 for k in ('k1', 'k2', 'k3', 'k4', 'p1', 'p2')):
        raise ValueError('unproject requires undistorted intrinsics')
    uv, depth = np.asarray(uv), np.asarray(depth)
    camera = np.stack(((uv[..., 0]-frame['cx'])/frame['fl_x']*depth,
                       (uv[..., 1]-frame['cy'])/frame['fl_y']*depth, depth), -1)
    pose = cv_pose(frame)
    return camera @ pose[:3, :3].T + pose[:3, 3]


def project(points, frame):
    if any(abs(float(frame.get(k, 0))) > 1e-10 for k in ('k1', 'k2', 'k3', 'k4', 'p1', 'p2')):
        raise ValueError('project requires undistorted intrinsics')
    pose = cv_pose(frame)
    xyz = (np.asarray(points)-pose[:3, 3]) @ pose[:3, :3]
    z = xyz[..., 2]
    safe = np.where(z > 0, z, 1)
    uv = np.stack((frame['fl_x']*xyz[..., 0]/safe+frame['cx'],
                   frame['fl_y']*xyz[..., 1]/safe+frame['cy']), -1)
    return uv, z


def ray_to_z(distance, uv, frame):
    norm = np.sqrt(1+((np.asarray(uv)[..., 0]-frame['cx'])/frame['fl_x'])**2
                     +((np.asarray(uv)[..., 1]-frame['cy'])/frame['fl_y'])**2)
    return np.asarray(distance)/norm


def orientations(rgb, scales=(1., 2., 4.)):
    """Undirected strand tangent (cos2theta,sin2theta), anisotropy confidence."""
    import cv2
    x = np.asarray(rgb, np.float32)
    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) if x.ndim == 3 else x
    best = np.zeros(gray.shape, np.float32)
    direction = np.zeros((*gray.shape, 2), np.float32)
    for scale in scales:
        smooth = cv2.GaussianBlur(gray, (0, 0), scale)
        dx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0)/32
        dy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1)/32
        xx = cv2.GaussianBlur(dx*dx, (0, 0), scale*2)
        yy = cv2.GaussianBlur(dy*dy, (0, 0), scale*2)
        xy = cv2.GaussianBlur(dx*dy, (0, 0), scale*2)
        delta = np.sqrt((xx-yy)**2+4*xy**2)
        # Small eigenvector is the tangent, perpendicular to the image gradient.
        tangent = np.stack((yy-xx, -2*xy), -1)/(delta[..., None]+1e-12)
        energy = xx+yy
        q = delta/(energy+1e-12) * np.minimum(energy/.0001, 1)
        use = q > best
        direction[use], best[use] = tangent[use], q[use]
    return direction, best


def orientation_planes(uv, double_angle, frame):
    """Normals to world-space planes of possible 3D tangents at each pixel."""
    angle = .5*np.arctan2(double_angle[..., 1], double_angle[..., 0])
    ray = np.stack(((uv[..., 0]-frame['cx'])/frame['fl_x'],
                    (uv[..., 1]-frame['cy'])/frame['fl_y'], np.ones(angle.shape)), -1)
    tangent = np.stack((np.cos(angle)/frame['fl_x'], np.sin(angle)/frame['fl_y'],
                        np.zeros(angle.shape)), -1)
    normal = np.cross(ray, tangent) @ cv_pose(frame)[:3, :3].T
    return normal/np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)


def orientation_cost(normals, confidence):
    """Scale-normalized minimum eigenvalue, plus count of nonzero witnesses."""
    normals, confidence = np.asarray(normals), np.asarray(confidence)
    matrix = np.einsum('v...i,v...j,v...->...ij', normals, normals, confidence)
    values = np.linalg.eigvalsh(matrix)
    cost = values[..., 0]/np.maximum(values.sum(-1), 1e-12)
    support = (confidence > 0).sum(0)
    constrained = values[..., 1] >= .01*np.maximum(values.sum(-1), 1e-12)
    return np.where((support >= 3) & constrained, cost, np.inf), support


def confidence_override(base, proposed, gate, roi):
    base, proposed = np.asarray(base), np.asarray(proposed)
    take = np.asarray(roi, bool) & np.asarray(gate, bool) & np.isfinite(proposed) & (proposed > 0)
    return np.where(take, proposed, base), take


def composite_layers(base_rgb, foreground_premultiplied, alpha, background_rgb,
                     front_depth, back_depth, confidence, roi):
    """Already-warped linear RGB layers; reject invalid order, apply alpha once."""
    alpha = np.asarray(alpha)
    if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError('Alpha must be finite in [0,1]')
    front, back = np.asarray(front_depth), np.asarray(back_depth)
    valid = (np.asarray(roi, bool) & np.asarray(confidence, bool) & np.isfinite(front)
             & np.isfinite(back) & (front > 0) & (back > front))
    mixed = np.asarray(foreground_premultiplied)+(1-alpha[..., None])*np.asarray(background_rgb)
    valid &= np.isfinite(mixed).all(-1)
    return np.where(valid[..., None], mixed, base_rgb), valid


def volume_depths(t, weights, minimum_mass=.1, first_quantile=.1, peak_separation=2):
    """Diagnostic depths from ordered samples, NOT a certification of hair layers.

    Expected depth is conditional on termination. The unconditioned paper depth
    and total opacity are exported separately. Peaks are local maxima of volume
    weights; the caller still has to obtain independent multiview support.
    """
    t, weights = np.asarray(t), np.asarray(weights)
    if t.shape != weights.shape or np.any(np.diff(t, axis=-1) < 0):
        raise ValueError('Expected ordered samples with matching weights')
    if not np.isfinite(t).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Samples/weights must be finite and weights nonnegative')
    mass = weights.sum(-1)
    raw = (weights*t).sum(-1)
    expected = raw/np.maximum(mass, 1e-12)
    first_index = (weights.cumsum(-1) >= (mass*first_quantile)[..., None]).argmax(-1)
    first = np.take_along_axis(t, first_index[..., None], -1)[..., 0]
    maxima = ((weights >= np.concatenate((np.zeros_like(weights[..., :1]), weights[..., :-1]), -1))
              & (weights > np.concatenate((weights[..., 1:], np.zeros_like(weights[..., :1])), -1)))
    score = np.where(maxima, weights, -1)
    i1 = score.argmax(-1)
    indices = np.arange(t.shape[-1])
    score = np.where(np.abs(indices-i1[..., None]) >= peak_separation, score, -1)
    i2 = score.argmax(-1)
    p1 = np.take_along_axis(t, i1[..., None], -1)[..., 0]
    p2 = np.take_along_axis(t, i2[..., None], -1)[..., 0]
    # Localize a concave log-weight peak between fixed ray-march bin centers.
    previous, following = np.maximum(i1-1, 0), np.minimum(i1+1, t.shape[-1]-1)
    logw = np.log(np.maximum(weights, 1e-30))
    left, center, right = [np.take_along_axis(logw, x[..., None], -1)[..., 0]
                           for x in (previous, i1, following)]
    curvature = left-2*center+right
    interior = (i1 > 0) & (i1 < t.shape[-1]-1) & (curvature < -1e-8)
    offset = np.where(interior, .5*(left-right)/np.where(interior, curvature, 1), 0).clip(-.5, .5)
    tl, tr = [np.take_along_axis(t, x[..., None], -1)[..., 0] for x in (previous, following)]
    localized = p1+offset*np.where(offset >= 0, tr-p1, p1-tl)
    second_valid = score.max(-1) > 0
    valid = mass >= minimum_mass
    return dict(expected=np.where(valid, expected, 0), unconditioned=raw, opacity=mass,
                front=np.where(valid, first, 0), peak1=np.where(valid, p1, 0),
                peak1_subpixel=np.where(valid, localized, 0),
                peak2=np.where(valid & second_valid, p2, 0))


def packed_depth_quantile_reference(starts,ends,weights,indices,nears,fars,quantile=.5,minimum_mass=.1):
    """Conditional volume-weight quantile, respecting native adaptive intervals.

    Spacing is normalized to each ray's [near,far]. Constant density inside
    the crossing interval gives exponential, rather than linear, interpolation.
    Empty rays remain invalid. No uniform-spacing assumption or field refit.
    """
    if not 0<quantile<1:raise ValueError('Depth quantile must lie strictly inside (0,1)')
    if not np.isfinite(minimum_mass) or not 0<=minimum_mass<=1:raise ValueError('Invalid minimum ray mass')
    starts,ends,weights=[np.asarray(x,dtype=np.float64).reshape(-1) for x in (starts,ends,weights)]
    indices=np.asarray(indices,dtype=np.int64).reshape(-1)
    nears,fars=[np.asarray(x,dtype=np.float64).reshape(-1) for x in (nears,fars)]
    if not(len(starts)==len(ends)==len(weights)==len(indices)):raise ValueError('Packed interval binding mismatch')
    result=np.zeros(len(nears),np.float64)
    if not len(indices):return result
    order=np.lexsort((starts,indices));starts,ends,weights,indices=[x[order] for x in (starts,ends,weights,indices)]
    boundaries=np.r_[0,np.flatnonzero(np.diff(indices))+1,len(indices)]
    for begin,end in zip(boundaries[:-1],boundaries[1:]):
        ray=indices[begin];w=np.maximum(weights[begin:end],0);cdf=np.cumsum(w);mass=cdf[-1]
        if not np.isfinite(mass) or mass<=0 or mass<minimum_mass:continue
        target=mass*quantile;k=min(int(np.searchsorted(cdf,target)),len(w)-1)
        previous=cdf[k-1] if k else 0.;trans=max(1-previous,1e-12)
        alpha=np.clip(w[k]/trans,1e-12,1-1e-12);used=np.clip((target-previous)/trans,0,alpha)
        fraction=np.log1p(-used)/np.log1p(-alpha)
        position=starts[begin+k]+fraction*(ends[begin+k]-starts[begin+k])
        result[ray]=nears[ray]+position*(fars[ray]-nears[ray])
    return result


def packed_depth_quantile(starts,ends,weights,indices,nears,fars,quantile=.5,minimum_mass=.1):
    """The established scalar evaluator remains the campaign default."""
    return packed_depth_quantile_reference(starts,ends,weights,indices,nears,fars,quantile,minimum_mass)


def packed_depth_quantile_vectorized(starts,ends,weights,indices,nears,fars,quantile=.5,minimum_mass=.1):
    """Batched equivalent of the reference, with independent FP64 ray CDFs.

    Independent padded rows avoid cross-ray prefix-sum cancellation, including
    zero-weight plateaus exactly at the requested quantile. Bound the temporary
    matrix; very ragged/large inputs retain the scalar reference implementation.
    """
    if not 0<quantile<1:raise ValueError('Depth quantile must lie strictly inside (0,1)')
    if not np.isfinite(minimum_mass) or not 0<=minimum_mass<=1:raise ValueError('Invalid minimum ray mass')
    starts,ends,weights=[np.asarray(x,dtype=np.float64).reshape(-1) for x in (starts,ends,weights)]
    indices=np.asarray(indices,dtype=np.int64).reshape(-1)
    nears,fars=[np.asarray(x,dtype=np.float64).reshape(-1) for x in (nears,fars)]
    if not(len(starts)==len(ends)==len(weights)==len(indices)):raise ValueError('Packed interval binding mismatch')
    result=np.zeros(len(nears),np.float64)
    if not len(indices):return result
    order=np.lexsort((starts,indices));starts,ends,weights,indices=[x[order] for x in (starts,ends,weights,indices)]
    begin=np.r_[0,np.flatnonzero(np.diff(indices))+1];counts=np.diff(np.r_[begin,len(indices)]);maximum=int(counts.max())
    if len(begin)*maximum>1<<22:
        return packed_depth_quantile_reference(starts,ends,weights,indices,nears,fars,quantile,minimum_mass)
    rows=np.repeat(np.arange(len(begin)),counts);columns=np.arange(len(indices))-np.repeat(begin,counts)
    matrix=np.zeros((len(begin),maximum),np.float64);matrix[rows,columns]=np.maximum(weights,0)
    cdf=np.cumsum(matrix,axis=1);mass=cdf[np.arange(len(begin)),counts-1]
    valid=np.flatnonzero(np.isfinite(mass)&(mass>0)&(mass>=minimum_mass))
    if not len(valid):return result
    target=mass[valid]*quantile;k=np.argmax(cdf[valid]>=target[:,None],axis=1)
    previous=np.where(k>0,cdf[valid,np.maximum(k-1,0)],0);trans=np.maximum(1-previous,1e-12)
    alpha=np.clip(matrix[valid,k]/trans,1e-12,1-1e-12);used=np.clip((target-previous)/trans,0,alpha)
    fraction=np.log1p(-used)/np.log1p(-alpha);cross=begin[valid]+k
    position=starts[cross]+fraction*(ends[cross]-starts[cross]);ray=indices[begin[valid]]
    result[ray]=nears[ray]+position*(fars[ray]-nears[ray])
    return result


def packed_layer_depth_quantile(starts,ends,weights,indices,nears,fars,layer_mask,quantile=.5,minimum_mass=.01):
    """Conditional depth of a spatial density layer with actual full-ray transmittance.

    Excluded samples still attenuate the ray. Filtering weights and calling an
    ordinary conditional quantile would incorrectly change within-interval
    interpolation when another layer precedes this one.
    """
    if not 0<quantile<1 or minimum_mass<=0:raise ValueError('Invalid layer quantile or mass threshold')
    starts,ends,weights=[np.asarray(x,dtype=np.float64).reshape(-1) for x in (starts,ends,weights)]
    indices=np.asarray(indices,dtype=np.int64).reshape(-1);mask=np.asarray(layer_mask,dtype=bool).reshape(-1)
    nears,fars=[np.asarray(x,dtype=np.float64).reshape(-1) for x in (nears,fars)]
    if len({len(starts),len(ends),len(weights),len(indices),len(mask)})!=1:raise ValueError('Layer/interval binding mismatch')
    result=np.zeros(len(nears),np.float64);mass=np.zeros(len(nears),np.float64)
    if not len(indices):return result,mass
    order=np.lexsort((starts,indices));starts,ends,weights,indices,mask=[x[order] for x in (starts,ends,weights,indices,mask)]
    boundaries=np.r_[0,np.flatnonzero(np.diff(indices))+1,len(indices)]
    for begin,end in zip(boundaries[:-1],boundaries[1:]):
        ray=indices[begin];full=np.maximum(weights[begin:end],0);total=np.cumsum(full)
        selected=np.where(mask[begin:end],full,0);cdf=np.cumsum(selected);m=cdf[-1];mass[ray]=m
        if not np.isfinite(m) or m<minimum_mass:continue
        target=m*quantile;k=min(int(np.searchsorted(cdf,target)),len(full)-1)
        previous=cdf[k-1] if k else 0.;trans=max(1-(total[k-1] if k else 0.),1e-12)
        alpha=np.clip(full[k]/trans,1e-12,1-1e-12);used=np.clip((target-previous)/trans,0,alpha)
        fraction=np.log1p(-used)/np.log1p(-alpha)
        position=starts[begin+k]+fraction*(ends[begin+k]-starts[begin+k])
        result[ray]=nears[ray]+position*(fars[ray]-nears[ray])
    return result,mass
