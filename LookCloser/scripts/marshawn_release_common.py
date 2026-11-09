"""Read-only adapters for native NGP release evidence; imports are CPU-safe."""
from pathlib import Path
import json
import numpy as np

from marshawn_occupancy_audit import validate_grid, dilate_one
from probe_marshawn_occupancy import file_sha, integrate


def authoritative_grid(archive, receipt, checkpoint=None):
    """Keep the actual GPU-exported core; never recreate authority with CPUmean."""
    record = json.loads(Path(receipt).read_text())
    if file_sha(archive) != record['grid_archive_sha256']:
        raise ValueError('Grid archive hash mismatch')
    if checkpoint and file_sha(checkpoint) != record['checkpoint_sha256']:
        raise ValueError('Grid checkpoint hash mismatch')
    with np.load(archive, allow_pickle=False) as data:
        active, boxes = validate_grid(data['active'].copy(), data['cascade_aabbs'].copy())
        core = data['core'].copy()
        ema = data['ema'].copy().reshape(active.shape)
        if not np.array_equal(data['resolution'], np.array(active.shape[1:])):
            raise ValueError('Grid resolution metadata mismatch')
        if not np.array_equal(data['world_aabb'].reshape(6), boxes[0]):
            raise ValueError('Grid world/cascade bounds mismatch')
    if not np.isfinite(ema).all():
        raise ValueError('Nonfinite EMA')
    if core.dtype != np.bool_ or core.shape != active.shape:
        raise ValueError('Authoritative core shape/dtype mismatch')
    radius = int(record['radius'])
    if radius not in (0, 1):
        raise ValueError('Only radius0/1 source policies are supported')
    expected = core if radius == 0 else dilate_one(core)
    if not np.array_equal(active, expected):
        raise ValueError('Stored active mask violates declared noncumulative policy')
    if int(record['core_cells']) != int(core.sum()) or int(record['active_cells']) != int(active.sum()):
        raise ValueError('Grid receipt counts disagree')
    step, update = int(record['checkpoint_step']), int(record['core_update_step'])
    if not 0 <= step - update < int(record['native_period']):
        raise ValueError('Stale authoritative core export')
    valid = ema[ema >= 0]
    threshold = min(float(valid.mean()), .01) if len(valid) else 0.
    diagnostic = ema > threshold
    record = dict(record, authority='Exported actual GPU native core',
                  cpu_reconstruction_used=False,
                  adapter_cpu_core_difference=int(np.count_nonzero(diagnostic != core)))
    return core, active, ema, boxes, record


def authoritative_labels(core, active):
    labels = np.zeros(active.shape, np.uint8)
    labels[active & ~core] = 2
    labels[active & core] = 1
    labels[~active & core] = 4
    return labels


def validate_review_path(path, receipt, frames=None, expected_views=24):
    """Bind fixed path to its train-anchor geometry without reading source RGB."""
    from scipy.spatial.transform import Rotation, Slerp
    data=json.loads(Path(path).read_text());record=json.loads(Path(receipt).read_text())
    if file_sha(path)!=record['path_sha256']:
        raise ValueError('Review path hash mismatch')
    entries=data['camera_path']
    if len(entries)!=expected_views or len(record['records'])!=expected_views:
        raise ValueError('Wrong review path view count')
    if (data['render_width']<=0 or data['render_height']<=0 or
            record.get('heldout_rgb_used') is not False or record.get('direct_train_rgb_read') is not False):
        raise ValueError('Invalid path raster/provenance contract')
    by={f['physical_camera']:f for f in frames} if frames is not None else None
    for entry,source in zip(entries,record['records']):
        pose=np.asarray(entry['camera_to_world'],float).reshape(4,4)
        if (not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1],atol=1e-7) or
                not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-6) or
                not np.isclose(np.linalg.det(pose[:3,:3]),1,atol=1e-6)):
            raise ValueError('Nonrigid review path pose')
        if not 0<float(entry['fov'])<180 or not np.isclose(entry['aspect'],data['render_width']/data['render_height']):
            raise ValueError('Review path intrinsics mismatch')
        if by is not None:
            a,b=[np.asarray(by[name]['transform_matrix']) for name in source['anchors']]
            fraction=float(source['fraction'])
            rotation=Slerp([0.,1.],Rotation.from_matrix(np.array([a[:3,:3],b[:3,:3]])))([fraction]).as_matrix()[0]
            if not np.allclose(pose[:3,3],(1-fraction)*a[:3,3]+fraction*b[:3,3],atol=1e-6) or not np.allclose(pose[:3,:3],rotation,atol=1e-6):
                raise ValueError('Novel path no longer matches current train anchor geometry')
    return dict(camera_path_sha256=file_sha(path),path_receipt_sha256=file_sha(receipt),
        views=expected_views,rigid_poses=True,train_anchor_geometry_verified=frames is not None,
        novel_view_gt_available=False,source_path_type=record['path_type'])


def interpolate_intrinsics(first, second, fraction):
    """Interpolate the four actual calibrated pinhole parameters independently."""
    if not 0 <= fraction <= 1 or (first['w'],first['h'])!=(second['w'],second['h']):
        raise ValueError('Invalid anchor fraction/raster')
    result={key:(1-fraction)*float(first[key])+fraction*float(second[key])
            for key in ['fl_x','fl_y','cx','cy']}
    if not np.isfinite(list(result.values())).all() or min(result['fl_x'],result['fl_y'])<=0:
        raise ValueError('Invalid calibrated intrinsics')
    return dict(result,w=int(first['w']),h=int(first['h']))


def load_camera_bundle(bundle, receipt, frames=None):
    """No FOV encoding: verify exact anchor-interpolated K and prepared poses."""
    record=json.loads(Path(receipt).read_text())
    if file_sha(bundle)!=record['bundle_sha256']:
        raise ValueError('Camera bundle hash mismatch')
    with np.load(bundle,allow_pickle=False) as source:
        arrays={key:source[key].copy() for key in ['camera_to_worlds','fx','fy','cx','cy','width','height']}
    count=int(record['views'])
    if arrays['camera_to_worlds'].shape!=(count,3,4) or any(arrays[key].shape!=(count,) for key in ['fx','fy','cx','cy','width','height']):
        raise ValueError('Camera bundle shape mismatch')
    if not all(np.isfinite(value).all() for value in arrays.values()) or min(arrays['fx'].min(),arrays['fy'].min(),arrays['width'].min(),arrays['height'].min())<=0:
        raise ValueError('Invalid camera bundle values')
    rotation=arrays['camera_to_worlds'][:,:,:3]
    if not np.allclose(rotation.transpose(0,2,1)@rotation,np.eye(3),atol=1e-6) or not np.allclose(np.linalg.det(rotation),1,atol=1e-6):
        raise ValueError('Nonrigid camera bundle')
    if frames is not None:
        from audit_marshawn_body_rays import geometry_hash
        if geometry_hash(frames)!=record['train_geometry_sha256']:
            raise ValueError('Camera bundle/current train geometry mismatch')
        by={f['physical_camera']:f for f in frames}
        for index,row in enumerate(record['records']):
            expected=interpolate_intrinsics(*[by[name] for name in row['anchors']],row['fraction'])
            for key,source_key in [('fx','fl_x'),('fy','fl_y'),('cx','cx'),('cy','cy'),('width','w'),('height','h')]:
                if not np.isclose(arrays[key][index],expected[source_key],atol=1e-10,rtol=0):
                    raise ValueError('Camera bundle does not retain actual interpolated K')
    return arrays,dict(record,train_anchor_geometry_verified=frames is not None,
        path_class='Matched interpolated train intrinsics',fov_serialization=False,
        novel_view_ground_truth_available=False)


def point_cells(points, active, boxes):
    """Base cascade point membership; outside points are separate from misses."""
    points = np.asarray(points, float)
    box = boxes[0]
    relative = (points - box[:3]) / (box[3:] - box[:3])
    inside = ((relative >= 0) & (relative < 1)).all(1)
    indices = np.floor(relative * np.array(active.shape[1:])).astype(np.int64)
    occupied = np.zeros(len(points), bool)
    occupied[inside] = active[(0, *indices[inside].T)]
    return indices, inside, occupied


def edge_point_sets(inside, active, core, sigma, subject_ids, limit=12):
    """Actual misses and guard-covered core misses are distinct review buckets."""
    result={}
    for label,mask in [('active_miss',inside & ~active),
                       ('guard_admitted_core_miss',inside & active & ~core)]:
        ids=np.flatnonzero(mask)
        ids=ids[np.argsort(np.asarray(sigma)[ids])[::-1]]
        body=np.isin(ids,subject_ids)
        result[label]=np.r_[ids[body],ids[~body]][:limit]
    return result


def ray_box_interval(origin, direction, box, near=.01, far=1000.):
    origin, direction, box = np.asarray(origin, float), np.asarray(direction, float), np.asarray(box, float).reshape(6)
    entry, exit = float(near), float(far)
    for axis in range(3):
        if direction[axis] == 0:
            if not box[axis] <= origin[axis] < box[axis+3]:
                raise ValueError('Audit ray misses AABB')
        else:
            times = (box[[axis, axis+3]] - origin[axis]) / direction[axis]
            entry, exit = max(entry, float(times.min())), min(exit, float(times.max()))
    if not exit > entry:
        raise ValueError('Audit ray has no positive AABB interval')
    return entry, exit


def native_bounds(model, atol=1e-6):
    """Native VolumetricSampler can clip its own AABB without a collider module."""
    def array(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, float).reshape(2, 3)
        if not np.isfinite(value).all() or (value[1] <= value[0]).any():
            raise ValueError('Invalid live bounds')
        return value
    values = {'model.scene_aabb': array(model.scene_aabb),
              'model.scene_box.aabb': array(model.scene_box.aabb),
              'field.aabb': array(model.field.aabb),
              'occupancy.aabbs[0]': array(model.occupancy_grid.aabbs[0])}
    collider = getattr(model, 'collider', None)
    if collider is not None and getattr(collider, 'scene_box', None) is not None:
        values['collider.scene_box.aabb'] = array(collider.scene_box.aabb)
    elif getattr(model.config, 'enable_collider', True) or getattr(getattr(model, 'sampler', None), 'occupancy_grid', None) is not model.occupancy_grid:
        raise ValueError('Missing validated native occupancy AABB clipping')
    ref = values['model.scene_aabb']
    if any(not np.allclose(value, ref, rtol=0, atol=atol) for value in values.values()):
        raise ValueError('Live bounds disagree')
    return {key: value.tolist() for key, value in values.items()}


def trace_native(model, ray):
    """Capture actual native renderer weights, without requiring LookCloser outputs."""
    from nerfstudio.field_components.field_heads import FieldHeadNames
    field_calls, weight_calls = [], []

    def field_hook(_module, args, output):
        sample = args[0]
        field_calls.append(tuple(value.detach().float().cpu().numpy().reshape(-1).copy()
                                 for value in (sample.frustums.starts, sample.frustums.ends,
                                               output[FieldHeadNames.DENSITY])))

    def weight_hook(_module, args, kwargs):
        weights = kwargs.get('weights', args[0] if args else None)
        if weights is None:
            raise ValueError('Native accumulation renderer weight input unavailable')
        weight_calls.append(weights.detach().float().cpu().numpy().reshape(-1).copy())

    handles = [model.field.register_forward_hook(field_hook),
               model.renderer_accumulation.register_forward_pre_hook(weight_hook, with_kwargs=True)]
    try:
        import torch
        with torch.no_grad():
            result = model(ray)
    finally:
        for handle in handles:
            handle.remove()
    if len(field_calls) != 1 or len(weight_calls) != 1:
        raise ValueError('Expected one native field and accumulation call per audit ray')
    starts, ends, sigma = field_calls[0]
    actual = weight_calls[0]
    alpha, trans, weights = integrate(starts, ends, sigma)
    if actual.shape != weights.shape:
        raise ValueError('Native packed weight shape mismatch')
    error = float(np.max(np.abs(weights-actual), initial=0))
    if error > 1e-4 or abs(float(result['accumulation'].sum()) - float(actual.sum())) > 1e-4:
        raise ValueError('Native weights disagree with captured density/integration')
    return dict(starts=starts, ends=ends, sigma=sigma, alpha=alpha, T=trans,
                weights=actual, weight_max_error=error)


def trace_profiles(model, ray):
    """Stored policy as-is, then all admitted cells, with guaranteed exact restore."""
    import torch
    grid = model.occupancy_grid
    saved = grid.binaries.detach().clone()
    native = trace_native(model, ray)
    try:
        with torch.no_grad():
            grid.binaries.fill_(True)
        all_cells = trace_native(model, ray)
    finally:
        with torch.no_grad():
            grid.binaries.copy_(saved)
    if not torch.equal(saved, grid.binaries):
        raise ValueError('Temporary all-cells trace failed exact mask restore')
    return {'native': native, 'all_cells': all_cells}
