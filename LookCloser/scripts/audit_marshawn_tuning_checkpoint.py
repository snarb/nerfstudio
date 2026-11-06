"""Independent, read-only audit of frozen LookCloser tuning checkpoints.

prepare: fixed semantic interior/boundary rays and train-only SfM witnesses.
offline: authoritative exported-grid atlas, ray intervals, witness membership.
trace: actual GPU core/halo/all-cells replay plus independent dense integration
and nine subcell density probes in every cell. Defaults to an idle GPU; an
explicit authorized read-only overlap uses a strict memory cap. Never
modifies checkpoint files, live training processes or production source.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def fixed_mask_pixels(mask):
    """Deterministic targets sample silhouette extrema and core, no predictions."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.ndim != 2 or not mask.any():
        raise ValueError('Nonempty boolean semantic mask required')
    yy, xx = np.where(mask)
    x0, x1, y0, y1 = xx.min(), xx.max(), yy.min(), yy.max()
    boundary = mask & ~binary_erosion(mask)
    by, bx = np.where(boundary)
    targets = [('head', .5, 0.), ('feet_left', .2, 1.), ('feet_right', .8, 1.),
               ('left_edge', 0., .45), ('right_edge', 1., .45), ('lower_edge', .5, .8)]
    result = []
    used = set()
    for name, fx, fy in targets:
        scores = ((bx-(x0+fx*(x1-x0)))/max(x1-x0, 1))**2 + ((by-(y0+fy*(y1-y0)))/max(y1-y0, 1))**2
        chosen = int(scores.argmin())
        xy = int(bx[chosen]), int(by[chosen])
        if xy not in used:
            result.append(dict(region=name, x=xy[0], y=xy[1], semantic='actor_boundary'))
            used.add(xy)
    distance = distance_transform_edt(mask)
    y, x = np.unravel_index(distance.argmax(), mask.shape)
    result.append(dict(region='interior', x=int(x), y=int(y), semantic='actor_interior'))
    # A just-outside sample is UNKNOWN 3D geometry, never a free-space label.
    outer = ~mask & (distance_transform_edt(~mask) > 0) & (distance_transform_edt(~mask) <= 3)
    oy, ox = np.where(outer)
    if len(ox):
        chosen = int(((ox-x0)**2 + (oy-(y0+.5*(y1-y0)))**2).argmin())
        result.append(dict(region='outside_unknown', x=int(ox[chosen]), y=int(oy[chosen]), semantic='outside_actor_unknown_geometry'))
    return result


def ray_for_pixel(frame, x, y):
    pose = np.asarray(frame['transform_matrix'], float)
    direction = pose[:3, :3] @ np.array([(x+.5-frame['cx'])/frame['fl_x'], -(y+.5-frame['cy'])/frame['fl_y'], -1.])
    direction /= np.linalg.norm(direction)
    return dict(origin=pose[:3, 3].tolist(), direction=direction.tolist(), near=0.)


def prepare(data, output, witnesses=None):
    from PIL import Image, ImageDraw
    from prepare_marshawn_tuning import verify_inventory, sha, write
    ready = verify_inventory(data)
    frames = json.loads((data / 'transforms.json').read_text())['frames']
    by_camera = {f['physical_camera']: f for f in frames}
    rows = {r['camera']: r for r in ready['records']}
    output.mkdir(parents=True, exist_ok=False)
    pixels, rays = [], []
    for ci, camera in enumerate(('DC', 'HC', 'KA')):
        row = rows[camera]
        image = Image.open(data / row['image']).convert('RGB')
        draw = ImageDraw.Draw(image)
        with np.load(data / 'semantics' / f'{camera}.npz', allow_pickle=False) as src:
            for actor in ('black', 'grey'):
                for pixel in fixed_mask_pixels(src[actor]):
                    pixel.update(camera=ci, camera_name=camera, actor=actor, query_id=len(pixels))
                    pixels.append(pixel)
                    rays.append(ray_for_pixel(by_camera[camera], pixel['x'], pixel['y']))
                    x, y = pixel['x'], pixel['y']
                    draw.ellipse((x-4, y-4, x+4, y+4), outline='red', width=2)
                    draw.text((x+6, y-6), str(pixel['query_id']), fill='yellow')
        image.save(output / f'{camera}_queries.png')
    points = np.empty((0, 3), float)
    if witnesses:
        with np.load(witnesses, allow_pickle=False) as source:
            raw = source['xyz']
            transform = np.asarray(ready['world_to_box'])
            points = raw @ transform[:3, :3].T + transform[:3, 3]
        np.save(output / 'train_witnesses_floor.npy', points)
    request = dict(split='eval', pixels=pixels, rays=rays,
                   ready_sha256=sha(data / 'ready.json'), transforms_sha256=sha(data / 'transforms.json'),
                   camera_order=['DC', 'HC', 'KA'],
                   witnesses_sha256=sha(witnesses) if witnesses else None,
                   witnesses_file='train_witnesses_floor.npy' if witnesses else None,
                   witnesses=int(len(points)), source='Frozen semantic masks and train-only triangulations',
                   limitation='Pixel semantic labels do not certify 3D free space; SfM witnesses have uncertainty.')
    write(output / 'queries.json', request)
    return request


def witness_membership(points, core, active, boxes):
    """All observed points; no implication that unobserved cells are empty."""
    from marshawn_release_common import point_cells
    indices, inside, admitted = point_cells(points, active, boxes)
    _, _, supported = point_cells(points, core, boxes)
    return dict(points=len(points), inside=int(inside.sum()), outside=int((~inside).sum()),
                active_missed_indices=np.flatnonzero(inside & ~admitted).tolist(),
                admitted_noncore_indices=np.flatnonzero(inside & admitted & ~supported).tolist(),
                limitation='Point coverage only; triangulation uncertainty and visibility require photo review.')


def offline(receipt, checkpoint, queries, output):
    from marshawn_tuning_occupancy import load_snapshot, audit_snapshot, atomic_json, sha
    core, active, ema, boxes, record = load_snapshot(receipt, checkpoint)
    request = json.loads(queries.read_text())
    output.mkdir(parents=True, exist_ok=False)
    report = audit_snapshot(receipt, output / 'grid', checkpoint=checkpoint, queries={'rays': request['rays']})
    if request.get('witnesses_file'):
        points = np.load(queries.parent / request['witnesses_file'], allow_pickle=False)
        report['witness_membership'] = witness_membership(points, core, active, boxes)
    report.update(queries_sha256=sha(queries), binary_warmup=record['binary_warmup'],
                  cell_size=((boxes[:, 3:]-boxes[:, :3])/np.array(active.shape[1:])).tolist(),
                  occupancy_conclusion='All-active warmup; halo quality cannot yet be assessed' if record['binary_warmup'] else 'Learned traversal; consult ray traces and source geometry before judging errors')
    atomic_json(output / 'offline.json', report)
    return report


def actual_profiles(native, ema, source_radius, step, binary_warmup, *, receipt, checkpoint):
    """Adapter for old probe: use stored GPU core, NEVER CPU EMA threshold."""
    from marshawn_tuning_occupancy import load_snapshot
    from marshawn_occupancy_audit import dilate_one
    core, active, saved_ema, _, record = load_snapshot(receipt, checkpoint)
    if (not np.array_equal(native, active) or not np.array_equal(np.asarray(ema).reshape(core.shape), saved_ema)
            or source_radius != record['radius'] or step != record['checkpoint_step']):
        raise ValueError('Live grid does not match authoritative checkpoint export')
    if bool(step < binary_warmup) != record['binary_warmup']:
        raise ValueError('Warmup policy mismatch')
    profiles = {'native': active, 'all_cells': np.ones_like(active)}
    if not record['binary_warmup']:
        profiles = {'native': active, 'base': core, 'halo1': dilate_one(core), 'all_cells': np.ones_like(active)}
    return profiles, dict(record, cpu_reconstruction_used=False, source_receipt=str(receipt),
                         halo1_available=not record['binary_warmup'])


def summarize_traces(trace_dir):
    """Distinguish traversal effects from learned multilayer/fog candidates.

    Significant layer heuristic is declared here: 96 uniform depth bins,
    three-bin smoothing, peaks separated by at least eight bins, with >=10% of
    total termination mass in each peak's +/-3-bin neighborhood. This is a
    diagnostic trigger, not ground-truth geometry classification.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    from marshawn_tuning_occupancy import atomic_json
    report = json.loads((trace_dir / 'audit.json').read_text())
    rows = []
    for row in report['rows']:
        profiles = row['profiles']
        full = profiles['uniform2x']
        conv = row['uniform_convergence']
        converged = (conv['opacity_abs_difference'] <= .005 and
                     conv['foreground_rgb_max_abs_difference'] <= 1/255)
        with np.load(trace_dir / full['trace'], allow_pickle=False) as source:
            mid = (source['t_start'].reshape(-1)+source['t_end'].reshape(-1))*.5
            weights = source['weights'].reshape(-1)
            edges = np.linspace(row['near'], row['far'], 97)
            mass, _ = np.histogram(mid, bins=edges, weights=weights)
            smooth = gaussian_filter1d(mass, sigma=1.)
            peaks, _ = find_peaks(smooth, distance=8)
            significant = [int(p) for p in peaks if mass[max(0, p-3):p+4].sum() >= .1*weights.sum()]
            cdf = np.cumsum(weights)
            quantiles = [float(mid[min(np.searchsorted(cdf, q*cdf[-1]), len(mid)-1)])
                         for q in (.1, .5, .9)] if len(mid) and weights.sum() > 0 else []
        native = np.asarray(profiles['native']['integrated_foreground_rgb'])
        reference = np.asarray(full['integrated_foreground_rgb'])
        halo = np.asarray(profiles.get('halo1', profiles['native'])['integrated_foreground_rgb'])
        witness = None
        if 'witness_xyz' in row['pixel']:
            point = np.asarray(row['pixel']['witness_xyz'])
            origin, direction = np.asarray(row['origin']), np.asarray(row['direction'])
            distance = float((point-origin) @ direction)
            with np.load(trace_dir / full['trace'], allow_pickle=False) as source:
                mid = (source['t_start'].reshape(-1)+source['t_end'].reshape(-1))*.5
                weights = source['weights'].reshape(-1)
                band = abs(mid-distance) <= .02
                skipped = ~source['native_occupied'][:, 0]
            witness = dict(witness_t=distance, perpendicular_error=float(np.linalg.norm(point-(origin+distance*direction))),
                           halfwidth=.02, early_termination_mass=float(weights[mid < distance-.02].sum()),
                           witness_band_mass=float(weights[band].sum()),
                           skipped_witness_band_mass=float(weights[band & skipped].sum()))
        rows.append(dict(pixel=row['pixel'], uniform_converged=converged, witness_depth=witness,
                         native_uniform_rgb_max_difference=float(abs(native-reference).max()),
                         halo_uniform_rgb_max_difference=float(abs(halo-reference).max()),
                         native_halo_rgb_max_difference=float(abs(native-halo).max()),
                         skipped_uniform_termination_mass=full['native_culled_weight_per_cascade'],
                         significant_peak_bins=significant, termination_t_quantiles=quantiles,
                         multilayer_candidate=bool(converged and len(significant) >= 2)))
    result = dict(rows=rows, rays=len(rows), converged_rays=sum(r['uniform_converged'] for r in rows),
                  multilayer_candidates=sum(r['multilayer_candidate'] for r in rows),
                  supplemental_geometry_trigger=any(r['multilayer_candidate'] for r in rows),
                  trigger_scope='Converged learned-field multi-layer heuristic, not independently proven scene geometry.',
                  no_training_or_grid_mutation=True)
    atomic_json(trace_dir / 'diagnostic_summary.json', result)
    return result


def require_idle_gpu(index, allow_readonly_overlap=False):
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    by_index = {int(row.split(',')[0]): row.split(',')[1].strip() for row in devices.strip().splitlines()}
    uuid = by_index[index]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if not allow_readonly_overlap and any(row.split(',')[0].strip() == uuid for row in processes.strip().splitlines()):
        raise RuntimeError('Audit GPU is occupied; do not contend with training')
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    return uuid


def trace(args):
    overlap = bool(getattr(args, 'allow_readonly_overlap', False))
    gpu = require_idle_gpu(args.gpu, overlap)
    # Use the exact production implementation, without editing its files.
    sys.path.insert(0, str(args.code_root))
    sys.path.insert(0, str(args.code_root / 'LookCloser' / 'scripts'))
    import torch
    cap = float(getattr(args, 'memory_cap_gib', 10.))
    if not 0 < cap <= 12:
        raise ValueError('Read-only audit requires a positive memory cap at most12GiB')
    torch.cuda.set_per_process_memory_fraction(min(.9, cap*2**30/torch.cuda.get_device_properties(0).total_memory))
    import probe_marshawn_occupancy as probe
    import nerfstudio.utils.eval_utils as eval_utils
    from nerfstudio.cameras.rays import RaySamples, Frustums
    from marshawn_tuning_occupancy import load_snapshot, scan_subcell_density, atomic_json, sha
    torch.set_num_threads(2)
    core, active, _, boxes, _ = load_snapshot(args.receipt, args.checkpoint)
    # Root is a fresh audit directory; old trace program owns its own subdirectory.
    args.output.mkdir(parents=True, exist_ok=False)
    original_profiles, original_setup = probe.occupancy_profiles, eval_utils.eval_setup

    def authoritative(native, ema, radius, step, warmup, cap=.01, mean_mult=1.):
        return actual_profiles(native, ema, radius, step, warmup, receipt=args.receipt, checkpoint=args.checkpoint)

    def setup(*a, **kw):
        cfg, pipeline, checkpoint, step = original_setup(*a, **kw)
        model = pipeline.model
        actual_sh = bool(model.field.correct_sh_directions)
        if actual_sh != bool(model.config.correct_sh_directions):
            raise ValueError('Loaded field/config SH convention mismatch')
        expected_sh = getattr(args, 'expected_sh', None)
        if expected_sh is not None and actual_sh != (expected_sh == 'true'):
            raise ValueError('Loaded field uses unexpected SH convention')
        # This non-tensor field attribute is not necessarily restored by a plain
        # render-only loader. Bind the reviewed ancestry explicitly, no callback.
        strength = 1. if step <= 75940 else .3
        model.config.feature_reweighting_strength = strength
        model.field.feature_reweighting_strength = strength
        request = json.loads(args.queries.read_text())
        dataset = getattr(pipeline.datamanager, request.get('split', 'eval') + '_dataset')
        actual_names = [Path(p).stem.split('_')[-1] for p in dataset.image_filenames]
        if actual_names != request['camera_order']:
            raise ValueError('Actual camera ordering differs from fixed queries')
        # Verify the independent OpenGL geometry used for source triangulations
        # against actual Nerfstudio camera rays before interpreting depth errors.
        checks = []
        for pixel, expected_ray in zip(request['pixels'], request['rays']):
            camera = dataset.cameras[pixel['camera']:pixel['camera']+1].to(model.device)
            ray = camera.generate_rays(0, coords=torch.tensor([[pixel['y']+.5, pixel['x']+.5]], device=model.device), keep_shape=False)
            origin, direction = ray.origins[0].cpu().numpy(), ray.directions[0].cpu().numpy()
            checks.append(dict(origin_error=float(np.max(abs(origin-np.asarray(expected_ray['origin'])))),
                               direction_error=float(np.max(abs(direction-np.asarray(expected_ray['direction']))))))
        if len(checks) != len(request['pixels']) or any(max(r.values()) > 1e-6 for r in checks):
            raise ValueError('Independent prepared-frame rays and actual model rays disagree')
        atomic_json(args.output / 'coordinate_chain.json', dict(camera_order=actual_names,
                    per_ray_errors=checks, tolerance=1e-6, queries_sha256=sha(args.queries),
                    prepared_data_transforms_sha256=sha(args.data / 'transforms.json'),
                    feature_reweighting_strength=strength, correct_sh_directions=actual_sh,
                    config_sha256=sha(args.config), source_root=str(args.code_root),
                    field_source_sha256=sha(args.code_root/'nerfstudio/fields/lookcloser_field.py')))
        # CPU-side adapter precondition already validates hashes; compare live masks
        # before the density scan as well as in the old probe's replay.
        authoritative(model.occupancy_grid.binaries.cpu().numpy(), model.occupancy_grid.occs.cpu().numpy(),
                      int(model.config.occupancy_dilation_radius), step, int(model.config.occupancy_binary_warmup_steps))
        before = probe.digest_model(model)
        def density_fn(xyz):
            positions = torch.as_tensor(xyz, dtype=torch.float32, device=model.device)
            zeros = torch.zeros((len(positions), 1), device=model.device)
            samples = RaySamples(frustums=Frustums(origins=positions, directions=torch.zeros_like(positions),
                                                  starts=zeros, ends=zeros, pixel_area=torch.ones_like(zeros)))
            with torch.inference_mode():
                sigma, _ = model.field.get_density(samples)
            return sigma.float().cpu().numpy().reshape(-1)
        if args.skip_wholegrid:
            scan = dict(skipped=True, reason='Matched ray-only continuation audit; whole-grid scan not requested')
        else:
            maximum, scan = scan_subcell_density(core, active, boxes, density_fn)
            np.savez_compressed(args.output / 'subcell_density.npz', sigma_max=maximum)
        if probe.digest_model(model) != before:
            raise ValueError('Density scan mutated model state')
        scan.update(checkpoint_sha256=sha(checkpoint), gpu_uuid=gpu, source_receipt_sha256=sha(args.receipt),
                    feature_reweighting_strength=strength, correct_sh_directions=actual_sh,
                    authorized_readonly_overlap=overlap, memory_cap_gib=cap, no_speed_claim=True)
        atomic_json(args.output / 'subcell_density.json', scan)
        return cfg, pipeline, checkpoint, step

    old_argv = sys.argv
    try:
        probe.occupancy_profiles = authoritative
        eval_utils.eval_setup = setup
        sys.argv = [probe.__file__, '--config', str(args.config), '--checkpoint', str(args.checkpoint),
                    '--queries', str(args.queries), '--output', str(args.output / 'rays'),
                    '--data', str(args.data), '--uniform-samples', '8192']
        probe.main()
        summarize_traces(args.output / 'rays')
    finally:
        sys.argv = old_argv
        probe.occupancy_profiles = original_profiles
        eval_utils.eval_setup = original_setup


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'offline', 'trace'])
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--witnesses', type=Path)
    parser.add_argument('--queries', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--code-root', type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip-wholegrid', action='store_true')
    parser.add_argument('--allow-readonly-overlap', action='store_true',
                        help='Only with explicit concurrent read-only authorization; no speed claims')
    parser.add_argument('--memory-cap-gib', type=float, default=10.)
    parser.add_argument('--expected-sh', choices=['true', 'false'])
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.data, args.output, args.witnesses)
    elif args.mode == 'offline':
        offline(args.receipt, args.checkpoint, args.queries, args.output)
    else:
        trace(args)
