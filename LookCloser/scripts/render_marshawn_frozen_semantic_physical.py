"""Render separately reviewed physical3D selectors using immutable native RGB/sigma.

No image-space erasing/compositing or source RGB/masks affect a rendered pixel.
Contribution partition under native weights is explicitly distinct from every
physical export, which recomputes transmittance after density selection.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
from prepare_marshawn_frozen_semantic_cpu import sha, write


def run(spec_path, execute):
    s = json.loads(spec_path.read_text())
    if not execute:
        print(json.dumps(dict(phase='physical_render3FHD', spec_sha256=sha(spec_path), cameras=['DC', 'HC', 'KA'],
            selectors=['identity_inside', 'soft_probability', 'hard_theta01', 'hard_theta05', 'exactzero_smoothstep01_05'],
            native_updates=0, optimizer_created=False, CUDA_started=False)))
        return
    release = json.loads(Path(s['root_release']).read_text())
    if release.get('reviewer') != 'root' or release.get('physical_render_authorized') is not True or release['spec_sha256'].get(str(spec_path)) != sha(spec_path):
        raise ValueError('Exact root physical-render gate required')
    for path, digest in release['bound_files'].items():
        if sha(path) != digest:
            raise ValueError('Bound source changed: '+path)
    from marshawn_gpu_isolation import require_controller_lease
    lease = require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    import numpy as np
    from PIL import Image
    m = json.loads(Path(s['manifest']).read_text())
    for path in reversed(m['native_python_path']): sys.path.insert(0, path)
    import torch
    import nerfacc
    import yaml
    from nerfstudio.cameras.cameras import Cameras
    from nerfstudio.data.scene_box import SceneBox
    from nerfstudio.engine.trainer import _capture_rng_state, _restore_rng_state
    from marshawn_frozen_semantic_head import FrozenActorGrid, assert_exact_restoration, tensor_digest, capture_native_packed
    from marshawn_frozen_semantic_selectors import selector_memberships
    from run_marshawn_frozen_semantic_probe import capture_health, nested_digest
    if torch.__version__ != '2.4.1+cu121' or torch.version.cuda != '12.1':
        raise ValueError('Historical dev3 native runtime required')
    torch.set_num_threads(2); torch.cuda.set_device(0)
    if torch.cuda.device_count() != 1: raise ValueError('One exact leased device required')
    signal.signal(signal.SIGALRM, lambda signum, stack: (_ for _ in ()).throw(TimeoutError('Physical3FHD render exceeded3600seconds')))
    signal.alarm(3600)
    head_completion = json.loads(Path(s['head_completion']).read_text())
    head_checkpoint = Path(head_completion['checkpoint'])
    if sha(head_checkpoint) != head_completion['checkpoint_sha256']:
        raise ValueError('Semantic head checkpoint differs')
    trained = torch.load(head_checkpoint, map_location='cpu', weights_only=False)
    if trained['iteration'] != s['expected_head_iteration'] or trained['spec_sha256'] != s['head_training_spec_sha256']:
        raise ValueError('Semantic head is not the reviewed first-visual-gate result')
    if trained['label_qualification_sha256'] != sha(s['label_qualification']):
        raise ValueError('Semantic source labels differ')
    initial_probability = float(trained.get('initial_probability', .99))
    if initial_probability != float(s.get('expected_initial_probability', .99)):
        raise ValueError('Semantic initial prior differs from the reviewed physical experiment')
    saved = torch.load(m['native_source']['checkpoint'], map_location='cpu', weights_only=False)
    state = {k[len('_model.'):]: v for k, v in saved['pipeline'].items() if k.startswith('_model.')}
    config = yaml.load(Path(m['native_config']).read_text(), Loader=yaml.Loader)
    model = config.pipeline.model.setup(scene_box=SceneBox(state['scene_aabb'].reshape(2, 3)), num_train_data=63,
        metadata={}, device='cuda:0', grad_scaler=None).cuda()
    model.load_state_dict(state, strict=True); assert_exact_restoration(model, state)
    model.eval(); model.requires_grad_(False)
    from nerfstudio.model_components.ray_samplers import VolumetricSampler
    if not isinstance(model.sampler, VolumetricSampler) or model.sampler.training or model.sampler.get_sigma_fn(None, None) is not None:
        raise ValueError('Physical export requires full occupied-interval traversal with native sigma_fn=None; no opacity early termination')
    from nerfstudio.model_components import renderers as native_renderers
    if native_renderers.BACKGROUND_COLOR_OVERRIDE is not None:
        raise ValueError('Native background override must remain None')
    # The exact bound native RGBRenderer.combine_rgb returns unblended compRGB
    # for historical policy "random": black-equivalent without RNG during
    # rendering. Do not rewrite its configuration or any native tensor.
    if not isinstance(model.renderer_rgb.background_color, str) or model.renderer_rgb.background_color not in ['black', 'random']:
        raise ValueError('Require native black or unblended-random background policy')
    native_digest = tensor_digest(model)
    if native_digest != trained['model_digest']:
        raise ValueError('Head belongs to another native radiance field')
    head = FrozenActorGrid(m['bounds'], m['native_world_to_box'], resolution=128, initial_probability=initial_probability).cuda()
    head.load_state_dict(trained['head'], strict=True); assert_exact_restoration(head, trained['head'])
    head.eval(); head.requires_grad_(False); head_digest = tensor_digest(head)
    if not torch.equal(head.bounds, torch.tensor(m['bounds'], device='cuda')) or not torch.equal(head.world_to_box, torch.tensor(m['native_world_to_box'], device='cuda')):
        raise ValueError('Semantic transform/bounds differ; no second frame application allowed')
    _restore_rng_state(saved['rng_state']); rng_before = nested_digest(_capture_rng_state())
    out = Path(s['output']); out.mkdir(parents=True, exist_ok=False)
    write(out/'initial_health.json', capture_health(torch, lease['expected_physical_uuid']))
    frames = {f['physical_camera']: f for f in json.loads((Path(m['native_data'])/'transforms.json').read_text())['frames']}
    baseline = dict(np.load(s['native_baseline_samples']))
    selector_names = ['identity_inside', 'soft_probability', 'hard_theta01', 'hard_theta05', 'exactzero_smoothstep01_05']
    summaries = []
    initial_logit = torch.tensor(np.log(initial_probability/(1-initial_probability)), dtype=head.logits.dtype, device='cuda')
    grid_initial_count = int((head.logits == initial_logit).sum())
    with torch.inference_mode():
        for name in ['DC', 'HC', 'KA']:
            f = frames[name]
            camera = Cameras(camera_to_worlds=torch.tensor(f['transform_matrix'], dtype=torch.float32)[None, :3],
                fx=f['fl_x'], fy=f['fl_y'], cx=f['cx'], cy=f['cy'], width=f['w'], height=f['h']).to('cuda:0')
            rays = camera.generate_rays(0, keep_shape=True).flatten()
            buffers = {k: np.zeros((len(rays), 4), np.float32) for k in ['native_full', 'native_weight_semantic_partition', *selector_names]}
            newly_visible = {k: np.zeros(len(rays), np.float32) for k in selector_names}
            hidden = dict(samples=0, sigma_positive_samples=0, p_sum=0., p_histogram20=[0.]*20,
                original_tau_sum=0., retained_tau_sum={k: 0. for k in selector_names},
                meaning='Native numerical weightZERO means no direct class gradient; not physicalfree-space. Sharedvertices may learn elsewhere. No support gate/prior change.')
            max_parity = 0.; samples = 0
            for first in range(0, len(rays), s['chunk']):
                rb = rays[first:first+s['chunk']]
                row = capture_native_packed(model, rb)
                xyz = row['xyz']; ids = row['ray_indices']; sigma = row['sigma']
                if not torch.isfinite(sigma).all(): raise ValueError('Nonfinite native density')
                box_xyz = xyz @ head.world_to_box[:3, :3].T+head.world_to_box[:3, 3]
                inside = ((box_xyz >= head.bounds[0]) & (box_xyz <= head.bounds[1])).all(-1)
                probability = head(xyz)  # Original world -> floor transform ONCE, inside head.
                memberships = dict(identity_inside=inside.to(probability.dtype), **selector_memberships(probability))
                native_alpha = row['native_alpha']
                buffers['native_full'][first:first+len(rb)] = torch.cat([row['native_rgb'], native_alpha], -1).cpu().numpy()
                partition_weight = row['weight']*probability
                partition_rgb = nerfacc.accumulate_along_rays(partition_weight, values=row['rgb'], ray_indices=ids, n_rays=len(rb))
                partition_alpha = nerfacc.accumulate_along_rays(partition_weight, values=None, ray_indices=ids, n_rays=len(rb))
                buffers['native_weight_semantic_partition'][first:first+len(rb)] = torch.cat([partition_rgb, partition_alpha], -1).cpu().numpy()
                info = nerfacc.pack_info(ids, len(rb))
                zero_native = (row['weight'] == 0) & inside
                tau = sigma*(row['ends']-row['starts'])
                hidden['samples'] += int(zero_native.sum())
                hidden['sigma_positive_samples'] += int((zero_native & (sigma > 0)).sum())
                if zero_native.any():
                    hp = probability[zero_native]
                    hidden['p_sum'] += float(hp.sum()); hidden['original_tau_sum'] += float(tau[zero_native].sum())
                    histogram = torch.histc(hp, bins=20, min=0, max=1).cpu().tolist()
                    hidden['p_histogram20'] = [a+b for a, b in zip(hidden['p_histogram20'], histogram)]
                for selector, membership in memberships.items():
                    selected_sigma = sigma*membership
                    weights = nerfacc.render_weight_from_density(t_starts=row['starts'], t_ends=row['ends'], sigmas=selected_sigma, packed_info=info)[0]
                    rgb = nerfacc.accumulate_along_rays(weights, values=row['rgb'], ray_indices=ids, n_rays=len(rb))
                    alpha = nerfacc.accumulate_along_rays(weights, values=None, ray_indices=ids, n_rays=len(rb))
                    buffers[selector][first:first+len(rb)] = torch.cat([rgb, alpha], -1).cpu().numpy()
                    newly_visible[selector][first:first+len(rb)] = nerfacc.accumulate_along_rays(weights*zero_native,
                        values=None, ray_indices=ids, n_rays=len(rb)).reshape(-1).cpu().numpy()
                    hidden['retained_tau_sum'][selector] += float((tau*membership)[zero_native].sum())
                samples += len(ids); max_parity = max(max_parity, row['parity_error'])
                if first//s['chunk'] % 128 == 0:
                    write(out/'progress.json', dict(time=time.time(), camera=name, rendered_pixels=first+len(rb), total_pixels=len(rays),
                        native_updates=0, health=capture_health(torch, lease['expected_physical_uuid'])))
            shape = (f['h'], f['w'])
            buffers = {k: v.reshape(*shape, 4) for k, v in buffers.items()}
            pixels = baseline[name+'_pixel_xy']; xy = (pixels[:, 1], pixels[:, 0])
            full_error = float(np.abs(buffers['native_full'][xy][:, :3]-baseline[name+'_full_rgb']).max())
            inside_error = float(np.abs(buffers['identity_inside'][xy][:, :3]-baseline[name+'_inside_rgb']).max())
            if max(full_error, inside_error) > 2e-5: raise ValueError('Saved190k full/identity baseline parity failed')
            view = out/name; view.mkdir()
            for variant, rgba in buffers.items():
                Image.fromarray(np.rint(np.clip(rgba[..., :3], 0, 1)*255).astype('uint8')).save(view/(variant+'.png'))
                Image.fromarray(np.rint(np.clip(rgba[..., 3], 0, 1)*255).astype('uint8')).save(view/(variant+'_alpha.png'))
            np.savez_compressed(view/'physical_and_partition.npz', **buffers)
            np.savez_compressed(view/'newly_visible_native_zero_weight_mass.npz', **{k: v.reshape(*shape) for k, v in newly_visible.items()})
            summaries.append(dict(camera=name, shape=list(shape), samples=samples, native_RGB_parity_max=max_parity,
                saved190k_full_error=full_error, saved190k_inside_error=inside_error, hidden_native_zero_weight=hidden,
                newly_visible_mass_pixels_gt001={k: int((v > .01).sum()) for k, v in newly_visible.items()},
                physical_archive_sha256=sha(view/'physical_and_partition.npz')))
    assert_exact_restoration(model, state)
    if tensor_digest(model) != native_digest or tensor_digest(head) != head_digest or nested_digest(_capture_rng_state()) != rng_before:
        raise ValueError('Frozen native/head/RNG state changed during physical rendering')
    if sha(m['native_source']['checkpoint']) != m['native_source']['checkpoint_sha256'] or sha(head_checkpoint) != head_completion['checkpoint_sha256']:
        raise ValueError('Source checkpoint bytes changed')
    signal.alarm(0)
    write(out/'complete.json', dict(records=summaries, native_model_digest=native_digest, head_digest=head_digest,
        head_checkpoint=str(head_checkpoint), head_checkpoint_sha256=sha(head_checkpoint), head_iteration=trained['iteration'],
        native_model_head_RNG_unchanged=True, native_and_head_optimizer_created=False, source_RGB_targets_or_masks_loaded=False,
        initial_semantic_probability=initial_probability, initial_semantic_logit=float(initial_logit),
        grid_nodes_exactly_initial=grid_initial_count, grid_nodes=head.logits.numel(),
        prior_interpretation='Explicit experimental initialbelief in unobservedregions, not semantic certainty or physicalemptiness.',
        initial_node_caveat='Initial-valuenodes are a prior diagnostic, not a perfect evidence certificate; cancellations may also return initialvalues.',
        no_undeclared_support_gate_or_prior_change=True, physical_selectors=selector_names,
        sampler_sigma_fn_is_None=True, opacity_early_termination_disabled=True,
        native_background_policy=model.renderer_rgb.background_color, background_override_is_None=True,
        background_semantics='BoundnativeRGBRenderer.combine_rgb random returns unblendedRGB, blackequivalent without RNG; no backgroundconfiguration changed.',
        traversal_limitation='Originalnativeoccupiedintervals only, with unchangedcore/EMA/bounds; native-grid-unqueried geometry cannot be added or certified by semantics.',
        contribution_partition='native_weight_semantic_partition uses originalfullvisibility; it is NOT a physicalactor export.',
        visual_review_gate_required=True, automatically_promoted_selector=None))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--spec', type=Path, required=True); p.add_argument('--execute', action='store_true')
    a = p.parse_args(); run(a.spec, a.execute)
