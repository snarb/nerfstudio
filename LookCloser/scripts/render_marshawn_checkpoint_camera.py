"""Read-only native camera render from a sealed source and exact checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    # Source imports precede helper imports; never use the working tree model.
    source = Path(spec['source'])
    sys.path[:0] = [str(source/'code'), str(source/'code/LookCloser/scripts')]
    from marshawn_gpu_isolation import require_controller_lease, verify_cuda
    lease = require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    assert lease['expected_physical_uuid'] == spec['GPU_uuid']
    import numpy as np
    import torch
    from PIL import Image
    from nerfstudio.engine.trainer import _capture_rng_state, _restore_rng_state
    from nerfstudio.utils.eval_utils import eval_setup

    verify_cuda(lease['expected_physical_uuid'], torch)
    torch.cuda.set_per_process_memory_fraction(10 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Camera render deadline')))
    signal.alarm(900)
    out = Path(spec['output']); out.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        tmp = out/(name+'.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(out/name)

    manifest = json.loads((source/'source_manifest.json').read_text())
    assert sha(source/'source_manifest.json') == spec['source_manifest_sha256']
    for name, digest in manifest['files_sha256'].items():
        assert sha(source/'code'/name) == digest, name
    checkpoint = Path(spec['checkpoint'])
    assert sha(checkpoint) == spec['checkpoint_sha256']
    assert sha(spec['config']) == spec['config_sha256']
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['step'] == spec['step']

    def tree_hash(value):
        h = hashlib.sha256()
        def visit(v):
            if torch.is_tensor(v):
                v = v.detach().cpu().contiguous()
                h.update(str((str(v.dtype), tuple(v.shape))).encode()); h.update(v.numpy().tobytes())
            elif isinstance(v, dict):
                for k in sorted(v, key=str): h.update(str(k).encode()); visit(v[k])
            elif isinstance(v, (list, tuple)):
                for item in v: visit(item)
            elif isinstance(v, np.ndarray): h.update(str((v.dtype, v.shape)).encode()); h.update(v.tobytes())
            else: h.update(repr(v).encode())
        visit(value)
        return h.hexdigest()

    expected = tree_hash(payload['pipeline'])
    def configure(config):
        config.eval_checkpoint = checkpoint
        return config
    config, pipe, actual, step = eval_setup(Path(spec['config']), test_mode='inference', update_config_callback=configure)
    model = pipe.model; pipe.eval(); pipe.requires_grad_(False); model.current_train_step = step
    assert Path(actual).resolve() == checkpoint.resolve() and step == spec['step']
    assert tree_hash(pipe.state_dict()) == expected, 'Checkpoint pipeline restoration differs'
    assert model.field.feature_reweighting_strength == .3
    assert model.config.max_steps_per_ray == 1024 and model.config.occupancy_eval_dilation_radius == 0
    assert model.config.eval_num_rays_per_chunk == 2048
    # Populate only the nonpersistent deterministic feature-weight cache.
    with torch.inference_mode():
        model.field._get_discrete_feature_weights(torch.arange(model.field.num_levels, device=model.device, dtype=model.freq_grid.grid.dtype).unsqueeze(-1))
    rng = _capture_rng_state(); rng_digest = tree_hash(rng)
    lut_digest = tree_hash((model.field._feature_weight_lut, model.field._feature_weight_lut_strength))
    dataset = pipe.datamanager.eval_dataset
    matches = [i for i, p in enumerate(dataset.image_filenames) if Path(p).stem.split('_')[-1] == spec['camera']]
    assert len(matches) == 1
    camera = dataset.cameras[matches[0]:matches[0]+1].to(model.device)
    assert [int(camera.width.item()), int(camera.height.item())] == [1920, 1012]
    write('progress.json', dict(phase='source_HD_parity', time=time.time(), checkpoint_sha256=spec['checkpoint_sha256']))
    with torch.inference_mode():
        reference_output = model.get_outputs_for_camera(camera)
    reference_pixels = np.rint(reference_output['rgb'].clamp(0, 1).cpu().numpy()*255).astype(np.uint8)
    archived = np.asarray(Image.open(spec['reference_RGB']).convert('RGB'))
    delta = np.abs(reference_pixels.astype(np.int16)-archived.astype(np.int16))
    parity = dict(max_RGB8_code_difference=int(delta.max()), changed_channels=int((delta != 0).sum()), changed_fraction=float((delta != 0).mean()))
    write('source_HD_parity.json', parity)
    assert parity['max_RGB8_code_difference'] <= 1, 'Saved native source RGB parity failed'
    del reference_output
    camera.rescale_output_resolution(spec['scale'])
    assert [int(camera.width.item()), int(camera.height.item())] == spec['raster']
    write('progress.json', dict(phase='native_4K_camera_render', raster=spec['raster'], time=time.time()))
    with torch.inference_mode():
        rendered = model.get_outputs_for_camera(camera)
    assert torch.isfinite(rendered['rgb']).all() and torch.isfinite(rendered['accumulation']).all()
    rgb = np.rint(rendered['rgb'].clamp(0, 1).cpu().numpy()*255).astype(np.uint8)
    alpha = np.rint(rendered['accumulation'][..., 0].clamp(0, 1).cpu().numpy()*65535).astype(np.uint16)
    assert list(rgb.shape[:2]) == spec['raster'][::-1]
    Image.fromarray(rgb).save(out/(spec['camera']+'_4K.png'))
    Image.fromarray(alpha).save(out/(spec['camera']+'_4K_alpha16.png'))
    _restore_rng_state(rng)
    assert tree_hash(pipe.state_dict()) == expected
    assert tree_hash(_capture_rng_state()) == rng_digest
    assert tree_hash((model.field._feature_weight_lut, model.field._feature_weight_lut_strength)) == lut_digest
    assert sha(checkpoint) == spec['checkpoint_sha256']
    signal.alarm(0)
    write('complete.json', dict(passed=True, spec_sha256=sha(args.spec), checkpoint_sha256=sha(checkpoint), step=step,
        source_pipeline_digest=expected, camera=spec['camera'], raster=spec['raster'], native_new_camera_rays=True,
        image_resizing_used=False, optimizer_updates=0, pipeline_unchanged=True, RNG_restored=True,
        nonpersistent_LUT_unchanged=True, source_HD_parity=parity,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        files={p.name:sha(p) for p in out.glob('*.png')}))


if __name__ == '__main__':
    main()
