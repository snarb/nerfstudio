"""Isolated CPU packed-raster gather candidate; original body80 proposal unchanged.

Source caches are immutable GT. Tensor replacements, cache reorders and tracked
PyTorch in-place mutations invalidate packing. Writes through external NumPy or
raw-storage aliases must call ``invalidate()`` explicitly: PyTorch does not expose
those writes through Tensor._version. Image/auxiliary rasters remain on CPU;
small camera-ID and ray-index metadata may reside on a GPU and is copied to CPU
exactly as in the original gather.
"""
import time
import numpy as np
import torch

from marshawn_tuning_sampling import body80_quotas
from native_crop_sampling import NativeCropSamplingAdapter, gather_rasters as original_gather


class PackedRaggedGather:
    """Own one packed copy of each CPU raster target and its camera layout."""
    def __init__(self):
        self._cache = None
        self.builds = 0
        self.last_pack_seconds = 0.
        self.total_pack_seconds = 0.

    def invalidate(self):
        self._cache = None

    @staticmethod
    def _tensor_signature(value):
        if not isinstance(value, torch.Tensor):
            return ('unsupported', id(value))
        return (id(value), value.data_ptr(), value.storage_offset(), tuple(value.shape),
                tuple(value.stride()), value.dtype, value.device, value._version)

    def _signature(self, batch):
        ids = batch.get('image_idx')
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1:
            raise ValueError('Packed CPU gather requires one-dimensional camera IDs')
        targets = []
        for key, values in batch.items():
            if key == 'image_idx' or values is None:
                continue
            if not isinstance(values, list):
                raise ValueError('Packed ragged targets must be lists: '+key)
            targets.append((key, id(values), tuple(self._tensor_signature(value) for value in values)))
        return (id(batch), self._tensor_signature(ids), tuple(ids.detach().cpu().long().tolist()), tuple(targets))

    def _build(self, batch, signature):
        self.invalidate()
        started = time.perf_counter()
        images = batch['image']
        ids = batch['image_idx'].detach().cpu().long()
        if not isinstance(images, list) or not images or len(ids) != len(images):
            raise ValueError('Expected one nonempty native image list and camera IDs')
        if (ids < 0).any() or len(ids.unique()) != len(ids):
            raise ValueError('Expected unique nonnegative dataset camera IDs')
        for image in images:
            if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[-1] != 3 or image.device.type != 'cpu':
                raise ValueError('Expected CPU HWC RGB native images')
        heights = torch.tensor([image.shape[0] for image in images], dtype=torch.long)
        widths = torch.tensor([image.shape[1] for image in images], dtype=torch.long)
        sizes = heights*widths
        offsets = torch.cumsum(sizes, 0)-sizes
        packed = {}
        for key, values in batch.items():
            if key == 'image_idx' or values is None:
                continue
            if not isinstance(values, list) or len(values) != len(images):
                raise ValueError('Expected one raster per camera for '+key)
            first = values[0]
            if not isinstance(first, torch.Tensor) or first.ndim < 2:
                raise ValueError('Expected raster tensor for '+key)
            trailing = tuple(first.shape[2:])
            for value, image in zip(values, images):
                if (not isinstance(value, torch.Tensor) or value.ndim < 2 or value.device.type != 'cpu'
                        or value.shape[:2] != image.shape[:2] or tuple(value.shape[2:]) != trailing
                        or value.dtype != first.dtype):
                    raise ValueError('Misaligned CPU raster target '+key)
            packed[key] = torch.cat([value.reshape(-1, *trailing) for value in values], dim=0)
        sorted_ids, order = ids.sort()
        self._cache = dict(signature=signature, source=batch, packed=packed,
                           sorted_ids=sorted_ids, order=order, heights=heights, widths=widths, offsets=offsets)
        self.builds += 1
        self.last_pack_seconds = time.perf_counter()-started
        self.total_pack_seconds += self.last_pack_seconds

    def cache_info(self):
        cache = self._cache
        return dict(builds=self.builds, last_pack_seconds=self.last_pack_seconds,
                    total_pack_seconds=self.total_pack_seconds,
                    packed_bytes=sum(t.numel()*t.element_size() for t in cache['packed'].values()) if cache else 0,
                    source_camera_order=cache['sorted_ids'][torch.argsort(cache['order'])].tolist() if cache else [])

    def __call__(self, batch, indices, *, keep_full_image=False):
        if isinstance(batch['image'], torch.Tensor):
            return original_gather(batch, indices, keep_full_image=keep_full_image)
        if not isinstance(indices, torch.Tensor) or indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError('Expected camera/y/x indices')
        signature = self._signature(batch)
        if self._cache is None or signature != self._cache['signature']:
            self._build(batch, signature)
        cache = self._cache
        index = indices.detach().cpu().long()
        camera, y, x = index.unbind(-1)
        sorted_position = torch.searchsorted(cache['sorted_ids'], camera.contiguous())
        safe_position = sorted_position.clamp_max(len(cache['sorted_ids'])-1)
        if (sorted_position >= len(cache['sorted_ids'])).any() or (cache['sorted_ids'][safe_position] != camera).any():
            raise ValueError('Sampled camera absent from crop cache')
        position = cache['order'][safe_position]
        if ((y < 0) | (x < 0) | (y >= cache['heights'][position]) | (x >= cache['widths'][position])).any():
            raise ValueError('Sampled pixel outside native crop')
        flat = cache['offsets'][position] + y*cache['widths'][position] + x
        result = {key: values[flat] for key, values in cache['packed'].items()}
        result['indices'] = indices
        if keep_full_image:
            result['full_image'] = batch['image']
        return result


class NativeCropSamplingAdapterV2(NativeCropSamplingAdapter):
    """Original proposal/RNG math; replace only the final list raster gather."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.packed_gather = PackedRaggedGather()

    def _body_draw(self, batch):
        if isinstance(batch['image'], torch.Tensor):
            return super()._body_draw(batch)
        n = int(self.sampler.num_rays_per_batch)
        counts = body80_quotas(n, self.step)
        sources = np.repeat(np.arange(4), counts)
        np.random.shuffle(sources)
        camera = np.random.randint(len(self.rows), size=n)
        flat = np.empty(n, np.int64)
        normalizer = float(np.mean([np.dot(counts/n, r['proposal_means']) for r in self.rows]))
        for c, row in enumerate(self.rows):
            for kind, poolname in ((0, 'core'), (2, 'band'), (3, 'valid')):
                pool = self._pool(row, poolname)
                chosen = np.flatnonzero((camera == c) & (sources == kind))
                flat[chosen] = pool[np.random.randint(len(pool), size=len(chosen))]
            chosen = np.flatnonzero((camera == c) & (sources == 1))
            fas = row['fas']; prob = fas['probabilities']
            if not prob.sum():
                pool = self._pool(row, 'core')
                flat[chosen] = pool[np.random.randint(len(pool), size=len(chosen))]
            else:
                levels = np.searchsorted(np.cumsum(prob), np.random.random(len(chosen))).clip(0, 15)
                for level, cells in enumerate(fas['cells']):
                    if not cells:
                        continue
                    take = chosen[levels == level]
                    selected_cells = np.random.randint(len(cells), size=len(take))
                    for index, cellid in zip(take, selected_cells):
                        pool = cells[cellid]; flat[index] = pool[np.random.randint(len(pool))]
        width = np.asarray([row['core'].shape[1] for row in self.rows], np.int64)[camera]
        indices = torch.from_numpy(np.stack([camera, flat//width, flat % width], -1))
        self.source_hits += np.bincount(sources, minlength=4)
        self.last_counts = counts.tolist()
        return self.packed_gather(batch, indices, keep_full_image=self.sampler.config.keep_full_image), normalizer


def install_sampling(pipeline, data_dir, objective='actor', sampling='body80', width=64):
    if hasattr(pipeline, 'marshawn_sampling_adapter'):
        raise ValueError('Sampling adapter already installed')
    adapter = NativeCropSamplingAdapterV2(pipeline, data_dir, objective, sampling, width)
    pipeline.marshawn_sampling_adapter = adapter
    return adapter
