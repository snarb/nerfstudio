"""Variable native-crop collation for the unchanged Marshawn body80 proposal.

Install instead of marshawn_tuning_sampling.install_sampling. Equal-size tensor
batches use the original implementation, including its exact random draws.
"""
import numpy as np
import torch

from marshawn_tuning_sampling import SamplingAdapter, body80_quotas, gather_rasters as tensor_gather


def gather_rasters(image_batch, indices, *, keep_full_image=False):
    """Gather aligned RGB/auxiliary rasters by absolute dataset camera ID."""
    images = image_batch['image']
    if isinstance(images, torch.Tensor):
        return tensor_gather(image_batch, indices, keep_full_image=keep_full_image)
    if not isinstance(images, list) or not images:
        raise ValueError('Expected nonempty list of native crop rasters')
    ids = image_batch['image_idx'].detach().cpu().long()
    if ids.ndim != 1 or len(ids) != len(images) or len(ids.unique()) != len(ids):
        raise ValueError('Expected unique camera ID for every cached crop')
    lookup = {int(c): i for i, c in enumerate(ids.tolist())}
    array = indices.detach().cpu().long()
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError('Expected camera/y/x indices')
    try:
        local = torch.tensor([lookup[int(c)] for c in array[:, 0]], dtype=torch.long)
    except KeyError as error:
        raise ValueError('Sampled camera absent from crop cache') from error
    result = {}
    for key, values in image_batch.items():
        if key == 'image_idx' or values is None:
            continue
        if not isinstance(values, list) or len(values) != len(images):
            raise ValueError('Unsupported auxiliary target '+key+'; refuse silent omission')
        first = values[0]
        if not isinstance(first, torch.Tensor) or first.ndim < 2:
            raise ValueError('Expected raster tensor for '+key)
        trailing = first.shape[2:]
        for value, image in zip(values, images):
            if (not isinstance(value, torch.Tensor) or value.ndim < 2
                    or value.shape[:2] != image.shape[:2] or value.shape[2:] != trailing
                    or value.dtype != first.dtype or value.device != first.device):
                raise ValueError('Misaligned crop auxiliary target '+key)
        output = torch.empty((len(array), *trailing), dtype=first.dtype, device=first.device)
        for position in local.unique().tolist():
            keep = local == position
            y, x = array[keep, 1:].unbind(-1)
            value = values[position]
            if ((y < 0) | (y >= value.shape[0]) | (x < 0) | (x >= value.shape[1])).any():
                raise ValueError('Sampled pixel outside native crop')
            output[keep.to(output.device)] = value[y.to(value.device), x.to(value.device)]
        result[key] = output
    result['indices'] = indices
    if keep_full_image:
        result['full_image'] = images
    return result


class NativeCropSamplingAdapter(SamplingAdapter):
    """Retain body80 pool/FAS/normalizer math while accepting jagged rasters."""

    def _validate_batch(self, batch):
        images = batch['image']
        if isinstance(images, torch.Tensor):
            return super()._validate_batch(batch)
        if not isinstance(images, list) or len(images) != len(self.rows):
            raise ValueError('Require complete native crop cache')
        ids = tuple(batch['image_idx'].detach().cpu().long().tolist())
        if len(ids) != len(self.rows) or set(ids) != set(range(len(self.rows))):
            raise ValueError('Require complete unique dataset camera cache for body80')
        for camera, image in zip(ids, images):
            if (not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[-1] != 3
                    or tuple(image.shape[:2]) != self.rows[camera]['core'].shape):
                raise ValueError('Prepared masks and native crop must share actual raster')
        signature = (ids, tuple(tuple(image.shape) for image in images),
                     tuple(image.data_ptr() for image in images))
        if self._cache_signature is not None and signature != self._cache_signature:
            self._baseline_means = None
        self._cache_signature = signature
        if self.sampling != 'body80':
            raise ValueError('Variable native crop overlay supports body80 only')
        return ids

    def _body_draw(self, batch):
        if isinstance(batch['image'], torch.Tensor):
            return super()._body_draw(batch)
        # Same draw order and probabilities as SamplingAdapter._body_draw.
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
        return gather_rasters(batch, indices, keep_full_image=self.sampler.config.keep_full_image), normalizer


def install_sampling(pipeline, data_dir, objective='actor', sampling='body80', width=64):
    """Drop-in replacement for the archived install_sampling entrypoint."""
    if hasattr(pipeline, 'marshawn_sampling_adapter'):
        raise ValueError('Sampling adapter already installed')
    adapter = NativeCropSamplingAdapter(pipeline, data_dir, objective, sampling, width)
    pipeline.marshawn_sampling_adapter = adapter
    return adapter
