"""Runtime-only actor weighting/body80 adapter for the actual LookCloser pipeline.

No RGB or alpha targets are edited. The objective is E_q[w*Charbonnier]/E_q[w],
where q is the declared sampling proposal; changing q intentionally changes
emphasis. Its normalizer is analytic and never estimated from the drawn batch.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import torch


def quotas(probabilities, count):
    p = np.asarray(probabilities, np.float64)
    if p.ndim != 1 or (p < 0).any() or not np.isfinite(p).all() or not np.isclose(p.sum(), 1):
        raise ValueError('Normalized finite probabilities required')
    expected = p*count
    out = np.floor(expected).astype(np.int64)
    out[np.argsort(-(expected-out))[:count-int(out.sum())]] += 1
    return out


def body80_quotas(count, step):
    half_core_fas = .5*min(1., max(0., (step-4096)/4096))
    return quotas([2/3*(1-half_core_fas), 2/3*half_core_fas, 2/15, 1/5], count)


def weighted_charbonnier(prediction, target, weights, normalizer):
    if not np.isfinite(normalizer) or normalizer <= 0 or normalizer > 1:
        raise ValueError('Positive fixed proposal expectation in (0,1] required')
    w = weights.to(prediction.device, dtype=torch.float32).reshape(-1)
    if prediction.shape != target.shape or prediction.shape != (len(w), 3):
        raise ValueError('RGB/weight shape mismatch')
    if not torch.isfinite(w).all() or (w < 0).any() or (w > 1).any():
        raise ValueError('Finite confidence in [0,1] required')
    # The historical model adds epsilon=1e-4 directly, NOT its square.
    error = torch.sqrt((prediction.float()-target.to(device=prediction.device,dtype=torch.float32)).square()+1e-4).mean(-1)
    return (error*w).mean()/float(normalizer)


def gather_rasters(image_batch, indices, *, keep_full_image=False):
    """Gather every per-pixel auxiliary tensor with the SAME absolute indices."""
    ids = image_batch['image_idx'].detach().cpu().long().numpy()
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate cached camera IDs')
    lookup = {int(c): i for i, c in enumerate(ids)}
    array = indices.detach().cpu().long().numpy()
    local = np.asarray([lookup[int(c)] for c in array[:, 0]], np.int64)
    image = image_batch['image']
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError('Homogeneous NHWC raster cache required')
    n, h, w = image.shape[:3]
    result = {}
    for key, value in image_batch.items():
        if key == 'image_idx' or value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.ndim < 3 or tuple(value.shape[:3]) != (n,h,w):
            raise ValueError('Unsupported auxiliary target '+key+'; refuse silent omission')
        c = torch.as_tensor(local, device=value.device)
        y = torch.as_tensor(array[:,1], device=value.device)
        x = torch.as_tensor(array[:,2], device=value.device)
        result[key] = value[c,y,x]
    result['indices'] = indices
    if keep_full_image:
        result['full_image'] = image
    return result


class SamplingAdapter:
    def __init__(self, pipeline, data_dir, objective, sampling, width):
        if objective not in ('full', 'actor', 'actor-only', 'actor_only') or sampling not in ('default','body80') or width not in (32,64):
            raise ValueError('Expected full/actor, default/body80 and width32/64')
        self.pipeline = pipeline
        self.sampler = pipeline.datamanager.train_pixel_sampler
        self.objective, self.sampling, self.width = objective, sampling, width
        self.step = 0
        self.rows = []
        self.source_records = []
        self.audit = dict(batches=0, rays=0, black=0, grey=0, band=0, outside=0)
        self.source_hits = np.zeros(4, np.int64)
        self.last_normalizer = None
        self.last_counts = None
        self._baseline_means = None
        self._cache_signature = None
        dataset = pipeline.datamanager.train_dataset
        root = Path(data_dir)
        for path in dataset.image_filenames:
            stem = Path(path).stem
            camera = stem[6:] if stem.startswith('train_') else stem[5:] if stem.startswith('eval_') else stem
            candidates = [root/'actor_masks'/(stem+'.npz'), root/'masks'/(stem+'.npz'), root/'masks'/(camera+'.npz'), root/'semantics'/(camera+'.npz')]
            source = next((p for p in candidates if p.is_file()), None)
            if source is None:
                raise FileNotFoundError('Actor NPZ not found: '+str(candidates))
            with np.load(source, allow_pickle=False) as z:
                row = {k: z[k].copy() for k in ('core','band','black','grey','valid','W32','W64')}
            shape = row['core'].shape
            if len(shape) != 2:
                raise ValueError('Actor NPZ must contain HW rasters')
            for key, val in row.items():
                if val.shape != shape:
                    raise ValueError('Actor raster shapes differ')
                if key.startswith('W'):
                    if not np.issubdtype(val.dtype, np.floating) or not np.isfinite(val).all() or (val<0).any() or (val>1).any():
                        raise ValueError('Invalid confidence '+key)
                    if (val[~row['valid']] != 0).any():
                        raise ValueError('Invalid-source confidence must be zero')
                elif val.dtype != bool:
                    raise ValueError('Boolean actor/support rasters required')
            if (row['core'] & row['band']).any():
                raise ValueError('Core and protective band must be disjoint')
            row['core'] &= row['valid']; row['band'] &= row['valid']
            row['weight'] = row['W'+str(width)]
            row['actor_union'] = row['black'] | row['grey']
            row['pools'] = {k: np.flatnonzero(row[k]) for k in ('core','band','valid')}
            if not len(row['pools']['valid']):
                raise ValueError('Camera has no valid pixels')
            if sampling == 'body80':
                freq_path = root/'lookcloser_frequencies'/(stem+'.pt')
                freq = torch.load(freq_path, map_location='cpu', weights_only=True).float().numpy()
                meta = json.loads(freq_path.with_suffix('.json').read_text())
                if meta.get('value_type') != 'scalar_resolution' or meta['patch_size'] != 8 or meta.get('stride',8) != 8:
                    raise ValueError('Reviewed scalar-resolution patch8/stride8 metadata required')
                if freq.shape != (shape[0]//8, shape[1]//8):
                    raise ValueError('Frequency cells must cover complete actual patches, no resize')
                levels = np.rint(np.log(freq/float(meta.get('min_res',16)))/np.log(float(meta.get('max_res',8192))/float(meta.get('min_res',16)))*15).clip(0,15).astype(int)
                row['fas'] = self._body_cells(row, levels)
                means = []
                for kind in ('core','band','valid'):
                    pool = self._pool(row,kind)
                    means.append(float(row['weight'].flat[pool].mean(dtype=np.float64)))
                fas = row['fas']
                fas_mean = sum(fas['probabilities'][level]*sum(float(row['weight'].flat[cell].mean(dtype=np.float64)) for cell in cells)/len(cells)
                               for level,cells in enumerate(fas['cells']) if cells) if fas['probabilities'].sum() else means[0]
                row['proposal_means'] = np.asarray([means[0],fas_mean,means[1],means[2]])
                self.source_records.append(dict(camera=camera, mask_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    frequency_sha256=hashlib.sha256(freq_path.read_bytes()).hexdigest()))
            else:
                self.source_records.append(dict(camera=camera, mask_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            self.rows.append(row)
        self._original_sample = self.sampler.sample
        self.sampler.sample = self.sample
        self._original_loss = pipeline.model.get_loss_dict
        if objective != 'full':
            if pipeline.model.config.reconstruction_loss_type != 'charbonnier':
                raise ValueError('Actor adapter requires unchanged leader Charbonnier')
            def get_loss(outputs, batch, metrics_dict=None):
                result = self._original_loss(outputs, batch, metrics_dict)
                if 'actor_confidence' in batch:
                    result['rgb_loss'] = weighted_charbonnier(outputs['rgb'],batch['image'],
                        batch['actor_confidence'],batch['actor_proposal_mean_weight'])
                return result
            pipeline.model.get_loss_dict = get_loss

    @staticmethod
    def _body_cells(row, levels):
        h,w = row['core'].shape
        out = []
        original_union = row['actor_union']
        for level in range(16):
            cells = []
            for cy,cx in np.argwhere(levels == level):
                y,x = int(cy)*8,int(cx)*8
                original = original_union[y:y+8,x:x+8]
                core = row['core'][y:y+8,x:x+8]
                if original.mean() < .9 or not core.any():
                    continue
                yy,xx = np.nonzero(core & original)
                if len(yy):
                    cells.append((yy+y)*w+xx+x)
            out.append(cells)
        weights = np.asarray([bool(cells) for cells in out],np.float64)*np.linspace(1,3,16)
        return dict(cells=out, probabilities=weights/weights.sum() if weights.sum() else weights)

    def set_step(self, step):
        if type(step) is not int or step < 0:
            raise ValueError('Nonnegative optimizer step required')
        self.step = step

    def _validate_batch(self, batch):
        images = batch['image']
        if not isinstance(images,torch.Tensor) or images.ndim != 4:
            raise ValueError('Homogeneous full-cache NHWC images required')
        ids = tuple(batch['image_idx'].detach().cpu().long().tolist())
        # Preserve original tensor-FAS cache remapping, including its historical
        # dataset-ID/cache-position ambiguity for shuffled complete caches.
        if len(ids) != len(self.rows) or set(ids) != set(range(len(self.rows))):
            raise ValueError('Require complete unique dataset camera cache for historical FAS parity')
        if any(r['core'].shape != tuple(images.shape[1:3]) for r in self.rows):
            raise ValueError('Prepared masks and training images must share actual raster')
        signature=(ids,tuple(images.shape),batch.get('mask',None).data_ptr() if isinstance(batch.get('mask'),torch.Tensor) else None)
        if self._cache_signature is not None and signature != self._cache_signature:
            self._baseline_means = None
        self._cache_signature=signature
        return ids

    def _initialize_baseline_means(self, batch):
        sampler=self.sampler
        if sampler.config.enable_fas and not sampler.is_initialized:
            sampler._initialize_buckets(self.pipeline.datamanager.train_dataset)
        if sampler.config.training_patch_size != 1 or sampler.config.fas_patch_group_size != 1:
            raise ValueError('Analytic historical expectation requires independent ray FAS')
        # Original collator remaps selected local c through image_idx[c].
        ids=batch['image_idx'].detach().cpu().long().tolist()
        weights=np.stack([self.rows[c]['weight'] for c in ids])
        mask=batch.get('mask')
        if isinstance(mask,torch.Tensor) and not sampler.config.ignore_mask:
            valid=mask.detach().cpu().numpy().reshape(weights.shape).astype(bool)
            uniform_mean=float(weights[valid].mean())
        else:
            uniform_mean=float(weights.mean(dtype=np.float64))
        level_means=[]
        if sampler.config.enable_fas:
            integral=np.pad(weights.astype(np.float64),((0,0),(1,0),(1,0))).cumsum(1).cumsum(2)
            h,w=weights.shape[1:]
            size,stride=sampler.patch_size,sampler.patch_stride
            for level in range(sampler.config.num_levels):
                cells=sampler.buckets[level].cpu().numpy()
                if not len(cells):
                    # Empty-level fallback ignores the validity mask in original FAS.
                    level_means.append(float(weights.mean(dtype=np.float64)));continue
                c,y,x=cells.T;y=y*stride;x=x*stride
                if (y<0).any() or (x<0).any() or (y+size>h).any() or (x+size>w).any():
                    raise ValueError('Out-of-bounds FAS patch requires explicit clamp expectation')
                sums=integral[c,y+size,x+size]-integral[c,y,x+size]-integral[c,y+size,x]+integral[c,y,x]
                level_means.append(float((sums/(size*size)).mean()))
        self._baseline_means=(uniform_mean,np.asarray(level_means))

    def _baseline_normalizer(self,batch):
        if self._baseline_means is None:self._initialize_baseline_means(batch)
        uniform,means=self._baseline_means
        if not self.sampler.config.enable_fas:return uniform
        n=self.sampler.num_rays_per_batch
        fas=int(round(n*self.sampler._active_fas_strength()))
        counts=quotas(self.sampler.probs,fas)
        return float(((n-fas)*uniform+np.dot(counts,means))/n)

    def _pool(self,row,kind):
        pool=row['pools'][kind]
        if not len(pool):pool=row['pools']['core']
        if not len(pool):pool=row['pools']['valid']
        return pool

    def _body_draw(self,batch):
        n=int(self.sampler.num_rays_per_batch)
        counts=body80_quotas(n,self.step)
        sources=np.repeat(np.arange(4),counts)
        np.random.shuffle(sources)
        camera=np.random.randint(len(self.rows),size=n)
        flat=np.empty(n,np.int64)
        normalizer=float(np.mean([np.dot(counts/n,r['proposal_means']) for r in self.rows]))
        for c,row in enumerate(self.rows):
            for kind,poolname in ((0,'core'),(2,'band'),(3,'valid')):
                pool=self._pool(row,poolname)
                chosen=np.flatnonzero((camera==c)&(sources==kind))
                flat[chosen]=pool[np.random.randint(len(pool),size=len(chosen))]
            chosen=np.flatnonzero((camera==c)&(sources==1))
            fas=row['fas'];prob=fas['probabilities']
            if not prob.sum():
                pool=self._pool(row,'core')
                flat[chosen]=pool[np.random.randint(len(pool),size=len(chosen))]
            else:
                levels=np.searchsorted(np.cumsum(prob),np.random.random(len(chosen))).clip(0,15)
                for level,cells in enumerate(fas['cells']):
                    if not cells:continue
                    take=chosen[levels==level]
                    selected_cells=np.random.randint(len(cells),size=len(take))
                    for index,cellid in zip(take,selected_cells):
                        pool=cells[cellid];flat[index]=pool[np.random.randint(len(pool))]
        width=self.rows[0]['core'].shape[1]
        indices=torch.from_numpy(np.stack([camera,flat//width,flat%width],-1))
        self.source_hits+=np.bincount(sources,minlength=4)
        self.last_counts=counts.tolist()
        return gather_rasters(batch,indices,keep_full_image=self.sampler.config.keep_full_image),float(normalizer)

    def sample(self,image_batch,*,commit_sample_count=True,**kwargs):
        self._validate_batch(image_batch)
        if kwargs:raise ValueError('Unsupported sampler options '+str(kwargs))
        if self.sampling == 'default':
            normalizer=self._baseline_normalizer(image_batch) if self.objective!='full' else 1.
            result=self._original_sample(image_batch,commit_sample_count=commit_sample_count)
        else:
            result,normalizer=self._body_draw(image_batch)
            if commit_sample_count:self.sampler.sample_count+=1
        ids=result['indices'].detach().cpu().long().numpy()
        confidence=np.empty(len(ids),np.float32)
        for c in np.unique(ids[:,0]):
            take=ids[:,0]==c;y,x=ids[take,1:].T;row=self.rows[c]
            confidence[take]=row['weight'][y,x]
            for key in ('black','grey','band'):
                self.audit[key]+=int(row[key][y,x].sum())
            self.audit['outside']+=int((~row['actor_union'][y,x]).sum())
        self.audit['batches']+=1;self.audit['rays']+=len(ids)
        self.last_normalizer=normalizer
        if self.objective!='full':
            if not 0<normalizer<=1:raise ValueError('Actor proposal has zero or invalid expected confidence')
            result['actor_confidence']=torch.from_numpy(confidence)
            result['actor_proposal_mean_weight']=normalizer
        return result

    def state_dict(self):
        return dict(schema_version=1,step=self.step,objective=self.objective,sampling=self.sampling,width=self.width,
                    audit=dict(self.audit),source_hits=self.source_hits.tolist(),sample_count=int(self.sampler.sample_count))

    def load_state_dict(self,state):
        if (state['schema_version'],state['objective'],state['sampling'],state['width']) != (1,self.objective,self.sampling,self.width):
            raise ValueError('Sampling state policy mismatch')
        self.set_step(state['step']);self.audit=dict(state['audit']);self.source_hits=np.asarray(state['source_hits'],np.int64)
        self.sampler.sample_count=int(state['sample_count'])

    def audit_state(self):
        return dict(**self.state_dict(),last_normalizer=self.last_normalizer,last_source_counts=self.last_counts,
                    source_names=['core_uniform','person_FAS','protective_band','whole_valid_image'],
                    normalizer='analytic E_q[w]; weighting once; no batch self-normalization',sources=self.source_records,
                    historical_cache_caveat='default preserves original tensor FAS dataset-ID/cache-position remapping; analytic expectation uses actual shuffled cache IDs',
                    cache_order=list(self._cache_signature[0]) if self._cache_signature else None)


def install_sampling(pipeline,data_dir,objective='full',sampling='default',width=64):
    sampling = 'default' if sampling == 'baseline' else sampling
    if hasattr(pipeline,'marshawn_sampling_adapter'):
        raise ValueError('Sampling adapter already installed')
    adapter=SamplingAdapter(pipeline,data_dir,objective,sampling,width)
    pipeline.marshawn_sampling_adapter=adapter
    return adapter
