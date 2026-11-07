"""Loss-only confidence policy, separately checkpointed from inherited sampler state."""
import copy,hashlib,json
from pathlib import Path
import numpy as np

ARMS=('original','bag6874','bag6874_delta2','bag6874_delta4')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def structure_sha(adapter):
    h=hashlib.sha256()
    for row in adapter.rows:
        for key in ('core','band','valid','black','grey','actor_union','W32','W64'):h.update(np.ascontiguousarray(row[key]).tobytes())
        for key in ('core','band','valid'):h.update(row['pools'][key].tobytes())
        h.update(row['fas']['probabilities'].tobytes())
        for cells in row['fas']['cells']:
            for cell in cells:h.update(cell.tobytes())
    return h.hexdigest()


def proposal_means(adapter,row):
    means=[float(row['weight'].flat[adapter._pool(row,k)].mean(dtype=np.float64))for k in ('core','band','valid')];fas=row['fas']
    fm=sum(fas['probabilities'][level]*sum(float(row['weight'].flat[cell].mean(dtype=np.float64))for cell in cells)/len(cells)for level,cells in enumerate(fas['cells'])if cells)if fas['probabilities'].sum()else means[0]
    return np.asarray([means[0],fm,means[1],means[2]],np.float64)


class ConfidencePolicy:
    """No sampling wrapper, RNG, buffers, field selectors or proposal changes."""
    def __init__(self,adapter,manifest,expected_sha,arm):
        if arm not in ARMS or adapter.objective!='actor'or adapter.sampling!='body80'or adapter.width!=64:raise ValueError('Exact confidence/body80/W64 policy required')
        path=Path(manifest);assert sha(path)==expected_sha;meta=json.loads(path.read_text());assert meta['kind']=='MATCHED_TRAIN63_RGB_CONFIDENCE_6874_RAW_DELTA2_DELTA4'and meta['uses_EVAL']is False
        records={r['camera']:r for r in meta['records']};assert len(records)==63==len(adapter.rows)
        before=structure_sha(adapter);self.adapter=adapter;self.arm=arm;self.binding=dict(schema_version=1,arm=arm,manifest_sha256=expected_sha,structure_sha256=before,weights_sha256={},proposal_means={})
        for row,source in zip(adapter.rows,adapter.source_records):
            name=source['camera'];record=records[name];assert source['mask_sha256']==record['original_semantic_sha256'];p=path.parent/record['archive'];assert sha(p)==record['archive_sha256']
            with np.load(p,allow_pickle=False)as z:w=z[arm].copy()
            assert w.dtype==np.float32 and w.shape==row['W64'].shape and np.isfinite(w).all()and(w>=0).all()and(w<=1).all()and(w[~row['valid']]==0).all()
            assert array_sha(w)==record['weights_sha256'][arm]
            if arm=='original':np.testing.assert_array_equal(w,row['W64'])
            row['weight']=w;row['proposal_means']=proposal_means(adapter,row)
            np.testing.assert_allclose(row['proposal_means'],np.asarray(record['proposal_means'][arm],np.float64),rtol=0,atol=1e-15)
            self.binding['weights_sha256'][name]=array_sha(w);self.binding['proposal_means'][name]=row['proposal_means'].tolist()
        assert structure_sha(adapter)==before;adapter._baseline_means=None;adapter.last_normalizer=None;adapter.last_counts=None
        self.binding['proposal_means_sha256']=hashlib.sha256(json.dumps(self.binding['proposal_means'],sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def normalizer(self,step,count):
        from marshawn_tuning_sampling import body80_quotas
        return float(np.mean([r['proposal_means']for r in self.adapter.rows],axis=0)@(body80_quotas(count,step)/count))

    def state_dict(self):
        return dict(binding=copy.deepcopy(self.binding),last_normalizer=self.adapter.last_normalizer,last_counts=copy.deepcopy(self.adapter.last_counts))

    def load_state_dict(self,state):
        if state['binding']!=self.binding:raise ValueError('Confidence maps/normalizer/source proposal changed across resume')
        normalizer=state['last_normalizer'];counts=state['last_counts']
        if normalizer is not None:
            count=int(self.adapter.sampler.num_rays_per_batch)
            if abs(normalizer-self.normalizer(self.adapter.step,count))>1e-15:raise ValueError('Checkpoint normalizer differs from exact proposal expectation')
            from marshawn_tuning_sampling import body80_quotas
            if counts!=body80_quotas(count,self.adapter.step).tolist():raise ValueError('Checkpoint confidence quotas changed')
        self.adapter.last_normalizer=normalizer;self.adapter.last_counts=counts
