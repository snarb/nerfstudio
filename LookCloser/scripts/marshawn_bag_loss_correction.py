"""Opt-in RGB confidence correction; original masks, proposals and geometry stay intact."""
import copy,hashlib,json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def validate_manifest(path,expected):
    path=Path(path)
    if sha(path)!=expected:raise ValueError('Bag loss manifest changed')
    meta=json.loads(path.read_text())
    if (meta['kind']!='QUALIFIED_TRAIN_FALSE_GREY_RGB_WEIGHT_ONLY' or meta['expected_false_grey_pixels']!=6522
        or meta['uses_EVAL']is not False or meta['fixed_admission_changed']is not False
        or meta['proposal_pools_changed']is not False):raise ValueError('Unreviewed bag correction policy')
    if [r['camera']for r in meta['records']]!=['MA','MB','MC','NA','NB','ND']:raise ValueError('Unexpected qualified camera set')
    for row in meta['records']:
        for key,suffix in [('exclusion','bag_exclusion_HD.npy'),('unknown','bag_unknown_HD.npy')]:
            if row[key]!=row['camera']+'_'+suffix or row[key]not in meta['files_sha256']:raise ValueError('Unbound qualified mask path')
    for name,digest in meta['files_sha256'].items():
        if Path(name).name!=name or sha(path.parent/name)!=digest:raise ValueError('Bag source artifact changed: '+name)
    return meta

def corrected_confidence(row,exclusion,unknown,expected_count):
    """Never alter raw W64, union, pools, shell, black actor or unqualified pixels."""
    weight=row['weight'];grey=row['grey'];black=row['black']
    if exclusion.dtype!=bool or unknown.dtype!=bool or exclusion.shape!=grey.shape or unknown.shape!=grey.shape:raise ValueError('Bag source raster shape/dtype differs')
    if (exclusion&unknown).any():raise ValueError('Qualified interior overlaps unknown shell')
    chosen=exclusion&grey
    if (chosen&black).any()or int(chosen.sum())!=expected_count:raise ValueError('Qualified false-grey membership changed')
    if not np.isfinite(weight).all()or not(weight[chosen]>0).all():raise ValueError('Expected finite positive original bag confidence')
    result=weight.copy();result[chosen]=0
    if not np.array_equal(result[~chosen],weight[~chosen]):raise AssertionError('Unqualified confidence changed')
    return result,chosen

def proposal_means(adapter,row):
    means=[float(row['weight'].flat[adapter._pool(row,k)].mean(dtype=np.float64))for k in ['core','band','valid']]
    fas=row['fas']
    fas_mean=sum(fas['probabilities'][level]*sum(float(row['weight'].flat[cell].mean(dtype=np.float64))for cell in cells)/len(cells)
                 for level,cells in enumerate(fas['cells'])if cells)if fas['probabilities'].sum()else means[0]
    return np.asarray([means[0],fas_mean,means[1],means[2]])

def apply_correction(adapter,path,expected):
    if getattr(adapter,'_bag_loss_correction_installed',False):raise ValueError('Bag correction already installed')
    if adapter.objective!='actor'or adapter.sampling!='body80'or adapter.width!=64:raise ValueError('Exact actor/body80/W64 parent required')
    path=Path(path);meta=validate_manifest(path,expected);records={r['camera']:r for r in meta['records']};applied=[]
    if len(adapter.rows)!=63 or len(adapter.source_records)!=63:raise ValueError('Exact63 TRAIN camera cache required')
    for row,source in zip(adapter.rows,adapter.source_records):
        name=source['camera']
        if name not in records:continue
        record=records[name]
        if source['mask_sha256']!=record['original_semantic_sha256']:raise ValueError('Original TRAIN mask changed')
        exclusion=np.load(path.parent/record['exclusion'],allow_pickle=False);unknown=np.load(path.parent/record['unknown'],allow_pickle=False)
        before=array_sha(row['weight']);old_means=row['proposal_means'].copy()
        weight,chosen=corrected_confidence(row,exclusion,unknown,record['false_grey_pixels'])
        row['weight']=weight;row['proposal_means']=proposal_means(adapter,row)
        applied.append(dict(camera=name,false_grey_pixels=int(chosen.sum()),old_effective_weight_sha256=before,new_effective_weight_sha256=array_sha(weight),changed_mask_sha256=array_sha(chosen),unchanged_core_pool_still_contains=int(np.isin(row['pools']['core'],np.flatnonzero(chosen)).sum()),proposal_means_before=old_means.tolist(),proposal_means_after=row['proposal_means'].tolist()))
    if len(applied)!=6 or sum(r['false_grey_pixels']for r in applied)!=6522:raise ValueError('Incomplete exact bag correction')
    adapter._baseline_means=None
    binding=dict(manifest_sha256=expected,changed_pixels=6522,effective_weights_sha256={s['camera']:array_sha(r['weight'])for s,r in zip(adapter.source_records,adapter.rows)})
    original_state,original_load=adapter.state_dict,adapter.load_state_dict
    def state_dict():return dict(original_state(),bag_loss_correction=copy.deepcopy(binding))
    def load_state_dict(state):
        if state.get('bag_loss_correction')!=binding:raise ValueError('Resume bag confidence policy changed')
        original_load(state)
    adapter.state_dict=state_dict;adapter.load_state_dict=load_state_dict;adapter._bag_loss_correction_installed=True
    receipt=dict(passed=True,manifest_sha256=expected,records=applied,changed_RGB_confidence_pixels=6522,raw_masks_and_W64_unchanged=True,proposal_pools_FAS_unchanged=True,proposal_labels_intentionally_include_qualified_bag=True,unknown_shell_black_hair_and_other_pixels_unchanged=True,admission_or_model_changed=False,private_RNG_created=False,normalizer='Recomputed exact E_q[corrected confidence] under unchanged body80 proposal',state_binding=copy.deepcopy(binding))
    return receipt
