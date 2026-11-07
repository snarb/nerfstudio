"""Strictly bound runtime source-point gate; original sampler/weights untouched."""
import copy,hashlib,json
from pathlib import Path
import numpy as np

ARMS=('original_mapped','source_margin002')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class MarginPolicy:
    def __init__(self,model,adapter,manifest,expected_sha,arm):
        import torch
        from marshawn_source_point_admission import SourcePointAdmission
        from marshawn_source_margin_admission import GeometricSourcePointAdmission
        from marshawn_readonly_point_selector import compose_normalization
        if arm not in ARMS:raise ValueError('Unknown paired margin arm')
        if adapter is not None and(adapter.objective!='actor'or adapter.sampling!='body80'or adapter.width!=64):raise ValueError('Original actor/body80/W64 sampler required')
        if hasattr(model.field,'_margin_continuation_binding'):raise ValueError('Margin selector already installed')
        path=Path(manifest)
        if sha(path)!=expected_sha:raise ValueError('Margin policy manifest drift')
        meta=json.loads(path.read_text())
        if meta['kind']!='PAIRED_SOURCE_MARGIN002_RUNTIME' or meta['source_checkpoint_sha256']!='340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641':raise ValueError('Wrong source policy')
        if meta['geometric_Linf_margin']!=.002 or meta['numerical_epsilon']!=2e-7 or meta['outside_unknown']!='original_numerical_epsilon':raise ValueError('Margin semantics drift')
        for name,digest in meta['helper_sha256'].items():
            if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Audited selector helper drift')
        source=Path(meta['source_grid'])
        if sha(source)!=meta['source_grid_sha256']:raise ValueError('Source support drift')
        if not getattr(model.field,'_semantic_admission_installed',False):raise ValueError('Original mapped selector absent')
        self.adapter=adapter;self.model=model;self.arm=arm;self.original=model.field._normalize_positions;self.lookup=None
        self.binding=dict(schema_version=1,arm=arm,manifest_sha256=expected_sha,source_grid_sha256=meta['source_grid_sha256'],geometric_Linf_margin=.002 if arm=='source_margin002'else None,numerical_epsilon=2e-7,outside_unknown=meta['outside_unknown'],source_checkpoint_sha256=meta['source_checkpoint_sha256'],helpers_sha256=meta['helper_sha256'])
        if arm=='source_margin002':
            with np.load(source,allow_pickle=False)as z:
                self.lookup=GeometricSourcePointAdmission(torch.as_tensor(z['semantic_gate_pad1'],device=model.device),z['aabb'],margin=.002,epsilon=2e-7,base_class=SourcePointAdmission)
            compose_normalization(model.field,self.lookup)
        self.installed=model.field._normalize_positions
        model.field._margin_continuation_binding=copy.deepcopy(self.binding)

    def verify_installed(self):
        current=self.model.field._normalize_positions
        same=current is self.installed or (hasattr(current,'__func__') and current.__func__ is getattr(self.installed,'__func__',None) and current.__self__ is self.installed.__self__)
        if not same or self.model.field._margin_continuation_binding!=self.binding:raise ValueError('Actual runtime selector changed')

    def normalizer(self,step,count):
        from marshawn_tuning_sampling import body80_quotas
        if self.adapter is None:raise ValueError('Read-only gate has no sampler')
        return float(np.mean([r['proposal_means']for r in self.adapter.rows],axis=0)@(body80_quotas(count,step)/count))

    def state_dict(self):
        self.verify_installed()
        return dict(binding=copy.deepcopy(self.binding),last_normalizer=None if self.adapter is None else self.adapter.last_normalizer,last_counts=None if self.adapter is None else copy.deepcopy(self.adapter.last_counts))

    def load_state_dict(self,state):
        if state['binding']!=self.binding:raise ValueError('Selector/source policy differs across resume')
        self.verify_installed()
        if self.adapter is not None:
            normalizer=state['last_normalizer'];counts=state['last_counts']
            if normalizer is not None:
                count=int(self.adapter.sampler.num_rays_per_batch)
                if abs(normalizer-self.normalizer(self.adapter.step,count))>1e-12:raise ValueError('Original sampler normalizer drift')
                from marshawn_tuning_sampling import body80_quotas
                if counts!=body80_quotas(count,self.adapter.step).tolist():raise ValueError('Original sampler quotas drift')
            self.adapter.last_normalizer=normalizer;self.adapter.last_counts=copy.deepcopy(counts)


def install_checkpoint_margin(model,payload,spec_path):
    """Required inference adapter AFTER caller restores/verifies scientific state.

    Plain semantic eval_setup omits this Python selector; never treat that as
    a valid replay of a treatment checkpoint. No training sampler is installed.
    """
    spec_path=Path(spec_path);spec=json.loads(spec_path.read_text());state=payload.get('margin_fork_runtime')
    if not state or state['spec_sha256']!=sha(spec_path)or state['source_sha256']!=spec['parent_checkpoint_sha256']:raise ValueError('Missing or mismatched strict margin checkpoint lineage')
    policy=MarginPolicy(model,None,spec['margin_manifest'],spec['margin_manifest_sha256'],spec['arm'])
    policy.load_state_dict(state['selector_state'])
    return policy


def install_zero_source_margin(model,payload,checkpoint_sha256,spec_path):
    """Replay an OWN branch-zero selection, explicitly composed with original weights.

    This separate entry does not weaken child-resume checks. The exact immutable
    serialized parent seven states are checked; caller must restore and verify
    the actual model/cache/FR state before calling, as for ordinary child replay.
    """
    from train_marshawn_semantic_late_distortion import tree_digest
    path=Path(spec_path);spec=json.loads(path.read_text())
    if payload.get('margin_fork_runtime')is not None or payload.get('step')!=126316 or checkpoint_sha256!=spec['parent_checkpoint_sha256']:raise ValueError('Exact original zero-source checkpoint required')
    anchor_path=Path(spec['anchor_audit'])
    if sha(anchor_path)!=spec['anchor_audit_sha256']:raise ValueError('Zero-source anchor drift')
    anchor=json.loads(anchor_path.read_text())
    if anchor.get('checkpoint_sha256')!=checkpoint_sha256 or anchor.get('step')!=126316 or set(anchor['fullstate_hashes'])!={'pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture'}:raise ValueError('Exact source seven-state anchor required')
    for key,digest in anchor['fullstate_hashes'].items():
        if tree_digest(payload[key])!=digest:raise ValueError('Zero-source seven-state drift: '+key)
    return MarginPolicy(model,None,spec['margin_manifest'],spec['margin_manifest_sha256'],spec['arm'])
