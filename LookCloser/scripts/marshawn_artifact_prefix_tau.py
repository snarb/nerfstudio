"""Root-qualified196 conditional prefixes; unchanged sealed tau integral/private packets."""
import hashlib
import json
from pathlib import Path
from marshawn_artifact_prefix_sealed_tau import QualifiedClearTau as SealedTau

ARMS={'control':0.,'tau001':.001,'tau01':.01}
QUALIFICATION_SHA='0850f67b215affe471aca1975a07a5eea4952325074751aed7084f2b68cf988f'
SELECTED_SHA='ff6fd24f1443844bc2bbe089b5c0bf35b1331562d26d9380c366d60c7de4a250'
PATCHES={('BA',993,79),('BA',965,160),('AA',949,351),('BA',952,353)}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_targets(path,data,support_manifest):
    import numpy as np
    path,data=Path(path),Path(data)
    if sha(path)!=SELECTED_SHA or sha(path.with_name('root_four_prefix_qualification.json'))!=QUALIFICATION_SHA:
        raise ValueError('Exact root196 prefix qualification required')
    selected=json.loads(path.read_text()); rows=selected['rays']
    if selected['qualification_sha256']!=QUALIFICATION_SHA or len(rows)!=196:raise ValueError('Selected population differs')
    expected={(ca,x+dx,y+dy)for ca,x,y in PATCHES for dx in range(-3,4)for dy in range(-3,4)}
    if {(r['camera'],r['x'],r['y'])for r in rows}!=expected:raise ValueError('Missing/unreviewed/duplicate rays')
    meta=json.loads((data/'transforms.json').read_text());frames={r['physical_camera']:r for r in meta['frames']if r['file_path']in meta['train_filenames']}
    bounds=np.array(json.loads((data/'ready.json').read_text())['aabb']);np.testing.assert_array_equal(bounds,[[-.84,-.21,-.07],[.15,.42,.63]])
    support={r['camera']:r for r in json.loads(Path(support_manifest).read_text())['records']}
    for camera in ('AA','BA'):
        record=support[camera];archive=path.parent/'source_archives'/f'{camera}.npz'
        if sha(archive)!=record['archive_sha256']:raise ValueError('Corrected qualified support changed')
        with np.load(archive,allow_pickle=False)as z:S,U,V=z['support'],z['unknown'],z['valid']
        with np.load(data/'semantics'/f'{camera}.npz',allow_pickle=False)as z:W,valid=z['W64'],z['valid']
        f=frames[camera];pose=np.array(f['transform_matrix']);origin=pose[:3,3]
        for r in rows:
            if r['camera']!=camera:continue
            x,y=r['x'],r['y']
            if not(V[y,x]and valid[y,x]and not S[y,x]and not U[y,x]and W[y,x]==0):raise ValueError('Source qualification changed')
            direction=pose[:3,:3]@np.array([(x+.5-f['cx'])/f['fl_x'],-(y+.5-f['cy'])/f['fl_y'],-1.]);direction/=np.linalg.norm(direction)
            q=(bounds-origin)/direction;near=float(np.max(q.min(0)));box_far=float(np.min(q.max(0)))
            end=min(box_far,(.025-origin[2])/direction[2]-1e-6)
            if not(end>near>=0 and direction[2]<0 and min(origin[2]+near*direction[2],origin[2]+end*direction[2])>.025):raise ValueError('Conditional prefix geometry differs')
            for actual,stored in [(origin,r['origin']),(direction,r['direction']),([near,end],[r['near'],r['far']])]:
                np.testing.assert_allclose(actual,stored,rtol=0,atol=1e-10)
    return {'targets':rows}


class QualifiedPrefixTau(SealedTau):
    """Only population196/coefficient allowlist/control skip differ from sealed math."""
    def __init__(self,model,targets,coefficient):
        import torch
        if type(coefficient)is not float or coefficient not in ARMS.values():raise ValueError('Coefficient')
        if len(targets['targets'])!=196:raise ValueError('Population')
        self.model=model;self.coefficient=coefficient;self.target_sha256=SELECTED_SHA
        self.generator=torch.Generator(device='cpu').manual_seed(73011)
        self.draws=0;self.first_hash=None;self.chain='0'*64;self.last=None
        self.ids_chain='0'*64;self.jitter_chain='0'*64;self.packet_state_before=None;self.packet_state_after=None
        self.o=torch.tensor([r['origin']for r in targets['targets']],device=model.device,dtype=torch.float32)
        self.d=torch.tensor([r['direction']for r in targets['targets']],device=model.device,dtype=torch.float32)
        self.near=torch.tensor([r['near']for r in targets['targets']],device=model.device,dtype=torch.float32)
        self.far=torch.tensor([r['far']for r in targets['targets']],device=model.device,dtype=torch.float32)

    def state_dict(self):
        return dict(super().state_dict(),population=196,qualification_sha256=QUALIFICATION_SHA,
                    ids_chain=self.ids_chain,jitter_chain=self.jitter_chain,
                    packet_state_before=self.packet_state_before,packet_state_after=self.packet_state_after)

    def load_state_dict(self,state):
        if state['population']!=196 or state['qualification_sha256']!=QUALIFICATION_SHA:raise ValueError('Source population changed')
        super().load_state_dict(state)
        for key in ('ids_chain','jitter_chain','packet_state_before','packet_state_after'):setattr(self,key,state[key])

    def draw(self):
        import torch
        self.packet_state_before=hashlib.sha256(self.generator.get_state().numpy().tobytes()).hexdigest()
        ids=torch.randint(196,(64,),generator=self.generator)
        jitter=torch.rand((64,512),generator=self.generator)
        self.packet_state_after=hashlib.sha256(self.generator.get_state().numpy().tobytes()).hexdigest()
        ib,jb=ids.numpy().tobytes(),jitter.numpy().tobytes();raw=ib+jb;digest=hashlib.sha256(raw).hexdigest()
        self.first_hash=self.first_hash or digest;self.chain=hashlib.sha256(bytes.fromhex(self.chain)+raw).hexdigest()
        self.ids_chain=hashlib.sha256(bytes.fromhex(self.ids_chain)+ib).hexdigest()
        self.jitter_chain=hashlib.sha256(bytes.fromhex(self.jitter_chain)+jb).hexdigest();self.draws+=1
        return ids.to(self.o.device),jitter.to(self.o.device),digest

    def loss(self):
        import torch
        if self.coefficient==0.:
            _,_,digest=self.draw();self.last=dict(draw=self.draws,ray_jitter_sha256=digest,coefficient=0.,density_query_skipped=True)
            return torch.zeros((),device=self.model.device,dtype=torch.float32)
        return super().loss()

    def evaluate(self,output,step):
        import torch
        from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
        rng=_capture_rng_state();before=self.generator.get_state().clone();training=self.model.training;self.model.eval()
        try:
            with torch.no_grad():
                # Prefix optical thickness, not full native alpha through observed floor.
                tau=torch.cat([self.tau(torch.arange(i,min(i+64,196),device=self.o.device))for i in range(0,196,64)]).cpu()
        finally:self.model.train(training);_restore_rng_state(rng)
        assert torch.equal(before,self.generator.get_state())
        result=dict(step=step,target_sha256=SELECTED_SHA,qualification_sha256=QUALIFICATION_SHA,
                    tau512_fixed_midpoint=tau.tolist(),auxiliary_generator_advanced=False,
                    scope='196 conditional TRAIN prefixes only; not full-ray alpha or global empty-space truth')
        dest=Path(output)/'prefix_targets'/f'step-{step:09d}.json';dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(result,indent=2)+'\n');return result
