"""Quarter fork: sealed tau integral and randint/jitter; separate approved populations."""
import hashlib
import json
from pathlib import Path
from marshawn_clear_tau import QualifiedClearTau as SealedTau

TARGET_SHA = '7b60c59e5cc6b175bf526aadb12c97491630ccab07fb5f79d6c5538ada6076b3'
OLD_REVIEW = 'fe417ec5d971234ae368362370d02882476a731bf05df649ae2b5ae9537b871b'
NEW_REVIEW = '888f565eacf85a7675ceb20ff607df39bb5aaaf825c1c5ae0088c677ae3dc0d2'
COUNTS = {'AA':1872,'BA':2689,'HA':2808,'EA':1116,'JA':1188,'LA':558,'NA':250}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate_targets(path, expected_sha, data, support_manifest, population):
    import numpy as np
    if population not in ('old7369','seven10481') or expected_sha != TARGET_SHA or digest(path)!=TARGET_SHA:
        raise ValueError('Only exact reviewed10481 universe and explicit7369/10481 population')
    path,data=Path(path),Path(data)
    for name,expected in [('old_root_review.json',OLD_REVIEW),('additional_root_review.json',NEW_REVIEW)]:
        if digest(path.parent/name)!=expected:raise ValueError('Source qualification changed')
    target=json.loads(path.read_text());tf=json.loads((data/'transforms.json').read_text())
    frames={f['physical_camera']:f for f in tf['frames'] if f['file_path'] in tf['train_filenames']}
    ready=json.loads((data/'ready.json').read_text());box=np.asarray(ready['aabb'])
    if not np.array_equal(target['AABB'],box) or not np.array_equal(target['world_to_box'],ready['world_to_box']):
        raise ValueError('Accepted floor coordinate or AABB changed')
    qualification=json.loads((path.parent/'source_qualification.json').read_text())
    source={r['camera']:r for r in json.loads(Path(support_manifest).read_text())['records']}
    rows=target['targets']
    if len(rows)!=10481 or len({(r['camera'],r['x'],r['y']) for r in rows})!=10481:
        raise ValueError('Missing/duplicate target')
    if {r['camera'] for r in rows[:7369]}!={'AA','BA','HA'}:raise ValueError('Old population prefix changed')
    for record in qualification['per_camera']:
        ca=record['camera'];f=frames.get(ca)
        if f is None or ca in {'DC','HC','KA'}:raise ValueError('NonTRAIN target')
        archive=path.parent/'source_archives'/f'{ca}.npz';semantic=data/'semantics'/f'{ca}.npz'
        for p,h in [(archive,record['support_archive_sha256']),(semantic,record['semantic_sha256']),
                    (data/f['file_path'],record['source_RGB_sha256'])]:
            if digest(p)!=h:raise ValueError('Source-qualified raster changed')
        if source[ca]['archive_sha256']!=record['support_archive_sha256']:raise ValueError('Admission/source archive mismatch')
        with np.load(archive,allow_pickle=False) as z:support,unknown,valid=z['support'],z['unknown'],z['valid']
        with np.load(semantic,allow_pickle=False) as z:trainvalid,w,core,black,grey=z['valid'],z['W64'],z['core'],z['black'],z['grey']
        selected=[r for r in rows if r['camera']==ca]
        if len(selected)!=COUNTS[ca]:raise ValueError('Camera population changed')
        m=np.asarray(f['transform_matrix']);o=m[:3,3]
        for r in selected:
            x,y=r['x'],r['y']
            if not(0<=x<1920 and 0<=y<1012 and valid[y,x] and trainvalid[y,x] and not support[y,x]
                   and not unknown[y,x] and w[y,x]==0 and not(core[y,x] or black[y,x] or grey[y,x])):
                raise ValueError('Actor/UNKNOWN/invalid target')
            d=m[:3,:3]@np.array([(x+.5-f['cx'])/f['fl_x'],-(y+.5-f['cy'])/f['fl_y'],-1.]);d/=np.linalg.norm(d)
            q=(box-o)/d;near=max(np.minimum(q[0],q[1]).max(),.01);far=np.maximum(q[0],q[1]).min();floor=-o[2]/d[2]
            if not(np.isfinite([near,far,floor]).all() and far>near and d[2]<0 and min(o[2]+near*d[2],o[2]+far*d[2])>.10 and floor>far):raise ValueError('Ray geometry rejected')
            for actual,expected in [(o,r['origin']),(d,r['direction']),([near,far,floor],[r['near'],r['far'],r['floor_t']])]:
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10)
    return target


class QualifiedClearTau(SealedTau):
    """Own generator; original randint(population), jitter and integral unchanged."""
    def __init__(self,model,targets,sha256,coefficient,population):
        import torch
        if type(coefficient)is not float or coefficient not in (0.,.001):raise ValueError('Coefficient')
        if population not in ('old7369','seven10481'):raise ValueError('Population')
        if len(targets['targets'])!=10481:raise ValueError('Exact10481 evaluation universe required')
        if coefficient==0. and population!='seven10481':raise ValueError('Control consumes10481 packets')
        super().__init__(model,{'targets':targets['targets'][:243]},sha256,coefficient)
        self.population=7369 if population=='old7369' else 10481
        self.population_name=population;self.jitter_chain='0'*64;self.ids_chain='0'*64
        self.packet_state_before=None;self.packet_state_after=None
        # All10481 geometry arrays are also used by identical evaluation in every arm.
        self.o=torch.tensor([r['origin'] for r in targets['targets']],device=model.device,dtype=torch.float32)
        self.d=torch.tensor([r['direction'] for r in targets['targets']],device=model.device,dtype=torch.float32)
        self.near=torch.tensor([r['near'] for r in targets['targets']],device=model.device,dtype=torch.float32)
        self.far=torch.tensor([r['far'] for r in targets['targets']],device=model.device,dtype=torch.float32)

    def state_dict(self):
        return dict(super().state_dict(),population=self.population,population_name=self.population_name,
                    jitter_chain=self.jitter_chain,ids_chain=self.ids_chain,
                    packet_state_before=self.packet_state_before,packet_state_after=self.packet_state_after)

    def load_state_dict(self,state):
        if state.get('population')!=self.population or state.get('population_name')!=self.population_name:
            raise ValueError('Private population changed during resume')
        super().load_state_dict(state)
        for k in ('jitter_chain','ids_chain','packet_state_before','packet_state_after'):setattr(self,k,state[k])

    def draw(self):
        import torch
        self.packet_state_before=hashlib.sha256(self.generator.get_state().numpy().tobytes()).hexdigest()
        ids=torch.randint(self.population,(64,),generator=self.generator)
        jitter=torch.rand((64,512),generator=self.generator)
        self.packet_state_after=hashlib.sha256(self.generator.get_state().numpy().tobytes()).hexdigest()
        ib,jb=ids.numpy().tobytes(),jitter.numpy().tobytes();raw=ib+jb;h=hashlib.sha256(raw).hexdigest()
        self.first_hash=self.first_hash or h;self.chain=hashlib.sha256(bytes.fromhex(self.chain)+raw).hexdigest()
        self.ids_chain=hashlib.sha256(bytes.fromhex(self.ids_chain)+ib).hexdigest()
        self.jitter_chain=hashlib.sha256(bytes.fromhex(self.jitter_chain)+jb).hexdigest();self.draws+=1
        return ids.to(self.o.device),jitter.to(self.o.device),h

    def evaluate(self,pipe,targets,output,step):
        import torch
        from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
        rng=_capture_rng_state();private=self.generator.get_state().clone();training=self.model.training
        self.model.eval();rows=[];order=[Path(p).stem.split('_')[-1] for p in pipe.datamanager.train_dataset.image_filenames]
        cameras=pipe.datamanager.train_dataset.cameras.to(self.model.device)
        domains={'common243':[],'old7126':[],'new3112':[]};centers={'AA':(961,239),'BA':(954,175),'HA':(1067,113)}
        for i,r in enumerate(targets['targets']):
            center=centers.get(r['camera']);common=center is not None and abs(r['x']-center[0])<=4 and abs(r['y']-center[1])<=4
            domains['common243' if common else 'old7126' if i<7369 else 'new3112'].append(i)
        if {k:len(v) for k,v in domains.items()}!={'common243':243,'old7126':7126,'new3112':3112}:raise ValueError('Evaluation domain membership')
        try:
            with torch.no_grad():
                for ca in COUNTS:
                    ids=[i for i,r in enumerate(targets['targets']) if r['camera']==ca];alphas=[];taus=[]
                    for start in range(0,len(ids),128):
                        chunk=ids[start:start+128];q=[targets['targets'][i] for i in chunk]
                        coords=torch.tensor([[r['y']+.5,r['x']+.5] for r in q],device=self.model.device)
                        rays=cameras[order.index(ca):order.index(ca)+1].generate_rays(0,coords=coords,keep_shape=False)
                        alphas.append(self.model(rays)['accumulation'].flatten().cpu())
                        taus.append(self.tau(torch.tensor(chunk,device=self.o.device)).cpu())
                    alpha,tau=torch.cat(alphas),torch.cat(taus);parts={}
                    for domain,members in domains.items():
                        membership=set(members);mask=torch.tensor([i in membership for i in ids])
                        if not bool(mask.any()):continue
                        aa,tt=alpha[mask],tau[mask]
                        parts[domain]=dict(pixels=int(mask.sum()),alpha_mean=float(aa.mean()),alpha_p95=float(torch.quantile(aa,.95)),tau_mean=float(tt.mean()),tau_p95=float(torch.quantile(tt,.95)))
                    rows.append(dict(camera=ca,indices=ids,actual_native_alpha=alpha.tolist(),tau512_fixed_midpoint=tau.tolist(),domains=parts))
        finally:
            self.model.train(training);_restore_rng_state(rng)
        if not torch.equal(private,self.generator.get_state()):raise ValueError('Evaluation consumed auxiliary RNG')
        result=dict(step=step,source_sha256=self.target_sha256,rows=rows,domain_counts={k:len(v) for k,v in domains.items()},
                    evaluation_population=10481,training_population=self.population,auxiliary_generator_advanced=False,
                    scope='Identical qualified TRAIN domains in ALL arms; no global free-space claim',evaluation_quadrature='fixed midpoint512; training sealed private randint+jitter')
        p=Path(output)/'clear_targets'/f'step-{step:09d}.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');return result
