"""Frame-only adapters for the SAME generic candidate and measured admission math."""
from pathlib import Path
import argparse,importlib,inspect,hashlib,sys
import numpy as np
from transfer_mhr_001195 import ROOT,HEAD,CONFORM,FINAL,FRAME,configure,verify
from study_multiview_face_prior import read,save,sha

CANDIDATES=ROOT/'candidates'
OUT=ROOT/'admission'
ARM='silhouette100'


def adapted(module,function,replacements,namespace):
    source=inspect.getsource(getattr(module,function));generated=source
    for before,after,count in replacements:
        assert generated.count(before)==count,(function,before)
        generated=generated.replace(before,after)
    state=dict(module.__dict__,**namespace)
    exec(compile(generated,'<explicit_frame_001195_adapter>','exec'),state)
    evidence=dict(frozen_path=str(Path(module.__file__).resolve()),frozen_sha256=sha(module.__file__),
        original_source=source,generated_source=generated,generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        replacements=[dict(before=a,after=b,expected_count=c) for a,b,c in replacements],
        wrapper_sha256=sha(__file__),config_sha256=sha(ROOT/'config.json'),frame=FRAME)
    return state[function],evidence


def candidate_builder(destination=CANDIDATES):
    import build_mhr_silhouette_patch_candidates as module
    fn,proof=adapted(module,'build',[("inherited=Path('/mnt/data/dec5_mhr_measured_conformance/smooth100/fit.npz')","inherited=CONFORM/'smooth100/fit.npz'",1),
        ("frame='001193'","frame=FRAME",1)],dict(ROOT=destination,PRIOR=FINAL,ARM=ARM,CONFORM=CONFORM,FRAME=FRAME))
    save(ROOT/('candidate_adapter_replay.json' if destination!=CANDIDATES else 'candidate_adapter.json'),proof)
    fn(destination)


def admission_configure():
    configure()
    import admit_mhr_local_patch_depth as m
    m.CANDIDATES=CANDIDATES;m.OUT=OUT;m.PRIOR=CANDIDATES/'prior';m.ARMS=[ARM]
    return m


def seal_prior():
    verify();assert read(ROOT/'head_audit.json')['status']=='passed'
    assert read(ROOT/'conformance_audit.json')['status']=='passed';assert read(FINAL/'audit.json')['status']=='passed'
    state=read(FINAL/'continuation_result.json');assert state['stop_reason'] in ['hard_cap_not_converged','converged']
    target=FINAL/'final_seal.json';assert not target.exists()
    checked={}
    for folder in [HEAD,CONFORM,ROOT/'canonical_pose']:
        for p in folder.rglob('*'):
            if p.is_file():checked[str(p)]=sha(p)
    config=read(ROOT/'config.json')
    for p,h in config['inputs'].items():assert sha(p)==h;checked[p]=h
    for p,h in config['producers'].items():path=Path(__file__).with_name(p);assert sha(path)==h;checked[str(path)]=h
    for p in [ROOT/'config.json',ROOT/'head_audit.json',ROOT/'conformance_audit.json',ROOT/'semantic_complete.json']:
        checked[str(p)]=sha(p)
    q=read(HEAD/'protocol.json')
    for p in [q['model_path'],q['semantic_path'],q['original_mesh']]:checked[str(p)]=sha(p)
    for file in ['review_mhr_transfer_001195.py','audit_mhr_transfer_001195.py','transfer_mhr_001195_semantics.py','transfer_mhr_001195.py']:
        p=Path(__file__).with_name(file);checked[str(p)]=sha(p)
    save(target,dict(status='passed',checked_bindings=checked,inventory={str(p.relative_to(FINAL)):sha(p) for p in sorted(FINAL.rglob('*')) if p.is_file()},
        script_sha256=sha(__file__),frame=FRAME,prior_only=True,visual_review=['initial G/A','anchor G/B','fitted G/B','final C/E clay','final G/B projection','requested-hole clay'],
        whole_prior_rejected=True,local_candidate_admission_required=True,production_accepted=False))
    print('prior sealed',len(checked),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['seal_prior','build','admit','candidate_audit','admission_audit','review','rgb','rgb_review','occlusion']);p.add_argument('--views',nargs='+',default=['old_moving','F004_E','M004_B','C004_E']);a=p.parse_args()
    verify();configure()
    if a.stage=='seal_prior':seal_prior()
    elif a.stage=='build':candidate_builder()
    elif a.stage in ['admit','admission_audit']:
        m=admission_configure()
        if a.stage=='admit':
            proof=dict(wrapper_sha256=sha(__file__),config_sha256=sha(ROOT/'config.json'),candidate_adapter_sha256=sha(ROOT/'candidate_adapter.json'),
                final_prior_fit_sha256=sha(FINAL/'fit.npz'),final_prior_topology_sha256=sha(FINAL/'review_v2/silhouette_topology.npz'),
                final_prior_seal_sha256=sha(FINAL/'final_seal.json'),frame=FRAME,admission_math_unchanged=True)
            original=m.save
            def extended(path,value):
                if Path(path)==OUT/'request.json':value=dict(value,transfer=proof)
                original(path,value)
            m.save=extended;m.run()
        else:importlib.import_module('audit_mhr_local_patch_depth').main()
    elif a.stage=='candidate_audit':
        replay=ROOT/'candidate_replay';candidate_builder(replay)
        assert read(replay/'request.json')==read(CANDIDATES/'request.json')
        for name in ['proposal_evidence.npz','domain_evidence.npz']:
            x=np.load(CANDIDATES/ARM/name);y=np.load(replay/ARM/name)
            assert x.files==y.files
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
        assert sha(CANDIDATES/ARM/'local_raw.ply')==sha(replay/ARM/'local_raw.ply')
        save(ROOT/'candidate_audit.json',dict(status='passed',all_arrays_and_raw_ply_exact=True,wrapper_sha256=sha(__file__)))
    elif a.stage=='review':
        import admit_mhr_silhouette_patch as wrapper
        wrapper.OUT=OUT;wrapper.CANDIDATES=CANDIDATES;wrapper.PRIOR=FINAL
        importlib.import_module('review_mhr_silhouette_patch').main()
    elif a.stage=='rgb':
        m=importlib.import_module('render_mhr_silhouette_patch_cpu')
        fn,proof=adapted(m,'main',[("'001193'",'FRAME',3)],dict(OUT=OUT,FRAME=FRAME))
        name='_'.join(a.views);save(ROOT/('rgb_adapter_'+name+'.json'),proof)
        sys.argv=[sys.argv[0],'--views',*a.views];fn()
    elif a.stage=='rgb_review':
        m=importlib.import_module('review_mhr_silhouette_patch_rgb');m.OUT=OUT;m.FRAME=FRAME
        sys.argv=[sys.argv[0],'--views',*a.views];m.main()
    elif a.stage=='occlusion':
        m=importlib.import_module('localize_mhr_silhouette_patch_occlusion')
        fn,proof=adapted(m,'main',[("'frames/001193'","'frames'/FRAME",1)],dict(OUT=OUT,FRAME=FRAME))
        save(ROOT/'occlusion_adapter.json',proof);fn()


if __name__=='__main__':main()
