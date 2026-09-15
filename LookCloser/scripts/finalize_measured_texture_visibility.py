"""Seal actual-source replay and explicit LLM verdict, without production approval."""
from pathlib import Path
from joint_temporal_texture import read,sha,atomic_json
from study_measured_texture_visibility import ROOT,FRAME,DEPTH_ROOT
from study_confidence_depth_prior import load_real
from review_measured_free_surface import VIEWS


def main():
    root=ROOT/FRAME;bindings={};rows=[]
    _,_,receipt=load_real(DEPTH_ROOT,FRAME)
    for view in VIEWS:
        folder=root/view;q=read(folder/'request.json');r=read(folder/'review/result.json')
        assert q['measured_texture_visibility']['depth_receipt']==receipt
        assert r['actual_sources_contradicting_stable_far'][1]==0 and r['mesh_and_target_depth_exact']
        bindings.update(r['bindings'])
        bindings.update({str(folder/'review'/n):h for n,h in r['hashes'].items()})
        bindings[str(folder/'guard_audit.json')]=r['guard_audit_sha256']
        bindings[str(Path(__file__).with_name('review_measured_texture_visibility.py'))]=r['script_sha256']
        bindings.update({str(Path(__file__).with_name(n)):h for n,h in q['script_hashes'].items()})
        rows.append(dict(view=view,new_black_rgb=r['new_black_rgb'],
            selected_far_before_after=r['actual_sources_contradicting_stable_far'],
            diagnostic_source_255=r['diagnostic_source_255']))
    # Preserve the initial wrong-lattice canary and bind its exact executed code.
    initial=Path('/mnt/data/dec5_measured_texture_visibility')
    iq=read(initial/FRAME/'K004_B005_1210DS/request.json')
    archived=initial/'config/executed_initial_worker.py'
    assert sha(archived)==iq['script_hashes']['study_measured_texture_visibility.py']
    bindings[str(archived)]=sha(archived)
    verdict=read(root/'visual_review.json')
    assert verdict['status']=='rejected_as_standalone_artifact_fix' and not verdict['production_promoted']
    bindings.update(verdict['reviewed_images'])
    for path,h in bindings.items():assert sha(path)==h,path
    files={str(p):sha(p) for base in [ROOT,initial] for p in base.rglob('*')
           if p.is_file() and p.name!='final_audit.json'}
    atomic_json(root/'final_audit.json',dict(status='passed',rows=rows,
        bindings=bindings,files=files,depth_receipt=receipt,script_sha256=sha(__file__),
        binding_count=len(bindings),file_count=len(files),visual_status=verdict['status'],
        initial_wrong_lattice_canary_preserved=True,production_promoted=False))
    print('measured texture study sealed',len(bindings),'bindings',len(files),'files',flush=True)


if __name__=='__main__':main()
