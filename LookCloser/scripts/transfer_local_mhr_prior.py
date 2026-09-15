"""Fresh per-time DEC5 prior with frozen fitting functions and explicit paths.

No previous fitted pose is read. Review and local admission remain separate gates.
"""
from pathlib import Path
import argparse, importlib, inspect, sys
import numpy as np
from study_multiview_face_prior import read, save, sha
from run_local_mhr_completion import adapt, require


def settings(spec):
    root=Path(spec['root']); inputs=Path(spec['inputs'])
    return dict(ROOT=root, FRAME=spec['frame'], CANONICAL=root/'canonical_pose', HEAD=root/'head',
        CONFORM=root/'measured_conformance', TEN=root/'silhouette10', FINAL=root/'silhouette100', RGB=inputs/'rgb')


def preflight(spec):
    from joint_temporal_texture import cameras, HELD_CAMERAS
    from study_confidence_depth_prior import load_real
    inputs=Path(spec['inputs']); require(not inputs.exists() and not Path(spec['root']).exists(), 'New private roots required')
    rows,raw,metadata=cameras(spec['frame']); names=[r['physical_camera'] for r in rows]
    require(len(names)==62 and not set(names)&HELD_CAMERAS,'Train-only input required')
    row=next(r for r in read(spec['production_request'])['inventory'] if r['frame_id']==spec['frame'])
    for k in ['mesh','metadata']:require(sha(row[k])==row[k+'_sha256'],'Production hash mismatch')
    require(Path(row['metadata'])==Path(metadata),'Normalization mismatch')
    source=read(Path(spec['measured_control'])/'request.json')
    require(source['frame']==spec['frame'] and source['source_mesh_sha256']==row['mesh_sha256'],'Measured-control source mismatch')
    dr,depths,receipt=load_real(Path(source['depth_root']),spec['frame']);require(receipt==source['depth_receipt'],'Depth receipt mismatch')
    mask=Path(row['source_masks']['root']);override=Path(source['mask_override']['root'])/spec['frame']
    oq=read(override/'request.json');result=read(override/'result.json')
    require(not oq['candidate_mesh_used'] and not oq['heldout_used'] and oq['depth_receipt']==receipt,'Override must be independent measured evidence')
    require(sha(override/'result.json')==source['mask_override']['result_sha256'],'Changed override')
    require(result['request_sha256']==sha(override/'request.json') and result['original_masks_sha256']==sha(mask/'masks.npz'),'Override source mismatch')
    for name,h in result['hashes'].items():require(sha(override/name)==h,'Override artifact changed')
    paths=[Path(__file__),Path(spec['production_request']),Path(spec['measured_control'])/'request.json',Path(raw),Path(metadata),
        mask/'masks.npz',mask/'cameras.json',override/'request.json',override/'result.json',override/'mask.npy',
        Path('/mnt/data/dec5_multiview_face_prior/face_landmarker.task'),Path('/mnt/data/dec5_multiview_face_prior/request.json')]
    for name in ['transfer_mhr_001195.py','study_multiview_face_prior.py','study_canonical_face_prior.py','study_mhr_local_head_prior.py',
        'fit_mhr_local_head_prior.py','fit_mhr_named_articulation.py','conform_mhr_measured_surface.py','fit_mhr_silhouette_conformance.py',
        'continue_mhr_silhouette_convergence.py','audit_mhr_transfer_001195.py','review_mhr_local_head_prior.py','review_mhr_silhouette_conformance.py','run_local_mhr_completion.py']:
        paths.append(Path(__file__).with_name(name))
    require(sha('/mnt/data/dec5_multiview_face_prior/face_landmarker.task')==read('/mnt/data/dec5_multiview_face_prior/request.json')['model_sha256'],'Face model mismatch')
    inputs.mkdir(parents=True)
    save(inputs/'preflight.json',dict(spec=spec,production=row,raw_mesh=str(raw),raw_mesh_sha256=sha(raw),
        depth_receipt=receipt,depth_root=source['depth_root'],train_cameras=names,override_regenerated=False,
        input_hashes={str(p):sha(p) for p in paths},no_previous_fitted_pose_input=True,production_modified=False))
    print('preflight passed',spec['frame'],len(depths),'native maps',flush=True)


def verify(spec):
    q=read(Path(spec['inputs'])/'preflight.json');require(q['spec']==spec,'Changed spec')
    for p,h in q['input_hashes'].items():require(sha(p)==h,'Changed preflight input: '+p)
    return q


def configure(spec):
    import transfer_mhr_001195 as base
    s=settings(spec)
    for k,v in s.items():setattr(base,k,v)
    original=base.configure
    def configured():
        canonical,head,conform,sil,cont=original()
        canonical.SOURCE=s['RGB'];canonical.DEPTH=Path(verify(spec)['depth_root']);head.RGB=s['RGB']
        head.anchors,proof=adapt(head,'anchors',[("Path('/mnt/data/dec5_jaw_measured_depth/analysis')",'DEPTH_ROOT',1)],
            dict(DEPTH_ROOT=canonical.DEPTH))
        return canonical,head,conform,sil,cont
    base.configure=configured
    return base,configured()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['preflight','stage_rgb','infer','init','semantic','anchors','fit','articulation','conform','silhouette10','silhouette100','review_head','review_final','audit_head','audit_conformance','audit_continuation','seal_prior'])
    p.add_argument('--spec',type=Path,required=True);a=p.parse_args();spec=read(a.spec);s=settings(spec);inputs=Path(spec['inputs'])
    if a.stage=='preflight':preflight(spec);return
    verify(spec)
    if a.stage in ['stage_rgb','infer']:
        import study_multiview_face_prior as face
        face.OUT=s['RGB'];face.FRAMES=[spec['frame']]
        if a.stage=='stage_rgb':
            face.stage();(s['RGB']/'face_landmarker.task').symlink_to('/mnt/data/dec5_multiview_face_prior/face_landmarker.task')
        else:face.infer()
    elif a.stage=='semantic':
        import study_mhr_local_head_prior as head
        head.OUT=s['HEAD'];head.FRAME=spec['frame'];head.RGB=s['RGB'];head.semantic()
        save(s['ROOT']/'semantic_complete.json',dict(wrapper_sha256=sha(__file__),producer_sha256=sha(head.__file__),same_semantic_function=True))
    else:
        base,(canonical,head,conform,sil,cont)=configure(spec)
        if a.stage=='init':
            base.configure=lambda:(canonical,head,conform,sil,cont)
            original_read=base.read
            def updated_read(path):
                value=original_read(path)
                if Path(path)==Path('/mnt/data/dec5_canonical_face_prior/protocol.json'):
                    value=dict(value,source_inference_sha256=sha(s['RGB']/'inference.json'),source_request_sha256=sha(s['RGB']/'request.json'))
                return value
            base.read=updated_read;base.init()
        elif a.stage in ['anchors','fit','articulation','conform','silhouette10','silhouette100']:
            # base.main calls configure again; retain the already configured modules
            # rather than stacking the path-only anchor adapter.
            base.configure=lambda:(canonical,head,conform,sil,cont)
            sys.argv=[sys.argv[0],a.stage];base.main()
        elif a.stage.startswith('audit_'):
            audit=importlib.import_module('audit_mhr_transfer_001195')
            for k,v in s.items():setattr(audit,k,v)
            stage=a.stage[6:]
            if stage=='head':
                fn,proof=adapt(audit,'head',[("Path('/mnt/data/dec5_jaw_measured_depth/analysis')",'DEPTH_ROOT',1)],dict(DEPTH_ROOT=Path(verify(spec)['depth_root'])))
                save(inputs/'head_audit_adapter.json',proof);fn()
            else:getattr(audit,stage)()
        elif a.stage=='review_head':
            importlib.import_module('review_mhr_local_head_prior').main(['similarity','head20','head20_neck6'])
        elif a.stage=='review_final':
            sil.ROOT=s['FINAL'];review=importlib.import_module('review_mhr_silhouette_conformance')
            source=inspect.getsource(review.main);begin=source.index("    parentpath=Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json')")
            end=source.index("    save(dest/'result.json'",begin);block=source[begin:end]
            fn,proof=adapt(review,'main',[(block,'    target=[]  # Frame-specific moving review is separate, never a fitting input.\n',1)],{})
            save(inputs/'final_review_adapter.json',proof);fn()
        elif a.stage=='seal_prior':
            for file in [s['ROOT']/'head_audit.json',s['ROOT']/'conformance_audit.json',s['FINAL']/'audit.json']:
                require(read(file)['status']=='passed','Prior audit missing')
            require(read(s['FINAL']/'continuation_result.json')['stop_reason'] in ['hard_cap_not_converged','converged'],'Failed continuation cannot be admitted')
            visual=read(inputs/'prior_visual_review.json');require(visual['reviewer']=='LLM' and visual['local_admission_allowed'],'Prior visual gate missing')
            checked=dict(verify(spec)['input_hashes'])
            for folder in [inputs,s['HEAD'],s['CONFORM'],s['CANONICAL']]:
                for f in folder.rglob('*'):
                    if f.is_file():checked[str(f)]=sha(f)
            for f in [s['ROOT']/'config.json',s['ROOT']/'head_audit.json',s['ROOT']/'conformance_audit.json']:
                checked[str(f)]=sha(f)
            dest=s['FINAL']/'final_seal.json';require(not dest.exists(),'Prior seal exists')
            save(dest,dict(status='passed',frame=spec['frame'],checked_bindings=checked,
                inventory={str(f.relative_to(s['FINAL'])):sha(f) for f in sorted(s['FINAL'].rglob('*')) if f.is_file()},
                script_sha256=sha(__file__),whole_prior_rejected=True,local_candidate_admission_required=True,production_accepted=False))
    save(inputs/(a.stage+'_complete.json'),dict(spec_sha256=sha(a.spec),wrapper_sha256=sha(__file__),stage=a.stage))


if __name__=='__main__':main()
