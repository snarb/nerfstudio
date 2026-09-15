"""Same frozen objective/base; only stopping budget and diagnostic observer differ."""
from pathlib import Path
import inspect
import hashlib
import numpy as np
import fit_mhr_silhouette_conformance as original
from study_multiview_face_prior import read, save, sha

ROOT = Path('/mnt/data/dec5_mhr_silhouette_convergence')
CONTROL = Path('/mnt/data/dec5_mhr_silhouette_conformance')
STOP = dict(maximum_outer_iterations=100, unconstrained_step_tolerance=1e-5,
            consecutive_small_steps=3, collapsed_area_ratio=1e-6,
            objective_increase_factor=1.25, consecutive_large_increases=3)


def instrument(source):
    needle = "        save(ROOT/'progress.json', dict(history=history))"
    assert source.count(needle) == 1
    return source.replace(needle, needle + '\n        if continuation_observe(locals()):\n            break')


def make_optimizer():
    frozen = read(CONTROL/'protocol.json')
    assert sha(original.__file__) == frozen['script_sha256']
    namespace = dict(original.__dict__)
    namespace['ROOT'] = ROOT
    namespace['RECIPE'] = dict(original.RECIPE, outer_iterations=100)
    state = dict(records=[], consecutive_small=0, consecutive_increases=0,
                 stop_reason=None, first10_exact=False)
    reference = np.load(CONTROL/'fit.npz')['vertices']
    history_reference = read(CONTROL/'result.json')['history']
    (ROOT/'iterates').mkdir(exist_ok=False)

    def observe(context):
        current, base, triangles = context['current'], context['base'], context['triangles']
        history, outer = context['history'], context['outer']
        displacement = (current[context['ids']]-base[context['ids']]).ravel()
        # Re-evaluate the nonlinear pseudo-Huber objective after the step.
        # Its derivative gives the original IRLS weights; diagnostics never feed
        # the solver. Association/FOV sets can change and are logged explicitly.
        o3d=context['o3d'];head_tri=context['head_tri']
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(current),o3d.utility.Vector3iVector(head_tri));mesh.compute_triangle_normals()
        cp=original.Scene2(current,head_tri).compute_closest_points(o3d.core.Tensor(context['points'].astype(np.float32)))
        d=np.linalg.norm(cp['points'].numpy()-context['points'],axis=1)
        dot=np.sum(np.asarray(mesh.triangle_normals)[cp['primitive_ids'].numpy()]*context['normals'],axis=1)
        use=(d<=.006)&(dot>=.25);neck=context['neck'];groups=np.where(neck[use],(use&neck).sum(),(use&~neck).sum())
        data=float(np.sum((np.sqrt(1+(d[use]/.001)**2)-1)/groups))
        sil = 0.
        available=0
        for _,values,_ in original.silhouette_samples(current[context['ids']], context['rows'], context['sdfs']):
            excess = np.maximum(values-2,0)
            sil += float(np.sum(128/(len(context['rows'])*context['count'])*(np.sqrt(1+(excess/8)**2)-1)))
            available+=len(values)
        step = context['step'].ravel()
        before = float(np.dot(context['rhs'],context['rhs']))
        residual = context['system']@step-context['rhs']
        after = float(np.dot(residual,residual))
        smooth = float(np.sum((context['smooth']@displacement)**2))
        magnitude = float(np.sum((context['magnitude']@displacement)**2))
        cross=np.cross(current[triangles[:,1]]-current[triangles[:,0]],current[triangles[:,2]]-current[triangles[:,0]])
        basecross=np.cross(base[triangles[:,1]]-base[triangles[:,0]],base[triangles[:,2]]-base[triangles[:,0]])
        area_ratio=np.linalg.norm(cross,axis=1)/np.maximum(np.linalg.norm(basecross,axis=1),1e-30)
        record=dict(iteration=outer+1, fixed_irls_linearized_objective_before=before,
                    fixed_irls_linearized_objective_after=after,
                    measured_data_pseudohuber_post_step=data, silhouette_pseudohuber_post_step=sil,
                    laplacian_post_step=smooth, magnitude_post_step=magnitude,
                    nonlinear_objective_post_step=data+sil+smooth+magnitude,
                    post_step_associated_points=int(use.sum()),post_step_available_silhouette_samples=available,
                    minimum_area_ratio=float(area_ratio.min()),
                    reversed_normals=int((np.sum(cross*basecross,axis=1)<=0).sum()),
                    unconstrained_maximum_step=history[-1]['unconstrained_maximum_step'])
        if outer == 9:
            np.testing.assert_array_equal(current,reference)
            assert history == history_reference
            state['first10_exact'] = True
        np.savez_compressed(ROOT/'iterates'/f'{outer+1:03d}.npz',vertices=current)
        previous=state['records'][-1]['nonlinear_objective_post_step'] if state['records'] else None
        increase=previous is not None and record['nonlinear_objective_post_step']>previous*STOP['objective_increase_factor']
        state['consecutive_increases']=state['consecutive_increases']+1 if increase else 0
        state['records'].append(record)
        small = record['unconstrained_maximum_step'] <= STOP['unconstrained_step_tolerance']
        state['consecutive_small'] = state['consecutive_small']+1 if small else 0
        # Fixed-IRLS trust-scaled step must decrease its own quadratic objective.
        if not np.isfinite(current).all() or not np.isfinite(after):
            state['stop_reason']='nonfinite_failure'
        elif area_ratio.min() <= STOP['collapsed_area_ratio']:
            state['stop_reason']='collapsed_triangle_failure'
        elif after > before*(1+1e-8)+1e-10:
            state['stop_reason']='linearized_objective_increase_failure'
        elif state['consecutive_increases']>=STOP['consecutive_large_increases']:
            state['stop_reason']='sustained_objective_instability_failure'
        elif state['consecutive_small'] >= STOP['consecutive_small_steps']:
            state['stop_reason']='converged'
        elif outer+1 >= STOP['maximum_outer_iterations']:
            state['stop_reason']='hard_cap_not_converged'
        save(ROOT/'continuation_progress.json',state)
        return state['stop_reason'] is not None

    namespace['continuation_observe'] = observe
    source = inspect.getsource(FROZEN_OPTIMIZER)
    generated = instrument(source)
    save(ROOT/'instrumented_source.json',dict(original_source=source,instrumented_source=generated,
        original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        instrumented_source_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        change='add post-step diagnostic/termination observer only'))
    exec(compile(generated, '<frozen_optimizer_with_stop_observer>', 'exec'),namespace)
    return namespace['optimize'],state


def main():
    original_root, original_recipe, original_optimizer, original_save = original.ROOT, original.RECIPE, original.optimize, original.save
    control=read(CONTROL/'protocol.json')
    assert sha(original.__file__)==control['script_sha256']
    for name,digest in control['helpers'].items():assert sha(Path(original.__file__).with_name(name))==digest
    proof=dict(wrapper_path=str(Path(__file__).resolve()),wrapper_sha256=sha(__file__),
        frozen_producer_path=str(Path(original.__file__).resolve()),frozen_producer_sha256=sha(original.__file__),
        control_root=str(CONTROL),control_protocol_sha256=sha(CONTROL/'protocol.json'),
        control_fit_sha256=sha(CONTROL/'fit.npz'),same_base_reference=True,stop=STOP,
        no_weight_or_mask_changes=True)
    state_holder={}

    def optimize(*args,**kwargs):
        worker,state=make_optimizer()
        state_holder['state']=state
        return worker(*args,**kwargs)

    def extended_save(path,value):
        if Path(path)==ROOT/'protocol.json': value=dict(value,continuation=proof)
        original_save(path,value)

    original.ROOT=ROOT
    original.RECIPE=dict(original_recipe,outer_iterations=100)
    original.optimize=optimize
    original.save=extended_save
    try:
        # make_optimizer must inspect the real frozen function, not this wrapper.
        globals()['FROZEN_OPTIMIZER']=original_optimizer
        original.main()
        state=state_holder['state']
        assert state['first10_exact']
        save(ROOT/'continuation_result.json',dict(state,protocol_sha256=sha(ROOT/'protocol.json'),
             fit_sha256=sha(ROOT/'fit.npz'),production_accepted=False))
    finally:
        original.ROOT,original.RECIPE,original.optimize,original.save=original_root,original_recipe,original_optimizer,original_save


if __name__=='__main__':main()
