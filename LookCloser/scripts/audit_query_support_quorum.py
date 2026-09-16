"""Independent footprint replay with explicit query-near quorum counterfactual."""
import inspect
from pathlib import Path
from study_multiview_face_prior import read, save, sha
import audit_measured_free_surface as original
from study_query_support_quorum import ROOT, FRAME


def main():
    root=ROOT/FRAME; q=read(root/'request.json')
    assert q['policy']==dict(query_near_protection_minimum=3,near_tolerance=.0015,near_native_radius=0,
        min_stable_far=6,min_corroborated_far=6,other_views_per_far_observation=3)
    assert sha(Path(__file__).with_name('study_query_support_quorum.py'))==q['script_sha256']
    assert not (root/'independent_audit.json').exists() and not (root/'audit_adapter.json').exists()
    code=inspect.getsource(original.audit)
    for before,after in [("q['parameters']['near_native_radius']","q['policy']['near_native_radius']"),
                         ("q['scripts'].items()","q['helpers'].items()"),
                         ('assert (near==0).all()','assert (near<3).all()')]:
        assert code.count(before)==1,before;code=code.replace(before,after)
    save(root/'audit_adapter.json',dict(script_sha256=sha(__file__),original_auditor_sha256=sha(original.__file__),
        generated_source=code,changed_query_support_protection=True,quality_approval=False))
    namespace=dict(original.__dict__)
    exec(compile(code,'<query-quorum-independent-footprints>','exec'),namespace)
    namespace['audit'](FRAME,ROOT)


if __name__=='__main__':main()
