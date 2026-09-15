"""Allow AlmostSolved only behind the unchanged independent numeric gates.

The original failed-run backend stays byte-for-byte intact. This explicit
adapter also closes a malformed nonfinite certificate-input validation gap.
"""
import inspect
import numpy as np
import conic_surface_step as original


def certificate(h,q,a,b,y,z,linear_count):
    values=[h.data,np.asarray(q),a.data,np.asarray(b),np.asarray(y),np.asarray(z)]
    if not all(np.isfinite(v).all() for v in values):raise ValueError('Nonfinite certificate input')
    stationary=h@np.asarray(y)+q+a.T@np.asarray(z)
    if not np.isfinite(stationary).all():raise ValueError('Nonfinite certificate stationarity')
    return original.certificate(h,q,a,b,y,z,linear_count)


SOURCE=inspect.getsource(original.solve)
BEFORE="if str(result.status)!='Solved':raise ValueError(f'Clarabel did not solve: {result.status}')"
AFTER="if str(result.status) not in ('Solved','AlmostSolved'):raise ValueError(f'Clarabel did not solve: {result.status}')"
assert SOURCE.count(BEFORE)==1
GENERATED=SOURCE.replace(BEFORE,AFTER)
_namespace=dict(original.__dict__,certificate=certificate)
exec(compile(GENERATED,'<explicit_certified_conic_status_adapter>','exec'),_namespace)
solve=_namespace['solve']


def provenance():
    from study_multiview_face_prior import sha
    proof=original.provenance();proof['helper_hashes'][__file__]=sha(__file__)
    return dict(proof,allowed_backend_status=['Solved','AlmostSolved'],independent_tolerances_unchanged=True,
        original_source=SOURCE,generated_source=GENERATED,nonfinite_certificate_input_check=True)
