"""Tighter internal linear-system refinement; certificate and problem unchanged."""
from types import FunctionType
from pathlib import Path
import certified_conic_surface_step as original

SETTINGS = dict(original.solve.__globals__['SETTINGS'],
    iterative_refinement_reltol=1e-15, iterative_refinement_abstol=1e-14,
    iterative_refinement_max_iter=30)
solve = FunctionType(original.solve.__code__, dict(original.solve.__globals__, SETTINGS=SETTINGS))


def provenance():
    from study_multiview_face_prior import sha
    proof = original.provenance()
    proof['helper_hashes'][str(Path(__file__))] = sha(__file__)
    return dict(proof, settings=SETTINGS, internal_linear_refinement_only=True,
                problem_and_acceptance_tolerances_unchanged=True)
