"""New private backend control; same fitting objective and geometry limits."""
from pathlib import Path
import fit_mhr_contact_correction as contact
import fit_mhr_bounded_contact_correction as bounded
import osqp_surface_step as backend
from study_multiview_face_prior import sha


if __name__=='__main__':
    source=Path(bounded.__file__).read_text()
    old="ROOT=Path('/mnt/data/dec5_mhr_bounded_contact_correction')"
    assert source.count(old)==1
    source=source.replace(old,"ROOT=Path('/mnt/data/dec5_mhr_osqp_correction')")
    old='maximum_displacement_unchanged=.001,discrete_guards_unchanged=True)'
    assert source.count(old)==1
    source=source.replace(old,'maximum_displacement_unchanged=.001,discrete_guards_unchanged=True,qp_backend=OSQP_PROVENANCE)')
    proof=backend.provenance();proof.update(wrapper_sha256=sha(__file__),
        original_driver_sha256=sha(bounded.__file__),only_solver_backend_and_output_path_changed=True)
    contact.lower_bound_lsq=backend.lower_bound_lsq
    exec(compile(source,'<pinned_osqp_correction_driver>','exec'),dict(__name__='__main__',
        __file__=bounded.__file__,OSQP_PROVENANCE=proof))
