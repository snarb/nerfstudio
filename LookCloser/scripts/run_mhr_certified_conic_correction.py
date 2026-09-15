"""Exact balls with independently checked Solved/AlmostSolved numerical output."""
from pathlib import Path
import fit_mhr_conic_correction as driver
import certified_conic_surface_step as backend


if __name__=='__main__':
    source=Path(driver.__file__).read_text()
    changes={"ROOT=Path('/mnt/data/dec5_mhr_conic_correction')":
             "ROOT=Path('/mnt/data/dec5_mhr_certified_conic_correction')",
             'import conic_surface_step as backend':'import certified_conic_surface_step as backend',
             'qp_backend=backend.provenance(),maximum_constraint_generation_rounds=9,':
             'qp_backend=backend.provenance(),execution_wrapper_sha256=sha(EXECUTION_WRAPPER),maximum_constraint_generation_rounds=9,'}
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    exec(compile(source,'<certified_conic_driver>','exec'),dict(__name__='__main__',
        __file__=driver.__file__,EXECUTION_WRAPPER=__file__))
