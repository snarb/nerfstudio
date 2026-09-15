"""Same anatomical fit with positive-clearance/all-pair contact proposals."""
from pathlib import Path
import run_mhr_anatomical_correction as anatomical_driver
import fit_mhr_conic_correction as cone_driver
import mesh_contact_clearance as contact
from study_multiview_face_prior import sha


if __name__=='__main__':
    cone_source=Path(cone_driver.__file__).read_text()
    replacements={
        'from mesh_contact_constraints import contact_constraints':
            'from mesh_contact_clearance import contact_constraints\nfrom guard_mhr_anatomical_correction import all_pairs as ALL_CONTACT_PAIRS',
        'new=parent.strict_pairs(trial,guard.t)-guard.allowed_pairs':
            'new=ALL_CONTACT_PAIRS(trial,guard.t)-guard.allowed_all',
        "Path(__file__).with_name('mesh_contact_constraints.py')":
            "Path(__file__).with_name('mesh_contact_clearance.py')"}
    for before,after in replacements.items():
        assert cone_source.count(before)==1,before
        cone_source=cone_source.replace(before,after)
    source=Path(anatomical_driver.__file__).read_text()
    changes={"ROOT=Path('/mnt/data/dec5_mhr_anatomical_correction')":
            "ROOT=Path('/mnt/data/dec5_mhr_clearance_correction')",
        'source=Path(driver.__file__).read_text()':'source=CONTACT_DRIVER_SOURCE',
        '[ANCHOR_PROTOCOL,Path(__file__),Path(anatomical.__file__)]':
            '[ANCHOR_PROTOCOL,Path(__file__),Path(anatomical.__file__),*EXTRA_FILES]',
        'new_all_intersection_pairs_forbidden=True,discrete_not_continuous_guard=True,':
            'new_all_intersection_pairs_forbidden=True,discrete_not_continuous_guard=True,positive_contact_clearance=CLEARANCE,all_pair_contact_proposals=True,'}
    for before,after in changes.items():
        assert source.count(before)==1,before
        source=source.replace(before,after)
    exec(compile(source,'<positive_clearance_anatomical_driver>','exec'),dict(__name__='__main__',
        __file__=anatomical_driver.__file__,CONTACT_DRIVER_SOURCE=cone_source,CLEARANCE=contact.CLEARANCE,
        EXTRA_FILES=[Path(__file__),Path(contact.__file__),Path(cone_driver.__file__)]))
