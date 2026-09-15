"""Single train-only weight16 control; other clearance/anatomical limits frozen."""
from pathlib import Path
import argparse
import run_mhr_clearance_correction as clearance
import run_mhr_anatomical_correction as anatomical


def sources():
    source=Path(anatomical.__file__).read_text()
    old="source=source.replace(old,'active = anatomical_domain(neutral)')"
    extra=(old+"\n    old_weight='weights = np.sqrt(4.*robust/(len(rows)*count))/2.'\n"
        "    assert source.count(old_weight)==1\n"
        "    source=source.replace(old_weight,'weights = np.sqrt(16.*robust/(len(rows)*count))/2.')")
    edits={old:extra,
        'ns=dict(worker.__globals__,anatomical_domain=anatomical.anatomical_domain)':
        "ns=dict(worker.__globals__,anatomical_domain=anatomical.anatomical_domain)\n    ns['RECIPE']=dict(ns['RECIPE'],silhouette_weight=16.)",
        'dict(proof,generated_source=source,':
        "dict(proof,generated_source=source,recipe=dict(proof['recipe'],silhouette_weight=16.),",
        'width_from_existing_anchor_protocol=True,':
        'width_from_existing_anchor_protocol=True,silhouette_weight=16.,baseline_silhouette_weight=4.,'}
    for before,after in edits.items():
        assert source.count(before)==1,before
        source=source.replace(before,after)
    driver=Path(clearance.__file__).read_text()
    edits={"ROOT=Path('/mnt/data/dec5_mhr_clearance_correction')":
           "ROOT=Path('/mnt/data/dec5_mhr_silhouette_weight16')",
           'source=Path(anatomical_driver.__file__).read_text()':'source=ANATOMICAL_SOURCE',
           'EXTRA_FILES=[Path(__file__),Path(contact.__file__),Path(cone_driver.__file__)]':
           'EXTRA_FILES=[Path(__file__),Path(contact.__file__),Path(cone_driver.__file__),WEIGHT_WRAPPER]'}
    for before,after in edits.items():
        assert driver.count(before)==1,before
        driver=driver.replace(before,after)
    return source,driver


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    source,driver=sources()
    if args.dry_run:
        print('Validated exact-source adapter: silhouette weight4->16; new output dec5_mhr_silhouette_weight16; no fit launched')
    else:
        exec(compile(driver,'<weight16_clearance_driver>','exec'),dict(__name__='__main__',
            __file__=clearance.__file__,ANATOMICAL_SOURCE=source,WEIGHT_WRAPPER=Path(__file__)))
