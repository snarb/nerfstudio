"""Seal the explicitly completed negative visual review; never promote geometry."""
from pathlib import Path
from joint_temporal_texture import read,sha,atomic_json
from review_instance_qualified_surface import PAIRS,VIEWS,geometry_audit


def main():
    report=Path(__file__).resolve().parents[1]/'experiments/dec5_instance_qualified_free_surface.md'
    tests=Path('/home/brans/lookcloser_temp/instance_qualified_tests.log')
    assert '4 passed' in tests.read_text()
    for variant,(path,_) in PAIRS.items():
        root=Path(path)/'000995';out=root/'semantic_review'
        assert not (out/'visual_review.json').exists()
        audit=read(out/'audit.json');bindings=geometry_audit(root)
        for p,h in audit['bindings'].items():
            assert sha(p)==h,p
            bindings[p]=h
        viewed=[]
        for view in VIEWS:
            for kind in ['head','lipstick']:viewed.append(out/view/(kind+'.png'))
        sheets=read(out/'black_sheets.json')
        for row in sheets['sheets']:
            assert sha(row['path'])==row['sha256']
            viewed.append(Path(row['path']))
        review=dict(status='fail',reviewer='main LLM',actually_viewed=[
            dict(path=str(p),sha256=sha(p)) for p in viewed],
            scope='three matched head/lipstick panels and all localized new-black components',
            artifact='cloth-textured wedge behind lipstick remains; partial fringe trim only',
            notes='No complete cylindrical recovery. Existing head defects unchanged. '
                  'Subdivision adds one moving-view black pixel beside false skin bridge.',
            metric_override=False,production_promoted=False,full_frame_quality_metrics=False,
            rendered_views=VIEWS,frame='000995')
        atomic_json(out/'visual_review.json',review)
        for p in root.rglob('*'):
            if p.is_file():bindings[str(p)]=sha(p)
        for p in [report,tests,Path(__file__).resolve(),Path(__file__).with_name('review_instance_qualified_surface.py')]:
            bindings[str(p)]=sha(p)
        # Review generation preceded addition of the optional sheet-only CLI.
        # Preserve its execution hash instead of claiming current bytes ran it.
        atomic_json(out/'final_seal.json',dict(status='complete_negative_not_promoted',
            bindings=bindings,review_execution_script_sha256=audit['script_sha256'],
            current_review_script_sha256=sha(Path(__file__).with_name('review_instance_qualified_surface.py')),
            review_script_change_after_generation='optional native contact-sheet helper and CLI only',
            tests_passed=4,visual_status='fail',production_changed=False))
        for p,h in read(out/'final_seal.json')['bindings'].items():assert sha(p)==h,p
        print(variant,'sealed',len(bindings),'bindings',flush=True)


if __name__=='__main__':main()
