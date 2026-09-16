"""Compose the two existing controls; separately audit the actual blue wedge.

The small diagnostic core is manually identified on the saved coordinate grid,
not used to choose geometry or sources. All rendering/guard thresholds stay fixed.
"""
import argparse
import inspect
import hashlib
import multiprocessing
from pathlib import Path
from study_multiview_face_prior import read,save,sha
from study_query_support_quorum import ROOT as GEOMETRY,FRAME
from review_measured_free_surface import VIEWS

ROOT=Path('/mnt/data/dec5_query_quorum_measured_texture')
CORE_POLYGON=[(156,1230),(164,1230),(164,1248),(156,1248)]


def baseline(frame,view):
    assert frame==FRAME
    return GEOMETRY/frame/'rgb'/view


def render_view(view):
    import study_measured_texture_visibility as control
    control.ROOT=ROOT;control.baseline=baseline
    control.run(view)


def review_view(view):
    import review_measured_texture_visibility as reviewer
    code=inspect.getsource(reviewer.review)
    old='from diagnose_lipstick_fin_depth import POLYGON';assert code.count(old)==1
    code=code.replace(old,'POLYGON=CORE_POLYGON')
    namespace=dict(reviewer.__dict__,ROOT=ROOT,baseline=baseline,CORE_POLYGON=CORE_POLYGON)
    exec(compile(code,'<combined-query-quorum-texture-review>','exec'),namespace)
    namespace['review'](view)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['render','review'])
    stage=p.parse_args().stage
    if stage=='render':
        import study_measured_texture_visibility as producer
        import review_measured_texture_visibility as reviewer
        assert not ROOT.exists();ROOT.mkdir()
        save(ROOT/'adapter.json',dict(wrapper_sha256=sha(__file__),producer_sha256=sha(producer.__file__),
            reviewer_sha256=sha(reviewer.__file__),geometry_result_sha256=sha(GEOMETRY/FRAME/'result.json'),
            baseline_requests={v:sha(baseline(FRAME,v)/'request.json') for v in VIEWS},
            diagnostic_core=CORE_POLYGON,diagnostic_core_used_in_prediction=False,
            source_guard_thresholds_unchanged=True,production_changed=False))
    else:
        q=read(ROOT/'adapter.json');assert q['wrapper_sha256']==sha(__file__)
        for v,h in q['baseline_requests'].items():assert sha(baseline(FRAME,v)/'request.json')==h
    with multiprocessing.get_context('spawn').Pool(3,maxtasksperchild=1) as pool:
        pool.map(render_view if stage=='render' else review_view,VIEWS)
