import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from run_temporal_face_angular_control import CLIPS,FRAMES,REUSE,ROOT,OUTPUTS,BASE,PILOT


def test_two_contiguous_real_time_clips_and_unique_work_inventory():
    assert len(CLIPS)==2
    for frames in CLIPS.values():
        assert len(frames)==24
        assert all(int(b)-int(a)==2 for a,b in zip(frames,frames[1:]))
    assert FRAMES==sorted(set(sum(CLIPS.values(),[]))) and len(FRAMES)==37
    assert REUSE<=set(FRAMES) and len(set(FRAMES)-REUSE)==33


def test_isolated_output_does_not_overlap_production_or_sealed_pilot():
    assert OUTPUTS.is_relative_to(ROOT)
    assert not ROOT.is_relative_to(BASE) and not ROOT.is_relative_to(PILOT)
    assert not BASE.is_relative_to(ROOT) and not PILOT.is_relative_to(ROOT)
