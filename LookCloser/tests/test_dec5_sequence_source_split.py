from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from render_dec5_native_head_benchmark import validate_source_split


def data(frame='000901'):
    rows=[dict(physical_camera='C'+str(i),file_path=str(i)+'.png') for i in range(65)]
    return dict(frame_id=frame,frames=rows,train_filenames=[r['file_path'] for r in rows[:64]],val_filenames=['64.png'])


def test_new64_split_is_explicit_and_does_not_relax_old_frame_exclusions():
    assert len(validate_source_split(data(),64))==64
    with pytest.raises(ValueError):validate_source_split(data('001055'),64)
    with pytest.raises(ValueError):validate_source_split(data())


def test_heldout_id_overlap_is_rejected_even_with64_training_files():
    value=data();value['frames'][-1]['physical_camera']='C0'
    with pytest.raises(ValueError,match='leaks'):validate_source_split(value,64)
