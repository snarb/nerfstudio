import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from study_forearm_tsdf_scale import command_for


def test_only_voxel_output_and_interpreter_change():
    original=['python','fuse.py','--output','old.ply','--voxel-length','.0005','--sdf-trunc','.004','--tensor-weight-threshold','2']
    result=command_for(original,Path('/tmp/study/mesh.ply'),.001)
    assert original[3]=='old.ply' and original[5]=='.0005'
    assert result[3]=='/tmp/study/mesh.ply' and result[5]=='0.001'
    assert result[1:3]==original[1:3] and result[6:]==original[6:]
