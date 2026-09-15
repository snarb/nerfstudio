"""Additional six real train cameras to diagnose a narrow silhouette envelope."""
from pathlib import Path
from joint_temporal_texture import cameras,sha,atomic_json
import study_wrist_observations as source

ROOT=Path('/mnt/data/dec5_wrist_wide_observations')


if __name__=='__main__':
    rows,_,_=cameras('001037')
    prefixes=('E004_A','E004_B','E004_C','F004_A','F004_C','F004_D')
    source.NAMES=[r['physical_camera'] for r in rows if r['physical_camera'].startswith(prefixes)]
    assert len(source.NAMES)==6
    source.stage(ROOT,'001037')
    atomic_json(ROOT/'wrapper.json',dict(script_sha256=sha(__file__),actual_camera_names=source.NAMES,
        source_stage_sha256=sha(source.__file__),frame_request_sha256=sha(ROOT/'001037/request.json'),
        original_observations_unchanged=True,heldout_used=False))
