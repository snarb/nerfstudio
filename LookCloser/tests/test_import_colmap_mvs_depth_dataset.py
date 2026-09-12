from __future__ import annotations

import importlib.util
from pathlib import Path
import struct

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "import_colmap_mvs_depth_dataset.py"
SPEC = importlib.util.spec_from_file_location("import_colmap_mvs_depth_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_colmap_dense(path: Path, values: np.ndarray) -> None:
    height, width, channels = values.shape
    with path.open("wb") as handle:
        handle.write(f"{width}&{height}&{channels}&".encode())
        values.transpose(1, 0, 2).reshape(-1, order="F").astype(np.float32).tofile(handle)


def test_read_colmap_dense_array_preserves_pixel_layout(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    path = tmp_path / "depth.bin"
    write_colmap_dense(path, values)

    loaded = MODULE.read_colmap_dense_array(path)

    assert loaded.shape == values.shape
    assert np.array_equal(loaded, values)


def test_normalized_name_removes_dot_prefix() -> None:
    assert MODULE.normalized_name("./images/a.jpg") == "images/a.jpg"


def test_import_snapshot_does_not_require_writable_filesystem_timestamps(tmp_path, monkeypatch):
    import argparse
    import json
    data=tmp_path/'data';(data/'images').mkdir(parents=True)
    raw=tmp_path/'maps';(raw/'images').mkdir(parents=True)
    output=tmp_path/'imported'
    payload={'frames':[{'file_path':'images/frame_train_00000.jpg'}],
             'train_filenames':['images/frame_train_00000.jpg']}
    (data/'transforms.json').write_text(json.dumps(payload))
    write_colmap_dense(raw/'images/frame_train_00000.jpg.geometric.bin',np.ones((2,3,1),np.float32))
    args=argparse.Namespace(data=data,depth_maps=raw,output=output,input_type='geometric',colmap_model=None,undistorted_images=None)
    monkeypatch.setattr(MODULE,'parse_args',lambda:args)
    def forbidden_metadata_copy(*args,**kwargs):
        raise PermissionError('NFS forbids timestamp metadata copy')
    monkeypatch.setattr(MODULE.shutil,'copystat',forbidden_metadata_copy)
    assert MODULE.main()==0
    assert (output/'transforms.source.json').read_bytes()==(data/'transforms.json').read_bytes()
    result=json.loads((output/'transforms.json').read_text())
    assert result['colmap_mvs_depth']['train_depth_count']==1


def test_binary_pinhole_calibration_reader_has_no_pycolmap_dependency(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    with (model / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<QiiQQ4d", 1, 7, 1, 1918, 1079, 1000.0, 1001.0, 958.5, 539.0))
    with (model / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Qi7di", 1, 11, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7))
        stream.write(b"images/camera.jpg\0")
        stream.write(struct.pack("<Q", 0))

    calibration = MODULE.load_binary_pinhole_calibration(model)

    assert calibration["images/camera.jpg"] == {
        "fl_x": 1000.0,
        "fl_y": 1001.0,
        "cx": 958.5,
        "cy": 539.0,
        "w": 1918,
        "h": 1079,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }
