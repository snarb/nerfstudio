from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_nerfstudio_colmap_model.py"
SPEC = importlib.util.spec_from_file_location("export_nerfstudio_colmap_model", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_export_model_is_train_only_and_preserves_calibration(tmp_path: Path) -> None:
    data = tmp_path / "data"
    images = data / "images"
    images.mkdir(parents=True)
    for name in ("train.jpg", "eval.jpg"):
        Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(images / name)
    identity = np.eye(4).tolist()
    payload = {
        "camera_model": "OPENCV",
        "fl_x": 10.0,
        "fl_y": 11.0,
        "cx": 4.0,
        "cy": 3.0,
        "w": 8,
        "h": 6,
        "k1": 0.1,
        "k2": -0.02,
        "p1": 0.003,
        "p2": -0.004,
        "frames": [
            {"file_path": "images/train.jpg", "transform_matrix": identity, "colmap_im_id": 7},
            {"file_path": "images/eval.jpg", "transform_matrix": identity, "colmap_im_id": 9},
        ],
        "train_filenames": ["images/train.jpg"],
        "val_filenames": ["images/eval.jpg"],
    }
    (data / "transforms.json").write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "model"
    manifest = MODULE.export_model(data, output, split="train")

    assert manifest["image_count"] == 1
    assert manifest["uses_eval_images"] is False
    assert manifest["uses_sparse_points"] is False
    camera_row = [line for line in (output / "cameras.txt").read_text().splitlines() if not line.startswith("#")][0]
    tokens = camera_row.split()
    assert tokens[:4] == ["7", "OPENCV", "8", "6"]
    assert np.allclose([float(value) for value in tokens[4:]], [10, 11, 4, 3, 0.1, -0.02, 0.003, -0.004])
    images_text = (output / "images.txt").read_text()
    assert "images/train.jpg" in images_text
    assert "images/eval.jpg" not in images_text


def test_opt_in_pinhole_rejects_distortion_and_preserves_four_parameters(tmp_path):
    data=tmp_path/'data';data.mkdir();Image.new('RGB',(8,6)).save(data/'train.jpg')
    payload={'fl_x':10.,'fl_y':11.,'cx':4.,'cy':3.,'w':8,'h':6,'k1':.01,
             'frames':[{'file_path':'train.jpg','transform_matrix':np.eye(4).tolist()}],
             'train_filenames':['train.jpg']}
    (data/'transforms.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='zero distortion'):
        MODULE.export_model(data,tmp_path/'bad',split='train',camera_model='PINHOLE')
    payload['k1']=0.;(data/'transforms.json').write_text(json.dumps(payload))
    MODULE.export_model(data,tmp_path/'good',split='train',camera_model='PINHOLE')
    row=[r for r in (tmp_path/'good/cameras.txt').read_text().splitlines() if not r.startswith('#')][0]
    assert row.split()[1]=='PINHOLE'
    assert list(map(float,row.split()[4:]))==[10,11,4,3]
