"""OpenEXR dataset loading tests."""

from io import BytesIO

import cv2
import numpy as np
import pytest
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import load_exr_image, write_exr_image


def _write_exr(path, rgb: np.ndarray) -> None:
    assert cv2.imwrite(str(path), rgb[..., ::-1])


def _outputs(path) -> DataparserOutputs:
    return DataparserOutputs(
        image_filenames=[path],
        cameras=Cameras(
            camera_to_worlds=torch.eye(4, dtype=torch.float32)[None, :3],
            fx=1.0,
            fy=1.0,
            cx=1.0,
            cy=1.0,
            width=2,
            height=2,
        ),
    )


def test_load_exr_preserves_rgb_range_and_bytes(tmp_path):
    rgb = np.array(
        [
            [[-0.25, 0.5, 4.0], [0.0, 1.0, 8.0]],
            [[0.125, 2.0, 0.25], [1.5, -0.125, 0.75]],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "image.exr"
    _write_exr(path, rgb)

    decoded = load_exr_image(path)
    decoded_bytes = load_exr_image(BytesIO(path.read_bytes()))
    np.testing.assert_allclose(decoded, rgb, atol=2e-3, rtol=0)
    np.testing.assert_allclose(decoded_bytes, decoded, atol=0, rtol=0)
    assert decoded.dtype == np.float32
    assert decoded.flags.c_contiguous


def test_input_dataset_exr_requires_float32_cache(tmp_path):
    rgb = np.full((2, 2, 3), 2.5, dtype=np.float32)
    path = tmp_path / "image.exr"
    _write_exr(path, rgb)
    dataset = InputDataset(_outputs(path), cache_compressed_images=True)

    item = dataset.get_data(0, image_type="float32")
    torch.testing.assert_close(item["image"], torch.from_numpy(rgb))
    with pytest.raises(ValueError, match="cache_images_type='float32'"):
        dataset.get_data(0, image_type="uint8")


def test_load_exr_resizes_in_float(tmp_path):
    rgb = np.linspace(-1.0, 3.0, 4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
    path = tmp_path / "image.exr"
    _write_exr(path, rgb)
    decoded = load_exr_image(path, scale_factor=0.5)
    assert decoded.shape == (2, 3, 3)
    assert decoded.dtype == np.float32
    assert float(decoded.max()) > 1.0


def test_write_exr_round_trip(tmp_path):
    rgb = torch.tensor([[[-0.25, 0.5, 4.0], [8.0, 1.0, 0.0]]], dtype=torch.float32)
    path = tmp_path / "render.exr"
    write_exr_image(path, rgb)
    np.testing.assert_allclose(load_exr_image(path), rgb.numpy(), atol=2e-3, rtol=0)
