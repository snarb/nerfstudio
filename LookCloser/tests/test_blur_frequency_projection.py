import torch
from nerfstudio.pipelines.lookcloser_pipeline import normalized_frequency_resolution


def test_projection_uses_camera_z_and_uv_units():
    # 100 cells / UV-width, fx/W=2; depth z=4, scene span=3 -> 150 cells.
    args = [torch.tensor(x) for x in [100., 2000., 1000., 1000., 1000., 5., 1.25, [3., 2., 1.]]]
    torch.testing.assert_close(normalized_frequency_resolution(*args), torch.tensor(150.))
    # Reexpress the same geometry in different world units and image resolution.
    args[1] *= 2; args[2] *= 2; args[3] *= 2; args[4] *= 2
    args[5] *= 7; args[7] *= 7
    torch.testing.assert_close(normalized_frequency_resolution(*args), torch.tensor(150.))
