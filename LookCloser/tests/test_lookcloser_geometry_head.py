import torch

from nerfstudio.fields.lookcloser_geometry_head import LookCloserGeometryHead


def test_independent_surface_geometry_keeps_copied_radiance_and_ignores_density_logit():
    from nerfstudio.fields.lookcloser_surface_field import LookCloserSurfaceField

    class Encoding(torch.nn.Identity):
        n_output_dims = 3

    class RadianceFeatures(torch.nn.Module):
        def forward(self, features):
            return torch.cat([torch.full_like(features[:, :1], 10000.), .3+features[:, :1]], -1)

    class Color(torch.nn.Module):
        def forward(self, inputs):
            return inputs[:, :1].expand(-1, 3)

    field = LookCloserSurfaceField.__new__(LookCloserSurfaceField)
    torch.nn.Module.__init__(field)
    field.register_buffer('aabb', torch.tensor([[-1., -1., -1.], [1., 1., 1.]]))
    field.continuous_sdf = True
    field.enable_feature_reweighting = True
    field.encoding = Encoding()
    field.mlp_geo = RadianceFeatures()
    field.mlp_color = Color()
    field.opacity_head = torch.nn.Linear(1, 1)
    field._normalize_positions = lambda p: ((p+1)/2, torch.ones(len(p), dtype=torch.bool), p.shape[:-1], p)
    field._encode_with_optional_reweighting = lambda p, query_positions_flat: p*.5
    field.encode_directions = lambda d: d
    field._activate_rgb = lambda c: c
    field.enable_independent_geometry()
    positions = torch.tensor([[.2, 0., 0.]])
    directions = torch.tensor([[1., 0., 0.]])
    sdf, color, _ = field.query_surface(positions, directions)
    assert torch.allclose(sdf, torch.tensor([[-.05]]))
    assert torch.allclose(color, torch.full((1, 3), .6))
    with torch.no_grad():
        field.geometry_head.network[-1].bias.fill_(.1)
    shifted, unchanged, _ = field.query_surface(positions, directions)
    assert torch.allclose(shifted, torch.tensor([[.05]]))
    assert torch.equal(color, unchanged)


def test_zero_residual_preserves_initial_shape_despite_large_radiance_features():
    head = LookCloserGeometryHead(4)
    positions = torch.tensor([[.6, .5, .5], [.5, .7, .5]], requires_grad=True)
    features = torch.full((2, 4), 10000., dtype=torch.float16)
    residual = head(features, positions)
    assert residual.dtype == torch.float32
    assert torch.equal(residual, torch.zeros_like(residual))
    sdf = (positions - .5).norm(dim=-1, keepdim=True) - .25 + residual
    gradient, = torch.autograd.grad(sdf.sum(), positions)
    assert torch.allclose(gradient.norm(dim=-1), torch.ones(2))


def test_geometry_head_can_fit_a_plane_without_changing_radiance_features():
    torch.manual_seed(17)
    head = LookCloserGeometryHead(2, hidden_dim=16)
    positions = torch.rand(32, 3)
    features = torch.zeros(32, 2)
    target = .1 * (positions[:, :1] - .5)
    optimizer = torch.optim.Adam(head.parameters(), lr=.02)
    for _ in range(80):
        optimizer.zero_grad()
        error = (head(features, positions) - target).square().mean()
        error.backward()
        optimizer.step()
    assert (head(features, positions) - target).abs().max() < .005
    assert torch.equal(features, torch.zeros_like(features))
