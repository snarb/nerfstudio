import torch

from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components.lookcloser_colliders import LookCloserAABBBoxCollider
from nerfstudio.model_components.scene_colliders import AABBBoxCollider


def test_consistent_near_removes_only_camera_prefix_and_preserves_box_entry():
    box = SceneBox(aabb=torch.tensor([[-1., -1., -1.], [1., 1., 1.]]))
    origins = torch.tensor([[0., 0., 0.], [-2., 0., 0.]])
    directions = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
    old = AABBBoxCollider(box, near_plane=.05)
    fixed = LookCloserAABBBoxCollider(box, near_plane=.05, eval_use_training_near_plane=True)
    legacy = LookCloserAABBBoxCollider(box, near_plane=.05)
    training = old._intersect_with_aabb(origins, directions, box.aabb)
    assert all(torch.equal(a, b) for a, b in zip(training, fixed._intersect_with_aabb(origins, directions, box.aabb)))
    old.eval(); fixed.eval(); legacy.eval()
    historical = old._intersect_with_aabb(origins, directions, box.aabb)
    assert historical[0][0] == 0
    assert all(torch.equal(a, b) for a, b in zip(historical, legacy._intersect_with_aabb(origins, directions, box.aabb)))
    evaluated = fixed._intersect_with_aabb(origins, directions, box.aabb)
    assert evaluated[0][0] == .05
    assert evaluated[0][1] == historical[0][1]
    assert all(torch.equal(a, b) for a, b in zip(training, evaluated))


def test_near_plane_beyond_box_keeps_positive_dummy_interval():
    box = SceneBox(aabb=torch.tensor([[-1., -1., -1.], [1., 1., 1.]]))
    collider = LookCloserAABBBoxCollider(box, near_plane=2., eval_use_training_near_plane=True).eval()
    near, far = collider._intersect_with_aabb(torch.zeros(1, 3), torch.tensor([[1., 0., 0.]]), box.aabb)
    assert near.item() == 2.
    assert (far > near).all()
