"""LookCloser ray bounds with an optional consistent evaluation near plane."""
import torch

from nerfstudio.model_components.scene_colliders import AABBBoxCollider


class LookCloserAABBBoxCollider(AABBBoxCollider):
    def __init__(self, scene_box, near_plane=0.0, eval_use_training_near_plane=False):
        super().__init__(scene_box=scene_box, near_plane=near_plane)
        self.eval_use_training_near_plane = eval_use_training_near_plane

    def _intersect_with_aabb(self, rays_o, rays_d, aabb):
        nears, fars = super()._intersect_with_aabb(rays_o, rays_d, aabb)
        if not self.training and self.eval_use_training_near_plane:
            nears = nears.clamp_min(self.near_plane)
            fars = torch.maximum(fars, nears + 1e-6)
        return nears, fars
