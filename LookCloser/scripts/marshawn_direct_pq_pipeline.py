"""Explicit encoded-PQ control; frozen native NN, geometry auxiliaries and grid.

PQ codes are integrated numerically, NOT as physical radiance. Display requires
the fixed offline EOTF/color transform. This file does not alter the HDR model.
"""
import json
from pathlib import Path
import torch
from marshawn_final_pipeline import FinalNativePipeline
from staged_native_hash import file_sha


class DirectPQNativePipeline(FinalNativePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config
        if file_sha(__file__) != cfg.direct_pq_source_sha256:
            raise ValueError('Direct PQ wrapper changed')
        if cfg.final_grid_radius != 0 or cfg.final_geometry_variant != 'scene_loose':
            raise ValueError('Control requires original scene_loose/native core0')
        if cfg.datamanager.cache_images_type != 'float32':
            raise ValueError('PQ codes require numerical float32 cache')
        contract = json.loads((Path(cfg.datamanager.dataparser.data)/'color_contract.json').read_text())
        if contract['eval_rgb_used_to_fit_gains'] is not False:
            raise ValueError('No heldout RGB fitting')
        floor = float(cfg.direct_pq_floor_code)
        if abs(floor - 988/65535.) > 1e-12:
            raise ValueError('Expected archived canonical 0.005-nit PQ floor')
        # Existing renderer supports a fixed RGB tensor. This is a plain attribute,
        # not a parameter/buffer; checkpoint NN schema remains native unchanged.
        self.model.renderer_rgb.background_color = torch.full((3,), floor, device=self.device)

