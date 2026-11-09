"""Opt-in patch for isolated diagnostic launchers, not the production parser.

Import before training or evaluation to read model-space stage bounds from
data/geometry_teacher_bounds.json. The candidate review box is separate.
"""
import json,torch
from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio
_original=Nerfstudio._generate_dataparser_outputs
def _outputs(self,*args,**kwargs):
 result=_original(self,*args,**kwargs)
 p=self.config.data/'geometry_teacher_bounds.json'
 if p.exists():result.scene_box.aabb=torch.tensor(json.loads(p.read_text())['model_aabb'],dtype=torch.float32)
 return result
Nerfstudio._generate_dataparser_outputs=_outputs
