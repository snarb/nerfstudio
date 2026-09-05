from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hard_source_seam_leveling import level_hard_source_seams
from patchmatch_color_calibration import apply_camera_gain


def test_single_sided_leveling_preserves_texture_and_primary():
    source=torch.full((3,32,40),.4);source[:,::2]=.45
    secondary=apply_camera_gain(source,[1.3,1.1,.9])
    selection=torch.zeros((32,40),dtype=torch.long);selection[8:24,10:30]=1
    pred=torch.where((selection==1)[None],secondary,source)
    valid=[torch.ones((32,40),dtype=torch.bool)]*2
    out,gain,stats=level_hard_source_seams(pred,selection,[source,secondary],valid,torch.ones((32,40)))
    torch.testing.assert_close(out,source,atol=1e-4,rtol=0)
    assert torch.equal(out[:,selection==0],source[:,selection==0])
    assert stats['source_labels_unchanged'] and not stats['source_averaging']


def test_depth_boundary_not_leveled_and_invalid_overlap_not_used():
    source=torch.full((3,20,30),.4);secondary=torch.full_like(source,.6)
    selection=torch.zeros((20,30),dtype=torch.long);selection[:,15:]=1
    pred=torch.where((selection==1)[None],secondary,source)
    depth=torch.ones((20,30));depth[:,15:]=2
    valid=[torch.ones_like(selection,dtype=torch.bool)]*2
    out,_,_=level_hard_source_seams(pred,selection,[source,secondary],valid,depth)
    torch.testing.assert_close(out,pred)
    valid=[valid[0],selection==1]
    out,_,_=level_hard_source_seams(pred,selection,[source,secondary],valid,torch.ones_like(depth))
    torch.testing.assert_close(out,pred)
