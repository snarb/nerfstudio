from pathlib import Path
import sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from render_bandwidth_model_control import save_control_images


def test_prediction_and_source_map_are_written_in_same_output_directory(tmp_path):
    labels=np.array([[0,1,-1],[1,0,-1]],np.int32)
    rgb=np.zeros((2,3,3),np.uint8);rgb[labels==0]=[10,20,30];rgb[labels==1]=[40,50,60]
    directory=tmp_path/'control';save_control_images(directory,rgb,labels,2)
    np.testing.assert_array_equal(np.asarray(Image.open(directory/'eval_pred_0000.png')),rgb)
    selection=np.asarray(Image.open(directory/'source_selection.png'))
    np.testing.assert_array_equal(selection[labels==0],np.tile([230,25,75],(2,1)))
    assert not selection[labels<0].any()
