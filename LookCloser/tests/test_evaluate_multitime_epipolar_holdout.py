from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_multitime_epipolar_holdout import temporal_windows,records_for


def test_temporal_window_and_camera_inventory_extraction():
    audit={'results':[{'frame_id':'000105','secondary_camera':'B',
        'window':{'-1':'000103','0':'000105','1':'000107'},'records':[
            {'reference_xy':[1,2],'matches':{'0':{'point':[3,4]}},'reference_index':7,'spatial_block':[0,0]}]}]}
    assert temporal_windows(audit)=={'000103','000105','000107'}
    assert records_for(audit,'A')==[]
    assert records_for(audit,'B')[0]=={'frame_id':'000105','a':[1,2],'b':[3,4],'reference_index':7,'group':['000105',0,0]}
