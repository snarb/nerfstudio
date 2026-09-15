"""Additional real G/A-H/A stereo pair, covering the lower wrist better."""
import argparse
from pathlib import Path
from joint_temporal_texture import sha,atomic_json
ROOT=Path('/mnt/data/dec5_foundation_wrist_stereo')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','infer']);a=p.parse_args()
    if a.action=='stage':
        import stage_foundation_hand_stereo as source
        source.ROOT=ROOT;source.PAIRS=[('G004_A005_121071','H004_A005_1210M6')];source.stage()
        atomic_json(ROOT/'adapter.json',dict(script_sha256=sha(__file__),producer_sha256=sha(source.__file__),
            request_sha256=sha(ROOT/'001037/request.json'),pair=source.PAIRS))
    else:
        import infer_foundation_hand_stereo as source
        source.ROOT=ROOT/'001037';source.run()
