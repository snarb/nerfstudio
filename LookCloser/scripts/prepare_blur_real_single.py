"""Real-RGB single-camera saturation diagnostic; its eval copy is not held out."""
from copy import deepcopy
import json
from pathlib import Path


def main():
    root=Path('/home/brans/lookcloser_artifacts/blur_ablation_fresh')
    source=root/'real';teacher=root/'synthetic_single'
    real=json.loads((source/'transforms.json').read_text())
    synthetic=json.loads((teacher/'transforms.json').read_text())
    frame=next(r for r in real['frames'] if Path(r['file_path']).stem=='train_0033')
    reference=next(r for r in synthetic['frames'] if Path(r['file_path']).stem=='train_0033')
    assert frame['transform_matrix']==reference['transform_matrix']
    out=root/'real_single';out.mkdir(exist_ok=True)
    (out/'images').mkdir(exist_ok=True)
    for name in ['train_0033.png','eval_same_train_0033.png']:
        path=out/'images'/name
        if not path.exists():path.symlink_to(source/frame['file_path'])
    for name,target in [('masks',teacher/'masks'),('lookcloser_frequencies',source/'lookcloser_frequencies')]:
        if not (out/name).exists():(out/name).symlink_to(target)
    train=deepcopy(frame);train['mask_path']=reference['mask_path']
    evaluation=deepcopy(train);evaluation['file_path']='images/eval_same_train_0033.png'
    metadata=dict(camera_model='OPENCV',coordinate_system=real['coordinate_system'],
                  blur_aabb=synthetic['blur_aabb'],diagnostic_same_view=True,
                  note='Real RGB, teacher validity mask for train sampling only. Eval duplicates train; no novel-view claim.',
                  frames=[train,evaluation],train_filenames=[train['file_path']],
                  val_filenames=[evaluation['file_path']],test_filenames=[evaluation['file_path']])
    (out/'transforms.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
