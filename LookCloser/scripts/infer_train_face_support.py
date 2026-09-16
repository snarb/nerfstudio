"""Face-skin confidence from existing calibrated train crops; no target input.

Run in the isolated MediaPipe environment. The semantic prior is not depth or
ground-truth visibility. Does not overwrite the earlier hair-model experiment.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import mediapipe as mp
from build_train_hair_semantics import ROOT as SOURCE,read,write,sha

OUT=Path('/mnt/data/dec5_train_face_support_001123')
FRAME='001123'


def main():
    assert not OUT.exists();OUT.mkdir()
    q=read(SOURCE/'request.json');staged=read(SOURCE/FRAME/'stage.json')
    assert staged['request_sha256']==sha(SOURCE/'request.json')
    model=q['models']['multiclass'];assert sha(model['file'])==model['sha256']
    records=staged['records'];assert len(records)==62
    assert not {r['camera'] for r in records}&{'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
    write(OUT/'request.json',dict(frame=FRAME,source_stage=str(SOURCE/FRAME/'stage.json'),
        stage_sha256=sha(SOURCE/FRAME/'stage.json'),source_request_sha256=sha(SOURCE/'request.json'),
        crop=q['crop_portrait'],model=model,script_sha256=sha(__file__),
        mediapipe_version=mp.__version__,target_used=False,geometry_used=False,
        declared_channel='face-skin',records=records))
    options=mp.tasks.vision.ImageSegmenterOptions(base_options=mp.tasks.BaseOptions(
        model_asset_path=model['file'],delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,output_category_mask=False,output_confidence_masks=True)
    outputs=[]
    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
        assert segmenter.labels[3]=='face-skin'
        for i,r in enumerate(records):
            assert sha(r['input_path'])==r['input_sha256']
            rgb=np.array(Image.open(r['input_path']).convert('RGB'))
            masks=segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)).confidence_masks
            confidence=masks[3].numpy_view().copy();assert confidence.shape==rgb.shape[:2] and np.isfinite(confidence).all()
            path=OUT/(r['camera']+'.npz');np.savez_compressed(path,confidence=np.rint(confidence.clip(0,1)*255).astype(np.uint8))
            outputs.append(dict(camera=r['camera'],path=str(path),sha256=sha(path)))
            print(i+1,r['camera'],flush=True)
    write(OUT/'complete.json',dict(request_sha256=sha(OUT/'request.json'),outputs=outputs,
        probabilities_are_not_geometry_truth=True,visual_status='pending'))


if __name__=='__main__':main()
