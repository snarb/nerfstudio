from pathlib import Path
import json,hashlib,shutil,time
import cv2
root=Path('/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109');out=root/'operations/sparse24_review_20261111';images=out/'images';images.mkdir(exist_ok=True)
selection=json.loads((root/'sparse24_selected_models.json').read_text())['frames'];sources=json.loads((root/'sparse24_gc_sources.json').read_text());rows=[]
video=out/'Marshawn908_1263_GC_24fps_fullhd.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),24,(1920,1080));assert writer.isOpened()
try:
 for index,row in enumerate(selection):
  frame=row['frame'];p=Path(sources[str(frame)]) if str(frame) in sources else out/'input'/f'{frame:06}'/'GC_raw_fullhd.png';receipt=json.loads((p.parent/'render_receipt.json').read_text());assert receipt['checkpoint_sha256']==row['model_sha256'],frame
  im=cv2.imread(str(p));assert im.shape==(1080,1920,3),(frame,im.shape)
  target=images/f'{index:03}.png';shutil.copy2(p,target);writer.write(im)
  rows.append(dict(frame=frame,index=index,model_sha256=row['model_sha256'],image_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source=str(p)))
finally:writer.release()
capture=cv2.VideoCapture(str(video));assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==72;assert capture.get(cv2.CAP_PROP_FPS)==24;capture.release()
receipt=dict(time=time.time(),frames=72,fps=24,duration_seconds=3,width=1920,height=1080,calibrated_camera='GC',training_rgb_resized=False,video=str(video),video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),preferred_repaired_frames=[923,988],rows=rows)
(out/'preview_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}))
