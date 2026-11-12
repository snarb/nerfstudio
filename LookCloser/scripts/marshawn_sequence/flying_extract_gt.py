"""Export native BC source crops and original alpha; no RGB resize."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,cv2,numpy as np,hashlib
root=Path('/home/nvuser/lookcloser/campaigns/marshawn908_20261109')
sys.path.insert(0,str(root/'tools'))
from prepare import cameras,decode_rgb,display
out=Path('/raid/lookcloser/flying_strict_20261112');out.mkdir(exist_ok=True)
cv2.setNumThreads(1)
def one(frame):
 row=cameras(frame)['BC'];source=cv2.imread(row['source'],cv2.IMREAD_UNCHANGED)
 # Common overscan around all moving1920x1080 windows; decoded native pixels.
 x,y,w,h=2665,650,2120,1450
 rgb=np.uint8(np.clip(display(decode_rgb(source[y:y+h,x:x+w,::-1]))*255+.5,0,255))
 maskpath=Path('/raid/lookcloser/masks8k')/f'{frame:06}'/f'BC_{frame:06}.png'
 mask=cv2.imread(str(maskpath),cv2.IMREAD_GRAYSCALE)[y:y+h,x:x+w]
 cv2.imwrite(str(out/f'{frame:06}_rgb.png'),rgb[...,::-1]);cv2.imwrite(str(out/f'{frame:06}_alpha.png'),mask)
 return dict(frame=frame,source=row['source'],source_crop=[x,y,x+w,y+h],source_mask=str(maskpath),rgb_native_no_resize=True,calibration=row)
with ThreadPoolExecutor(4) as pool:rows=list(pool.map(one,range(1251,1270)))
(out/'manifest.json').write_text(json.dumps(dict(camera='BC',rows=rows),indent=2));print('EXPORTED',len(rows))
