"""Combine independently reviewed body/lying mattes on the previous camera path."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build(root):
 root=Path(root);op=root/'operations';base=op/'flying_strict_low_20261112';body=op/'body_matte_cleanup_20261112/hybrid_body';lying=op/'lying_cleanup_20261112/matte';out=op/'flying_previous_bodyclean_20261112';out.mkdir(exist_ok=True)
 (out/'path.json').write_bytes((base/'path.json').read_bytes())
 for name,target in [('gt',base/'gt_matanyone2_soft'),('codec',base/'codec')]:
  if not (out/name).exists():(out/name).symlink_to(target,target_is_directory=True)
 guards=json.loads((body/'receipt.json').read_text())['frames'];yy,xx=np.mgrid[:1080,:1920];rows=[]
 for i in range(70):
  rgb=np.array(Image.open(base/'neural'/f'{i:03}'/'strict.png'));a=np.array(Image.open(body/'factor'/f'{i:03}.png'))/65535;b=np.array(Image.open(lying/f'{i:03}_factor.png'))/65535
  factor=np.minimum(a,b);guard=np.zeros((1080,1920),bool)
  for x0,y0,x1,y1 in guards[i]['guard_boxes_xyxy']:guard|=(xx>=x0)&(xx<=x1)&(yy>=y0)&(yy<=y1)
  factor[guard]=1
  clean=np.uint8(np.clip(rgb*factor[...,None]+.5,0,255));assert np.all(clean<=rgb);assert np.array_equal(rgb[guard],clean[guard])
  dest=out/'neural'/f'{i:03}';dest.mkdir(parents=True,exist_ok=True);Image.fromarray(clean).save(dest/'strict.png');Image.fromarray(np.uint16(factor*65535+.5)).save(dest/'factor.png')
  rows.append(dict(index=i,baseline_sha256=sha(base/'neural'/f'{i:03}'/'strict.png'),output_sha256=sha(dest/'strict.png'),changed_pixels=int(np.any(clean!=rgb,axis=-1).sum()),guard_max_change=0,body_factor_sha256=sha(body/'factor'/f'{i:03}.png'),lying_factor_sha256=sha(lying/f'{i:03}_factor.png')))
  if i in [0,10,20,30,35,40,45,55,60,69]:
   panel=Image.new('RGB',(1920,564));panel.paste(Image.fromarray(rgb).resize((960,540)),(0,24));panel.paste(Image.fromarray(clean).resize((960,540)),(960,24));ImageDraw.Draw(panel).text((10,5),f'{i} previous / cleaned',fill='white');panel.save(out/f'compare_{i:03}.jpg')
 tail=[dict(index=i,sha256=sha(out/'gt'/f'{i:03}.png')) for i in range(70,84)]
 (out/'cleanup_receipt.json').write_text(json.dumps(dict(method='minimum of body-boundary and lying-actor factors, explicit hair/head/prop guards override both',path_unchanged=sha(out/'path.json')==sha(base/'path.json'),frames=rows,unchanged_tail=tail,models_retrained=False),indent=2))
 sys.path.insert(0,str(root/'tools'))
 from flying_delivery import assemble
 from flying_encode import encode
 assemble(out);encode(out)
 old=out/'Marshawn908_1267_flying_strict_BC_slowmo_24fps.mp4';target=out/'Marshawn_previous_flight_body_clean_MatAnyone2.mp4';old.replace(target)
 receipt=json.loads((out/'delivery_receipt.json').read_text());receipt.update(video=str(target),cleanup_receipt_sha256=sha(out/'cleanup_receipt.json'));(out/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2));print('FINAL',target,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);build(p.parse_args().root)
