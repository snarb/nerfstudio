"""Evaluation-only nail source diagnostics; never imported by a predictor."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from luster_hybrid_common import read,write
ROOT=Path('/fsx/tmp/luster_000540_hybrid_20261017T104440Z')
p=read(ROOT/'evaluation/protocol.json');selection=read(ROOT/'preflight/sources.json')['sources'];stats={}
for camera in [7,12,95,150]:
    row=p['cameras'][str(camera)];box=row['boxes']['nails'];x0,y0,x1,y1=box
    gt=np.asarray(Image.open(row['source']))[y0:y1,x0:x1]
    base=np.load(ROOT/f'geometry/cam{camera:03d}/rgb.npy',mmap_mode='r')[y0:y1,x0:x1]
    tiles=[('GT: evaluation only',gt),('LookCloser',np.uint8(np.clip(base,0,1)*255))];stats[str(camera)]={}
    for source in selection[str(camera)][:16]:
        d=ROOT/f'warps/cam{camera:03d}/src{source:03d}';r=read(d/'receipt.json');bx,by,_,_=r['box'];sl=np.s_[y0-by:y1-by,x0-bx:x1-bx]
        a={k:np.asarray(np.load(d/f'{k}.npy',mmap_mode='r')[sl]) for k in r['arrays']}
        good=(a['valid']>0)&(a['alpha']>=.99)&(a['depth_error']<=.005)&(a['roundtrip']<=1)&(a['uncertainty']<=2)&(a['resolution_ratio']>=.5)
        stats[str(camera)][str(source)]={'valid_fraction':float(good.mean()),'roundtrip_median':float(np.median(a['roundtrip'])),'uncertainty_median':float(np.median(a['uncertainty'])),'resolution_median':float(np.median(a['resolution_ratio']))}
        rgb=np.uint8(np.clip(a['rgb'],0,1)*255)
        tiles.append((f'src{source:03d} valid {good.mean():.0%} rt {np.median(a["roundtrip"]):.1f}',rgb))
    tw,th=360,330;canvas=Image.new('RGB',(tw*6,th*3),(35,35,35));draw=ImageDraw.Draw(canvas)
    for i,(title,rgb) in enumerate(tiles):
        tile=Image.fromarray(rgb);tile.thumbnail((tw,th-25));xx=(i%6)*tw;yy=(i//6)*th
        canvas.paste(tile,(xx,yy+25));draw.text((xx+4,yy+5),title,fill='white')
    canvas.save(ROOT/f'preflight/cam{camera:03d}_nail_single_sources.jpg',quality=95)
write(ROOT/'preflight/nail_source_diagnostics.json',stats)
