"""Compare finished candidates at fixed evaluation-only reference regression tiles."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw
from luster_hybrid_common import read,sha,write


def run(root,reference,names,output,count=3):
    assert count>0 and names
    tiles=root/'evaluation'/f'{reference}_spatial'/'regression_tiles.json'
    protocol=root/'evaluation/protocol.json';data=read(tiles);p=read(protocol)
    out=root/'evaluation'/output;out.mkdir(exist_ok=True);inputs={str(x):sha(x) for x in [tiles,protocol]}
    for target in [7,12,95,150]:
        rows=[r for r in data['rows'] if r['camera']==target][:count]
        paths=[Path(p['cameras'][str(target)]['source']),root/'baseline'/f'cam{target:03d}.png']
        paths += [root/'candidates'/n/f'cam{target:03d}.png' for n in names]
        titles=['GT','LookCloser',*names];images=[Image.open(x).convert('RGB') for x in paths]
        inputs.update({str(x):sha(x) for x in paths})
        sheet=Image.new('RGB',(256*len(images),292*len(rows)),(25,25,25));draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(rows):
            box=tuple(row['box']);assert box[2]-box[0]==box[3]-box[1]==128
            for i,(title,im) in enumerate(zip(titles,images)):
                # Labels wrap so long candidate identifiers remain inspectable.
                label=title if len(title)<37 else title[:36]+'\n'+title[36:]
                draw.text((256*i+3,292*j+2),label,fill='white')
                sheet.paste(im.crop(box).resize((256,256),Image.Resampling.NEAREST),(256*i,292*j+36))
        sheet.save(out/f'cam{target:03d}.png')
    write(out/'receipt.json',dict(usage='evaluation_only_never_construction',reference=reference,
        candidates=names,tiles_per_camera=count,zoom='2x nearest, native 128px crops',inputs=inputs,code_sha256=sha(Path(__file__))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',required=True)
    p.add_argument('--names',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=3)
    a=p.parse_args();run(a.root,a.reference,a.names,a.output,a.count)
