"""Native-resolution contact sheets and explicit black-pixel witnesses."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from study_multiview_face_prior import read,save,sha
from review_face_interior_visibility import panel

ROOT=Path('/mnt/data/dec5_face_angular_visibility')


def main():
    dest=ROOT/'transfer_review';assert not dest.exists();dest.mkdir();records=[]
    for f in ['001083','001119','001127']:
        review=ROOT/f/'review';names=sorted(review.glob('changed_*.png'))
        for start in range(0,len(names),9):
            paths=names[start:start+9];ims=[Image.open(p) for p in paths];width=max(x.width for x in ims);height=max(x.height for x in ims)+20
            out=Image.new('RGB',(3*width,((len(ims)+2)//3)*height));draw=ImageDraw.Draw(out)
            for i,(p,im) in enumerate(zip(paths,ims)):
                xy=((i%3)*width,(i//3)*height);draw.text((xy[0]+2,xy[1]+2),p.stem,fill='white');out.paste(im,(xy[0],xy[1]+20))
            path=dest/f'{f}_native_{start//9:02d}.png';out.save(path)
            records.append(dict(path=str(path),sha256=sha(path),native_no_resize=True,inputs={str(p):sha(p) for p in paths}))
    f='001083';r=read(ROOT/f/'consensus/result.json');base=Path(r['baseline'])
    a=np.array(Image.open(base/'frame.png'));b=np.array(Image.open(ROOT/f/'consensus/frame.png'))
    y,x=np.nonzero((a.max(2)>0)&(b.max(2)==0));points=[]
    for i,(xx,yy) in enumerate(zip(x,y)):
        box=[max(0,int(xx)-45),max(0,int(yy)-45),min(1080,int(xx)+46),min(1920,int(yy)+46)]
        crops=[]
        for rgb in [a,b]:
            im=Image.fromarray(rgb).crop(box);draw=ImageDraw.Draw(im)
            dx=int(xx)-box[0];dy=int(yy)-box[1];draw.rectangle((dx-3,dy-3,dx+3,dy+3),outline='magenta');crops.append(im)
        path=dest/f'001083_black_{i}.png';panel(crops,['baseline','candidate'],path)
        points.append(dict(portrait_xy=[int(xx),int(yy)],old_rgb=a[yy,xx].tolist(),new_rgb=b[yy,xx].tolist(),image=str(path),sha256=sha(path)))
    save(dest/'inventory.json',dict(contact_sheets=records,new_black_pixels=points,visual_status='pending'))
    print('native sheets',len(records),'black crops',len(points),flush=True)


if __name__=='__main__':main()
