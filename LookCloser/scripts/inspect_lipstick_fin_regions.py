"""Native coordinate grid to distinguish the tube wedge from the nail-side flap."""
from pathlib import Path
from PIL import Image,ImageDraw
from study_multiview_face_prior import read,save,sha
from study_query_support_quorum import ROOT,FRAME
from review_full_block_transfer import ROOT as DEPTH_ROOT
from diagnose_lipstick_fin_depth import CAMERA,BOX


def main():
    out=ROOT/FRAME/'region_inspection';assert not out.exists();out.mkdir()
    paths=[DEPTH_ROOT/FRAME/'review'/CAMERA/'train_gt.png',ROOT/FRAME/'rgb'/CAMERA/'frames'/FRAME/'frame.png']
    images=[]
    for path in paths:
        im=Image.open(path).convert('RGB').crop(BOX);grid=im.copy();draw=ImageDraw.Draw(grid)
        for x in range(((BOX[0]+19)//20)*20,BOX[2],20):
            draw.line([(x-BOX[0],0),(x-BOX[0],im.height)],fill=(180,180,180),width=1)
            draw.text((x-BOX[0]+1,1),str(x),fill='red')
        for y in range(((BOX[1]+19)//20)*20,BOX[3],20):
            draw.line([(0,y-BOX[1]),(im.width,y-BOX[1])],fill=(180,180,180),width=1)
            draw.text((1,y-BOX[1]+1),str(y),fill='red')
        images.extend([im,grid])
    sheet=Image.new('RGB',(images[0].width*4,images[0].height+24));draw=ImageDraw.Draw(sheet)
    for i,(im,name) in enumerate(zip(images,['GT','GT native grid','quorum3','quorum3 native grid'])):
        sheet.paste(im,(i*im.width,24));draw.text((i*im.width+2,2),name,fill='white')
    sheet.save(out/'grid.png')
    save(out/'result.json',dict(inputs={str(p):sha(p) for p in paths},box=BOX,
        script_sha256=sha(__file__),grid_sha256=sha(out/'grid.png'),diagnostic_only=True))


if __name__=='__main__':main()
