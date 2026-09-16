"""Pack every already generated side-effect crop at native pixel size."""
from PIL import Image,ImageDraw
from study_query_support_quorum import ROOT,FRAME
from study_multiview_face_prior import read,save,sha


def main():
    root=ROOT/FRAME/'review';r=read(root/'result.json');records=[]
    for row in r['records']:
        panels=[]
        for component in row['new_black_components']:
            p=component['path'];assert sha(p)==r['images'][str(__import__('pathlib').Path(p).relative_to(root))]
            panels.append((Image.open(p).copy(),component))
        width=max(im.width for im,c in panels)+8;height=max(im.height for im,c in panels)+28
        sheet=Image.new('RGB',(3*width,((len(panels)+2)//3)*height),(40,40,40));draw=ImageDraw.Draw(sheet)
        for i,(im,c) in enumerate(panels):
            x=(i%3)*width;y=(i//3)*height
            draw.text((x+2,y+2),f'{i}: {c["pixels"]} pixels',fill='white');sheet.paste(im,(x,y+24))
        path=root/row['view']/'all_side_effects_native.png';assert not path.exists();sheet.save(path)
        records.append(dict(view=row['view'],components=len(panels),path=str(path),sha256=sha(path)))
    save(root/'side_effect_sheets.json',dict(records=records,review_sha256=sha(root/'result.json'),native_size=True))


if __name__=='__main__':main()
