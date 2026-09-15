"""Native-size side-effect sheets and hash-bound sealing, not automatic approval."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from study_subface_free_space import ROOT,FRAME
from review_measured_free_surface import VIEWS


def sheets():
    records=[]
    for view in VIEWS:
        folder=ROOT/FRAME/'review'/view/'new_black';r=read(folder/'result.json')
        patches=[]
        for row in r['components']:
            assert sha(row['path'])==row['sha256']
            im=Image.open(row['path']);assert im.width<=300 and im.height<=130
            patches.append((im.copy(),row))
        sheet=Image.new('RGB',(900,150*((len(patches)+2)//3)),(40,40,40))
        draw=ImageDraw.Draw(sheet)
        for i,(im,row) in enumerate(patches):
            x=(i%3)*300;y=(i//3)*150
            draw.text((x+3,y+2),f"{view} component {row['component']} : {row['pixels']} px",fill='white')
            sheet.paste(im,(x,y+20))
        dest=folder/'all_native.png';sheet.save(dest)
        records.append(dict(view=view,path=str(dest),sha256=sha(dest),
                            component_receipt_sha256=sha(folder/'result.json'),all_components_shown=True,native_size=True))
    atomic_json(ROOT/FRAME/'review/side_effect_sheets.json',dict(records=records,script_sha256=sha(__file__)))


def seal():
    root=ROOT/FRAME;bindings={}
    audit=read(root/'independent_audit.json');bindings.update(audit['bindings'])
    assert audit['status']=='passed'
    bindings[str(root/'result.json')]=audit['result_sha256']
    bindings[str(Path(__file__).with_name('audit_subface_free_space.py'))]=audit['script_sha256']
    for view in VIEWS:
        folder=root/'review'/view;r=read(folder/'result.json');bindings.update(r['bindings'])
        bindings.update({str(folder/n):h for n,h in r['hashes'].items()})
        bindings[str(Path(__file__).with_name('review_subface_free_space.py'))]=r['script_sha256']
        local=read(folder/'new_black/result.json')
        bindings[str(folder/'result.json')]=local['render_review_sha256']
        bindings[str(Path(__file__).with_name('localize_subface_black_pixels.py'))]=local['script_sha256']
        bindings.update({row['path']:row['sha256'] for row in local['components']})
    sheets=read(root/'review/side_effect_sheets.json')
    for row in sheets['records']:
        bindings[row['path']]=row['sha256']
        bindings[str(Path(row['path']).parent/'result.json')]=row['component_receipt_sha256']
    verdict=read(root/'visual_review.json')
    assert verdict['status'] not in ['pending','uncertain'] and verdict['production_promoted'] is False
    bindings.update(verdict['reviewed_images'])
    for path,h in bindings.items():assert sha(path)==h,path
    local={str(p):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='final_audit.json'}
    atomic_json(root/'final_audit.json',dict(status='passed',binding_count=len(bindings),file_count=len(local),
        bindings=bindings,files=local,script_sha256=sha(__file__),visual_status=verdict['status'],
        production_promoted=False,artifact_free_approval=False))
    print('sealed',len(bindings),'bindings',len(local),'local files',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['sheets','seal'])
    globals()[p.parse_args().action]()
