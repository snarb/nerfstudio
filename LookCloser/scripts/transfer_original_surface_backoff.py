"""Cross-time regression check of the frozen, exact-original-surface fallback.

Consumes completed all-radius studies; never reruns fitting or alters meshes.
The HD stills here are diagnostics, not the delivered native 6K video.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label, find_objects
from study_multiview_face_prior import read, save, sha


def audit_splice(base, raw, fixed, base_ids, raw_ids, fixed_ids, mask):
    """Independent exact-array audit, including every pixel outside the fallback."""
    assert mask.dtype == bool and mask.shape == raw.shape[:2]
    assert np.all(raw.max(2)[mask] == 0) and np.all(base.max(2)[mask] > 0)
    assert np.all(raw_ids[mask] == 255) and np.all(base_ids[mask] < 62)
    np.testing.assert_array_equal(fixed[mask], base[mask])
    np.testing.assert_array_equal(fixed[~mask], raw[~mask])
    np.testing.assert_array_equal(fixed_ids[mask], base_ids[mask])
    np.testing.assert_array_equal(fixed_ids[~mask], raw_ids[~mask])
    np.testing.assert_array_equal(np.any(fixed != raw, axis=2), mask)
    return dict(recovered=int(mask.sum()),
        new_black_before=int(((base.max(2)>0)&(raw.max(2)==0)).sum()),
        new_black_after=int(((base.max(2)>0)&(fixed.max(2)==0)).sum()))


def panel(images, path, box):
    x0,y0,x1,y1=box; w=x1-x0; h=y1-y0
    out=Image.new('RGB',(3*w,h+24)); draw=ImageDraw.Draw(out)
    for j,(name,im) in enumerate(images):
        out.paste(Image.fromarray(im).crop(box),(j*w,24))
        draw.text((j*w+2,4),name,fill='white')
    out.save(path)


def run_case(case, output):
    frame,view,root=case; src=Path(root)/'admission/rgb'/view
    base=src/'baseline/frames'/frame; raw=src/'interpolated/frames'/frame
    dest=output/frame/view; dest.parent.mkdir(parents=True,exist_ok=True)
    helper=Path(__file__).with_name('recover_original_surface_texture.py')
    with (output/'logs'/f'{frame}_{view}.log').open('x') as log:
        subprocess.run([sys.executable,str(helper),'--baseline',str(base),
            '--candidate',str(raw),'--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True)
    result=read(dest/'result.json'); bindings=dict(result['input_hashes'])
    for p,h in bindings.items(): assert sha(p)==h,p
    for p,h in result['hashes'].items(): assert sha(dest/p)==h,p
    rgb=[np.array(Image.open(p/'prediction_native.png').convert('RGB')) for p in [base,raw,dest]]
    ids=[np.array(Image.open(p/'source_ids.png')) for p in [base,raw,dest]]
    ev=np.load(dest/'evidence.npz'); mask=ev['mask']
    stats=audit_splice(*rgb,*ids,mask)
    assert np.all(abs(ev['base_depth']-ev['depth'])<=1e-7)
    assert np.all(abs(ev['base_bary']-ev['bary'])<=1e-6)
    # The output intentionally references candidate depth: no new depth file,
    # inferred geometry or new-surface color is produced by this policy.
    depth=np.load(raw/'target_depth.npz')['depth']; bd=np.load(base/'target_depth.npz')['depth']
    added=(depth>0)&(bd<=0)
    np.testing.assert_array_equal(rgb[2][added],rgb[1][added])
    stats.update(added_geometry=int(added.sum()),
        added_geometry_uncolored=int((added&(rgb[2].max(2)==0)).sum()),
        lost_geometry=int(((bd>0)&(depth<=0)).sum()))
    review=dest/'review';review.mkdir()
    images=list(zip(['production','completed mesh','same-surface backoff'],[np.rot90(r) for r in rgb]))
    record=next(r for r in read(Path(root)/'review/result.json')['records'] if r['view']==view)
    panel(images,review/'head_native.png',record['crop'])
    # Show all recovery AND remaining-regression components; no top-k hiding.
    inventory=[]
    for kind,area in [('recovery',mask),('remaining_black',(rgb[0].max(2)>0)&(rgb[2].max(2)==0)),
                      ('new_uncolored',added&(rgb[2].max(2)==0))]:
        labels,n=label(np.rot90(area));boxes=find_objects(labels)
        for i,box in enumerate(boxes):
            y,x=box;crop=[max(0,x.start-25),max(0,y.start-25),min(1080,x.stop+25),min(1920,y.stop+25)]
            name=f'{kind}_{i:03d}.png';panel(images,review/name,crop)
            inventory.append(dict(kind=kind,path=str(review/name),pixels=int((labels==i+1).sum()),crop=crop))
    for p in [base/'target_depth.npz',raw/'target_depth.npz',dest/'result.json',dest/'evidence.npz',helper,Path(__file__)]:
        bindings[str(p)]=sha(p)
    report=dict(frame=frame,view=view,statistics=stats,components=inventory,
        candidate_depth=str(raw/'target_depth.npz'),candidate_mesh=read(raw/'result.json')['mesh_path'],
        input_hashes=bindings,images={str(p):sha(p) for p in review.glob('*.png')},
        preserved_added_geometry_and_colors=True,geometry_modified_by_backoff=False,
        full_video_accepted=False,visual_status='pending',diagnostic_resolution=[1080,1920])
    save(dest/'audit.json',report)
    print(frame,view,stats,flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--studies',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert not out.exists()
    cases=[];bindings={}
    for root in a.studies:
        root=root.resolve();q=read(root/'request.json');adapter=read(root/'admission/rgb_adapter.json')
        assert adapter['request_sha256']==sha(root/'request.json')
        for path in [root/'request.json',root/'admission/rgb_adapter.json',root/'review/result.json']:
            bindings[str(path)]=sha(path)
        for view in adapter['views']: cases.append((q['frame'],view,str(root)))
    assert len({(f,v) for f,v,_ in cases})==len(cases)
    out.mkdir();(out/'logs').mkdir()
    save(out/'request.json',dict(cases=cases,input_hashes=bindings,script_sha256=sha(__file__),
        helper_sha256=sha(Path(__file__).with_name('recover_original_surface_texture.py')),
        production_modified=False,texture_policy_only=True))
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(lambda case:run_case(case,out),cases))
    save(out/'result.json',dict(request_sha256=sha(out/'request.json'),
        cases=[dict(frame=r['frame'],view=r['view'],statistics=r['statistics'],
            audit_sha256=sha(out/r['frame']/r['view']/'audit.json')) for r in records],
        visual_status='pending',full_video_accepted=False))


if __name__=='__main__':main()
