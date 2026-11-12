"""Validate the actual delivered timeline, calibrated crop rays and saved weights."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def validate(root, out, verified_weights=None):
    root, out = Path(root), Path(out)
    path = json.loads((out/'path.json').read_text())
    inventory = {c['name']:c for c in json.loads((out/'camera_inventory.json').read_text())['cameras']}
    from flying_path import full_intrinsics
    native = full_intrinsics(inventory['BC'])
    rows = path['records']
    assert len(rows) == 84
    assert rows[-1]['source_frame']-rows[0]['source_frame']+1 == 360
    assert np.all(np.diff([r['source_frame'] for r in rows]) > 0)
    bc = np.array(inventory['BC']['pose'])
    for row in rows:
        pose = np.array(row['pose']);rot = pose[:3,:3]
        np.testing.assert_allclose(rot.T@rot,np.eye(3),atol=1e-6)
        assert abs(np.linalg.det(rot)-1)<1e-6
        assert np.isfinite(row['intrinsics']).all()
        assert min(row['intrinsics'][:2])>0
        if row['kind']=='native_gt':
            np.testing.assert_allclose(pose,bc,atol=1e-10)
            x,y,w,h=row['native_crop'];fx,fy,cx,cy=row['intrinsics']
            # Pixel-center ray equivalence between selected native window and output.
            u,v=1473.5,237.5
            np.testing.assert_allclose([(u-cx)/fx,(v-cy)/fy],
                [(x+u*w/1920-native[2])/native[0],(y+v*h/1080-native[3])/native[1]],atol=1e-12)
    assert rows[-1]['native_crop'][2:]==[1920,1080]
    np.testing.assert_allclose(rows[69]['pose'],rows[70]['pose'],atol=1e-12)
    np.testing.assert_allclose(rows[69]['intrinsics'],rows[70]['intrinsics'],atol=1e-12)
    stored={r['frame']:r for r in json.loads((root/'sparse24_storage_audit.json').read_text())['results']}
    model_checks=[]
    previous={r['frame']:r['sha256'] for r in json.loads(Path(verified_weights).read_text())['checkpoint_checks']} if verified_weights else {}
    for row in rows[:70]:
        saved=stored[row['source_frame']]
        digest=previous[row['source_frame']] if verified_weights else hashlib.sha256(Path(saved['path']).read_bytes()).hexdigest()
        assert digest==saved['model_sha256']
        receipt=json.loads((out/'neural'/f'{row["index"]:03}'/'receipt.json').read_text())
        assert receipt['checkpoint_sha256']==digest
        model_checks.append(dict(frame=row['source_frame'],sha256=digest))
    # Consecutive close-ups for temporal inspection: no interpolated RGB frames.
    contact=Image.new('RGB',(7*320,2*320))
    for j,row in enumerate(rows[70:]):
        im=Image.open(out/'gt'/f'{row["index"]:03}.png')
        im=im.crop((900,100,1540,740)).resize((320,320))
        ImageDraw.Draw(im).text((8,8),str(row['source_frame']),fill='white')
        contact.paste(im,((j%7)*320,(j//7)*320))
    contact.save(out/'hair_temporal_contact.jpg')
    result=dict(timeline=True,proper_rotations=True,native_crop_ray_parity=True,endpoint_pose_and_intrinsics=True,
        checkpoint_checks=model_checks,weight_audit_reused_from=verified_weights,output_frames=84,source_range=[908,1267],source_fps=120,video_fps=24,
        source_action_seconds=3,video_seconds=3.5,slowmo_speed=.2)
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VALIDATED timeline, rotations, crop rays, exact splice,70 unchanged weights',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--verified-weights');a=p.parse_args();validate(a.root,a.output,a.verified_weights)
