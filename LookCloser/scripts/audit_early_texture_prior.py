"""Audit the source-admission-only change against the same completed meshes."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
import study_forearm_plane_transfer_v3 as prior
from study_early_texture_prior import BASE


def run(root):
    prior.configure();name=prior.v2.v1.NAMES[1];records=[]
    for frame in ['001029','001033','001037']:
        for view in ['moving',name]:
            depths=[];results=[];images=[];labels=[];requests=[]
            for source in [BASE,root]:
                dest=source/'rgb'/frame/view/'guarded';folder=dest/'frames'/frame
                request=read(dest/'request.json');receipt=read(folder/'complete.json')
                if receipt['request_sha256']!=sha(dest/'request.json'):raise ValueError('Changed request')
                for p,h in receipt['hashes'].items():
                    if sha(folder/p)!=h:raise ValueError('Changed output')
                result=read(folder/'result.json');results.append(result);requests.append(request)
                if result['target_rgb_read'] or result['rgb_averaging'] or len(result['source_cameras'])!=62:raise ValueError('Changed source protocol')
                depths.append(np.load(folder/'target_depth.npz')['depth'])
                images.append(np.array(Image.open(folder/'frame.png')))
                labels.append(np.rot90(np.array(Image.open(folder/'source_ids.png'))))
            if requests[0]['inventory']!=requests[1]['inventory']:raise ValueError('Changed mesh, camera or source identity')
            if not np.array_equal(depths[0],depths[1]):raise ValueError('Changed target depth')
            if results[0]['source_cameras']!=results[1]['source_cameras'] or results[0]['fixed_exposure']!=results[1]['fixed_exposure']:raise ValueError('Changed camera profiles or exposure')
            r=dict(frame=frame,view=view,target_depth_exact=True,inventory_exact=True,source_changed_pixels=int((labels[0]!=labels[1]).sum()),
                black_pixel_counts=[int((im.max(2)==0).sum()) for im in images],
                scope='pixel counts are source/coverage diagnostics, not full-frame quality metrics')
            if view==name:
                mask=np.rot90(prior.v2.v1.masks(frame)[name]);own=results[0]['source_cameras'].index(name)
                r.update(fixed_skin_pixels=int(mask.sum()),own_camera_skin_pixels=[int((mask&(im==own)).sum()) for im in labels],
                    fixed_skin_black_pixels=[int((mask&(im.max(2)==0)).sum()) for im in images])
            records.append(r)
    metrics={}
    for label,source in [('late',BASE),('early',root)]:
        metrics[label]=[r for r in read(source/'metrics.json')['rows'] if r['variant']=='guarded']
    atomic_json(root/'audit.json',dict(records=records,metrics=metrics,script_sha256=sha(__file__),
        geometry_changed=False,source_images_changed=False,full_frame_quality_metrics=False,production_promoted=False))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_forearm_early_texture_prior'));run(p.parse_args().root)
