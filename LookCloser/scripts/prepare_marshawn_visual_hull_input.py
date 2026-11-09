"""Source-bound adapter from reviewed two-actor masks to reusable hull inputs."""
from pathlib import Path
import argparse,hashlib,json,sys,os,copy
import numpy as np
from PIL import Image

R=Path('/fsx/tmp/marshawn_001660_hd')
sys.path[:0]=[str(R/'code_native'),str(Path(__file__).parent)]
from marshawn_actor_coordinates import raw_mask_coverage_to_HD

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(source,identity,request,output,reuse=None):
    source,identity,request=map(Path,[source,identity,request]);m=json.loads(source.read_text());ids=json.loads(identity.read_text())
    if sha(identity)!=m['identity_manifest_sha256']:raise ValueError('Reviewed stable person identities changed')
    if len(m['rows'])!=66 or not m['native_hair66_complete']:raise ValueError('All66 reviewed native masks required')
    sources={r['camera']:r for r in json.loads(request.read_text())['records']};identityrows={r['camera']:r for r in ids['rows']}
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'masks').mkdir();rows=[]
    prior=json.loads((Path(reuse)/'manifest.json').read_text())if reuse else None
    priorrows={r['camera']:r for r in prior['cameras']}if prior else {}
    if prior and prior['identity_manifest_sha256']!=sha(identity):raise ValueError('Cannot reuse differently selected identities')
    tf=json.loads((Path(m['source_data_dir'])/'transforms.json').read_text());frames={r['physical_camera']:r for r in tf['frames']}
    for row in m['rows']:
        cam=row['camera'];idrow=identityrows[cam];rawsrc=sources[cam]
        previous=priorrows.get(cam)
        if previous and previous['source_archive_sha256']==row['archive_sha256']and previous['raw_archive_sha256']==row['raw_archive_sha256']:
            oldmask=Path(reuse)/previous['masks']
            if sha(oldmask)!=previous['masks_sha256']:raise ValueError('Reusable prepared mask changed')
            p=output/'masks'/(cam+'.npz');os.link(oldmask,p);record=copy.deepcopy(previous)
            record.update(masks=str(p.relative_to(output)),source_archive=row['archive'],raw_archive=row['raw_archive'])
            rows.append(record);print(cam,'reused exactsource',flush=True);continue
        if sha(row['archive'])!=row['archive_sha256']or sha(row['raw_archive'])!=row['raw_archive_sha256']:raise ValueError('Reviewed mask bytes changed')
        if sha(idrow['candidates_npz'])!=idrow['source_candidate_npz_sha256']:raise ValueError('Instance archive changed')
        with np.load(row['archive'],allow_pickle=False)as z:
            foreground=z['nominal_coverage']>0;unknown=z['source_ineligible'].copy()
        # Ambiguous/unselected instances are UNKNOWN for geometry, not RGB vetoes.
        # No detector rank or score picks a target; only the reviewed global IDs.
        with np.load(idrow['candidates_npz'],allow_pickle=False)as z:
            masks=z['masks'];selected=[idrow['black_actor_instance'],idrow['grey_actor_instance']]
            unselected=[i for i in range(len(masks))if i not in selected]
            ambiguous=masks[unselected].any(0)if unselected else np.zeros_like(masks[0])
        full=np.zeros((4320,8192),np.float32);x0,y0,x1,y1=rawsrc['crop_xyxy'];full[y0:y1,x0:x1]=ambiguous
        unknown|=raw_mask_coverage_to_HD(full,row['camera_row'])>1e-6;del full,ambiguous,masks
        valid_path=Path(m['source_data_dir'])/frames[cam]['mask_path']
        valid=np.asarray(Image.open(valid_path).convert('L'))>0
        if valid.shape!=foreground.shape:raise ValueError('Source valid raster mismatch')
        p=output/'masks'/(cam+'.npz');np.savez_compressed(p,foreground=foreground,unknown=unknown,valid=valid)
        rows.append(dict(camera=cam,masks=str(p.relative_to(output)),masks_sha256=sha(p),camera_row=row['camera_row'],
            native_scale_yx=[4320/foreground.shape[0],8192/foreground.shape[1]],
            stable_instance_global_ids={str(selected[0]):'fighter_black',str(selected[1]):'fighter_grey'},
            selected_global_ids=['fighter_black','fighter_grey'],all_unselected_person_instances_unknown=True,
            source_archive=row['archive'],source_archive_sha256=row['archive_sha256'],raw_archive=row['raw_archive'],
            raw_archive_sha256=row['raw_archive_sha256'],valid_source=str(valid_path),valid_source_sha256=sha(valid_path),
            foreground_pixels=int(foreground.sum()),unknown_pixels=int(unknown.sum()),source_rgb=row['rgb'],source_rgb_sha256=row['rgb_sha256']))
        print(cam,rows[-1]['foreground_pixels'],rows[-1]['unknown_pixels'],flush=True)
    boxsource=R/'geometry/boxes_approved.json';prep=R/'data/preparation.json'
    payload=dict(kind='REVIEWED_ACTOR_MASKS_VISUAL_HULL_INPUT_NOT_TRAINING_RELEASE',source_manifest=str(source),source_manifest_sha256=sha(source),
        identity_manifest=str(identity),identity_manifest_sha256=sha(identity),source_request=str(request),source_request_sha256=sha(request),
        search_domain=json.loads(boxsource.read_text())['scopes']['scene']['loose'],search_domain_source=str(boxsource),
        search_domain_source_sha256=sha(boxsource),normalization_source=str(prep),normalization_sha256=sha(prep),
        coordinate_system='prepared_world; physical unit unspecified',
        foreground_policy='ANYpositive exactRAW_AREA+calibration-remap nominalcoverage, including acceptednativehair',
        unknown_policy='Verifiedother/explicitunknown plus ALLunselectedhumaninstances andinvalid source; never treatedclearBG',
        source_RGB_and_calibration_unchanged=True,cameras=rows,adapter_sha256=sha(__file__))
    (output/'manifest.json').write_text(json.dumps(payload,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse',type=Path);a=p.parse_args();main(a.source,a.identity,a.request,a.output,a.reuse)
