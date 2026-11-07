"""CPU-only RGB-confidence profile feasibility; never edits training sources."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from marshawn_actor_coordinates import maps


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def profile(distance,delta):return (.1+.9*np.exp(-np.asarray(distance,dtype=np.float64)**2/(2*delta**2))).astype(np.float32)


def summarize(weights,masks):
    return {name:dict(pixels=int(mask.sum()),sum_weight=float(weights[mask].sum(dtype=np.float64)),mean_weight=float(weights[mask].mean(dtype=np.float64))if mask.any()else None)for name,mask in masks.items()}


def main(c,out):
    cv2.setNumThreads(2);out.mkdir(parents=True,exist_ok=False);(out/'profiles').mkdir()
    # c is tuning root; historical source archive lives beside tuning_20261106.
    conf=c.parent/'two_actor_isolation/confidence_native_hair_v2/manifest.json'
    if not conf.exists():conf=c.parent.parent/'two_actor_isolation/confidence_native_hair_v2/manifest.json'
    source=json.loads(conf.read_text());sources={r['camera']:r for r in source['rows']}
    tf=json.loads((c/'data/transforms.json').read_text());train=set(tf['train_filenames']);frames=[f for f in tf['frames']if f['file_path']in train];assert len(frames)==63
    bagroot=c/'semantic_actor_bagloss_prepared_20261106/bag_loss_source';bagmeta=json.loads((bagroot/'manifest.json').read_text());assert bagmeta['expected_false_grey_pixels']==6522 and bagmeta['uses_EVAL']is False
    for name,digest in bagmeta['files_sha256'].items():assert sha(bagroot/name)==digest,name
    bags={r['camera']:r for r in bagmeta['records']};rows=[];removed=0;start=time.monotonic()
    for frame in frames:
        name=frame['physical_camera'];src=sources[name];rawpath=Path(src['raw_archive']);assert sha(rawpath)==src['raw_archive_sha256'];archive=Path(src['archive']);assert sha(archive)==src['archive_sha256']
        with np.load(rawpath)as z:nominal=z['nominal'];veto=z['source_ineligible'];black=z['black'];grey=z['grey'];box=z['crop_xyxy']
        raw_hard=nominal&~veto;extra=raw_hard&~(black|grey)
        # Distances are measured on the original RAW lattice, in HD-equivalent units.
        # Signed EDT + linear centre sampling gives a continuous level-set boundary;
        # every original hard RAW pixel centre is on its multiplier-one side.
        spacing=(frame['h']/4320,frame['w']/8192)
        signed=distance_transform_edt(~raw_hard,sampling=spacing)-distance_transform_edt(raw_hard,sampling=spacing)
        mx,my=maps(src['camera_row']);mx=mx-float(box[0]);my=my-float(box[1])
        signed_hd=cv2.remap(signed.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=1e6)
        inside=signed_hd<=0;distance=np.maximum(signed_hd,0)
        raw_nearest=cv2.remap(raw_hard.astype(np.uint8),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
        hair_added=cv2.remap(extra.astype(np.uint8),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
        with np.load(c/'data/semantics'/f'{name}.npz')as z:sem={k:z[k]for k in ['W64','core','band','valid','nominal','black','grey']}
        current=sem['W64'];bag_only=current.copy();bagmask=np.zeros(current.shape,bool);shell=np.zeros(current.shape,bool)
        if name in bags:
            b=bags[name];assert sha(c/'data/semantics'/f'{name}.npz')==b['original_semantic_sha256'];ex=np.load(bagroot/b['exclusion']);shell=np.load(bagroot/b['unknown']);bagmask=ex&sem['grey'];assert not (ex&shell).any()and not(bagmask&sem['black']).any()and int(bagmask.sum())==b['false_grey_pixels'];bag_only[bagmask]=0;removed+=int(bagmask.sum())
        m2=profile(distance,2);m4=profile(distance,4);assert (m2[inside]==1).all()and(m4[inside]==1).all()
        weights={'original':current,'bag6522':bag_only,'delta2':bag_only*m2,'delta4':bag_only*m4}
        for w in weights.values():assert np.isfinite(w).all()and(w>=0).all()and(w<=1).all()and(w[~sem['valid']]==0).all()
        assert (weights['delta2']<=weights['delta4']+1e-7).all()and(weights['delta4']<=bag_only+1e-7).all()
        for key in ['delta2','delta4']:np.testing.assert_array_equal(weights[key][inside&~bagmask],current[inside&~bagmask]);assert (weights[key][bagmask]==0).all()
        masks={'inside_RAW_union':inside,'outside_RAW_union':~inside,'current_core_outside_RAW':sem['core']&~inside,'current_band':sem['band'],'qualified_bag6522':bagmask,'bag_unknown_shell':shell,'reviewed_hair_added_centres':hair_added,'RAW_vs_nearest_boundary_disagreement':inside^raw_nearest}
        for lo,hi in [(0,1),(1,2),(2,4),(4,8),(8,16),(16,1e9)]:masks[f'outside_distance_{lo}_{hi}_HD']=(distance>lo)&(distance<=hi)
        path=out/'profiles'/f'{name}.npz';np.savez_compressed(path,distance_HD=distance,inside_RAW_union=inside,multiplier_delta2=m2,multiplier_delta4=m4,weight_bag6522=bag_only,weight_delta2=weights['delta2'],weight_delta4=weights['delta4'])
        row=dict(camera=name,source_raw=str(rawpath),source_raw_sha256=sha(rawpath),original_confidence_sha256=sha(archive),main_semantic_sha256=sha(c/'data/semantics'/f'{name}.npz'),profile_archive_sha256=sha(path),profile_arrays_sha256={k:array_sha(v)for k,v in weights.items()},RAW_crop_xyxy=box.tolist(),RAW_metric_spacing_HD=list(spacing),pixels=int(current.size),raw_hard_pixels=int(raw_hard.sum()),boundary_membership_disagreement=int((inside^raw_nearest).sum()),statistics={k:summarize(w,masks)for k,w in weights.items()})
        rows.append(row);print(json.dumps(dict(camera=name,seconds=round(time.monotonic()-start,2),outside_current=row['statistics']['original']['outside_RAW_union']['sum_weight'],outside_delta2=row['statistics']['delta2']['outside_RAW_union']['sum_weight'],outside_delta4=row['statistics']['delta4']['outside_RAW_union']['sum_weight'])),flush=True)
    assert removed==6522
    result=dict(kind='CPU_ONLY_CONTINUOUS_OUTSIDE_RGB_CONFIDENCE_PROPOSALS',train_cameras=63,frame=1660,source_manifest=str(conf),source_manifest_sha256=sha(conf),bag_manifest_sha256=sha(bagroot/'manifest.json'),transforms_sha256=sha(c/'data/transforms.json'),helper_sha256=sha(__file__),coordinate_helper_sha256=sha(Path(__file__).with_name('marshawn_actor_coordinates.py')),rig_cameras_sha256=sha(Path('/fsx/oregon/gta_marshawn_2_1_prep_20261028/root/Marshawn_2_1/golden/rig/cameras.txt')),contract='Multiply current original W64 by .1+.9exp(-d_out^2/(2delta^2)), delta2HD or4HD; exact6522bagzero in every repaired treatment. d_out=max(0,bilinearly centre-sampled RAW signed EDT), metric sampling(hHD/4320,wHD/8192); inside RAW hard-union levelset multiplier1. No AREA-prefilter of distances/HDcore EDT.',sampling='Original pools/FAS/indices/RNG unchanged; analytic E_q[w] must be recomputed under that same proposal.',rows=rows,training_started=False,source_modified=False,new_segmentation=False,temporal_flicker_established=False,limitations=['One frame only; temporal flicker is not established.','RAW signed-EDT levelset/nearest-mask boundary disagreements are reported; hard pixel centres remain unchanged.','Unsegmented hair outside the original hard union is attenuated, not certified background.','These are RGB confidence weights, never alpha/depth/free-space labels.','Physical floor/mat/crew occlusion uncertainty is unchanged.'])
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();main(a.campaign,a.output)
