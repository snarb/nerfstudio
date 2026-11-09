#!/usr/bin/env python3
"""Compare existing HD PNG baseline and PQ-code predictions against ONE canonical GT.

No radiance-field loading/rendering or exposure fitting. The old fixed camera EV
is applied to BOTH new prediction and canonical GT; the PNG baseline is already
baked and is copied/scored unchanged. All producer code buffers remain untouched.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,shutil
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path('/fsx/tmp/marshawn_001660_hd')
CAMERAS=('DC','HC','KA')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for value in iter(lambda:f.read(8<<20),b''):h.update(value)
    return h.hexdigest()


def fixed_exposures(contract):
    if contract.get('eval_rgb_used_to_fit_gains') is not False or contract.get('global_base_gain')!=4:
        raise ValueError('Expected original train-derived fixed scalar exposure contract, basegain4')
    values={camera:float(contract['eval_gain_predictions'][camera]['ev']) for camera in CAMERAS}
    if not np.isfinite(list(values.values())).all():raise ValueError('Nonfinite fixed camera EV')
    for camera,ev in values.items():
        rows=[row for row in contract['rows'] if row['camera']==camera]
        if len(rows)!=1 or not np.isclose(rows[0]['exposure_correction_ev'],ev,rtol=0,atol=1e-12):
            raise ValueError('Old fixed exposure row/prediction mismatch')
    return values


def display_ap1_relative(linear,matrix,ev):
    linear=np.asarray(linear,dtype=np.float64)
    rec709=np.maximum(linear @ np.asarray(matrix,dtype=np.float64).T,0.)
    exposed=rec709*(4.*2.**float(ev))
    reinhard=exposed/(1.+exposed)
    return np.where(reinhard<=.0031308,12.92*reinhard,1.055*np.power(reinhard,1./2.4)-.055)


def display_codes(codes,vendor,matrix,gain,ev):
    # The vendor uses a FLOOR, not an additive offset: never subtract black.
    relative=vendor.pq_decode(codes)/float(gain)
    return display_ap1_relative(relative,matrix,ev)


def quantize_png(value):
    if not np.isfinite(value).all():raise ValueError('Nonfinite display RGB')
    return np.rint(np.clip(value,0,1)*255.).astype(np.uint8)


def native_rgb(array):
    value=np.asarray(array)
    if value.shape!=(1012,1920,3) or not np.isfinite(value).all():
        raise ValueError('Comparison requires finite native1920x1012 RGB; previews must not be upscaled')
    return value


def difference(first,second,data_range=None):
    delta=np.asarray(first,dtype=np.float64)-np.asarray(second,dtype=np.float64)
    absolute=np.abs(delta);mse=float(np.mean(delta**2))
    return dict(mean_abs=float(absolute.mean()),p95_abs=float(np.quantile(absolute,.95)),
        p99_abs=float(np.quantile(absolute,.99)),max_abs=float(absolute.max()),
        fraction_nonzero=float(np.mean(absolute!=0)),mse=mse,
        data_range=data_range,psnr_db=-10*math.log10(mse/data_range**2) if mse>0 and data_range else None,exact=mse==0)


def roundtrip_audit(codes,master,vendor,matrix,gain,ev,black=.005,peak=4000.):
    master=np.asarray(master,dtype=np.float64);codes=np.asarray(codes,dtype=np.float64)
    raw_nits=master*gain;ideal_nits=np.clip(raw_nits,black,peak)
    ideal_code=vendor.pq_encode(ideal_nits)
    expected_quantized=np.rint(ideal_code*65535.)/65535.
    decoded_nits=vendor.pq_decode(codes)
    raw_png=quantize_png(display_ap1_relative(master,matrix,ev))
    clipped_png=quantize_png(display_ap1_relative(ideal_nits/gain,matrix,ev))
    decoded_png=quantize_png(display_codes(codes,vendor,matrix,gain,ev))
    return dict(
        code_error_in_u16_units=difference(codes*65535.,expected_quantized*65535.),
        decoded_nits_vs_clipped_master=difference(decoded_nits,ideal_nits),
        decoded_relative_AP1_vs_raw_master=difference(decoded_nits/gain,master),
        policy_floor_channel_fraction=float(np.mean(raw_nits<black)),
        policy_peak_channel_fraction=float(np.mean(raw_nits>peak)),
        display_total_roundtrip_8bit_code=difference(decoded_png,raw_png,255.),
        display_policy_floor_peak_only_8bit_code=difference(clipped_png,raw_png,255.),
        display_quantization_precision_only_8bit_code=difference(decoded_png,clipped_png,255.),
        notes=['Master EXR is float32; re-quantizing it may differ by oneu16 code from the originalfloat64 producer.',
               'Floor and content-peak changes are intentional codec policy, separate from neural prediction error.',
               'Display audit PSNR uses range255; other physical-error summaries have no arbitrary PSNR range.'])


def load_exr(path):
    import os
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
    import cv2
    value=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if value is None or value.ndim!=3 or value.shape[2]!=3 or value.dtype!=np.float32:
        raise ValueError('Expected3-channel float32EXR: '+str(path))
    return value[...,::-1].copy()


def dataset_frames(data):
    meta=json.loads((data/'transforms.json').read_text());rows={f['physical_camera']:f for f in meta['frames']}
    return {camera:data/rows[camera]['file_path'] for camera in CAMERAS}


def metric_engine(device):
    import torch
    from torchmetrics.functional.image import structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    torch.set_num_threads(2)
    perceptual=LearnedPerceptualImagePatchSimilarity(normalize=True).eval().to(device)
    def score(prediction,target):
        p=torch.from_numpy(prediction.copy()).permute(2,0,1)[None].to(device=device,dtype=torch.float32)/255.
        t=torch.from_numpy(target.copy()).permute(2,0,1)[None].to(device=device,dtype=torch.float32)/255.
        with torch.inference_mode():
            mse=float(torch.mean((p-t)**2))
            result=dict(psnr=-10*math.log10(mse) if mse>0 else None,
                ssim=float(structural_similarity_index_measure(p,t,data_range=1.)),lpips=float(perceptual(p,t)))
        return result
    return score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new-code-predictions',type=Path)
    p.add_argument('--baseline-predictions',type=Path,default=ROOT/'final_pair/download/marshawn_1660_final_candidate')
    p.add_argument('--canonical-pq-data',type=Path,default=ROOT/'color_variants/canonical_signed_ap1_pq_codes_exr')
    p.add_argument('--canonical-linear-data',type=Path,default=ROOT/'color_variants/canonical_signed_ap1/linear_ap1_exr')
    p.add_argument('--exposure-contract',type=Path,default=ROOT/'color_variants/gain4_scalar_trainfit_poseeval/sdr_png/color_contract.json')
    p.add_argument('--color-audit',type=Path,default=ROOT/'report/canonical_colour_independent/audit.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args();contract=json.loads(a.exposure_contract.read_text());exposures=fixed_exposures(contract)
    from audit_marshawn_body_rays import geometry_hash
    geometries={}
    splits={}
    for label,data in [('canonical_codes',a.canonical_pq_data),('canonical_master',a.canonical_linear_data),('historical_scalar',a.exposure_contract.parent)]:
        meta=json.loads((data/'transforms.json').read_text())
        geometries[label]=geometry_hash(meta['frames'])
        splits[label]=sorted(Path(name).stem.split('_')[-1] for name in meta['train_filenames'])
    if len(set(geometries.values()))!=1 or any(value!=splits['canonical_codes'] for value in splits.values()):
        raise ValueError('GT/master/historical scalar camera geometry or split mismatch')
    audit=json.loads(a.color_audit.read_text());vendor_path=Path(audit['source']['vendor_pq'])
    if sha(vendor_path)!=audit['source']['vendor_sha256']:raise ValueError('Pinned vendor color source changed')
    spec=importlib.util.spec_from_file_location('pinned_vendor_PQ',vendor_path);vendor=importlib.util.module_from_spec(spec);spec.loader.exec_module(vendor)
    matrix=np.asarray(audit['matrices']['AP1_D60_to_Rec709_D65_Bradford'])
    if not np.allclose(matrix,vendor.rgb_to_rgb_matrix('ap1','rec709'),rtol=0,atol=1e-12):raise ValueError('Color matrix or transpose/whitepoint convention mismatch')
    gain=float(audit['source_bounds']['canonical_gain_to_nits'])
    if not np.isclose(gain,52.13360318135631,rtol=0,atol=1e-12):raise ValueError('Unexpected signed gain')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'raw').mkdir()
    code_frames=dataset_frames(a.canonical_pq_data);master_frames=dataset_frames(a.canonical_linear_data)
    score=metric_engine(a.device);rows=[];panels=[]
    for camera in CAMERAS:
        gt_codes=native_rgb(load_exr(code_frames[camera]));master=native_rgb(load_exr(master_frames[camera]))
        ev=exposures[camera];gt=quantize_png(display_codes(gt_codes,vendor,matrix,gain,ev))
        baseline_path=a.baseline_predictions/(camera+'.png');baseline=native_rgb(np.asarray(Image.open(baseline_path).convert('RGB')))
        Image.fromarray(gt).save(a.output/('GT_'+camera+'.png'))
        shutil.copy2(baseline_path,a.output/('baseline_'+camera+'.png'))
        np.savez_compressed(a.output/'raw'/('GT_'+camera+'_codes.npz'),rgb=gt_codes)
        row=dict(camera=camera,fixed_old_EV=ev,fixed_total_display_gain=4*2**ev,
            baseline_png_sha256=sha(baseline_path),canonical_input_codes_sha256=sha(code_frames[camera]),
            canonical_linear_master_sha256=sha(master_frames[camera]),
            baseline_metrics=score(baseline,gt),roundtrip=roundtrip_audit(gt_codes,master,vendor,matrix,gain,ev))
        oldrow=next(row for row in contract['rows'] if row['camera']==camera)
        old_gt_path=a.exposure_contract.parent/oldrow['png_file']
        if sha(old_gt_path)!=oldrow['png_sha256']:raise ValueError('Historical scalar target changed')
        old_gt=np.asarray(Image.open(old_gt_path).convert('RGB'));row['old_target_to_common_canonical_GT_8bit_code']=difference(old_gt,gt,255.)
        images=[gt,baseline]
        if a.new_code_predictions:
            candidates=[path for path in [a.new_code_predictions/f'eval_{camera}_buffers.npz',a.new_code_predictions/f'{camera}_buffers.npz'] if path.exists()]
            if len(candidates)!=1:raise ValueError('Missing/ambiguous native raw prediction buffer for '+camera)
            path=candidates[0]
            with np.load(path,allow_pickle=False) as source:
                prediction_codes=native_rgb(source['rgb'].copy())
                if 'gt'in source and not np.array_equal(source['gt'],gt_codes):raise ValueError('Prediction buffer GT differs from canonical input at native raster')
            if prediction_codes.min()<0 or prediction_codes.max()>1:raise ValueError('Predicted PQ codes outside codec0..1')
            prediction=quantize_png(display_codes(prediction_codes,vendor,matrix,gain,ev))
            Image.fromarray(prediction).save(a.output/('newPQ_'+camera+'.png'))
            target=a.output/'raw'/path.name;shutil.copy2(path,target)
            if sha(path)!=sha(target):raise ValueError('Raw producer buffer copy changed')
            row.update(new_code_buffer_sha256=sha(path),new_metrics=score(prediction,gt),
                metric_delta_new_minus_baseline={k:None for k in ['psnr','ssim','lpips']})
            for key in row['metric_delta_new_minus_baseline']:
                if row['new_metrics'][key] is not None and row['baseline_metrics'][key] is not None:
                    row['metric_delta_new_minus_baseline'][key]=row['new_metrics'][key]-row['baseline_metrics'][key]
            images.append(prediction)
        panels.append(images);rows.append(row)
        (a.output/'progress.json').write_text(json.dumps(dict(cameras_complete=len(rows),total=3))+'\n')
    averages={}
    for label in ['baseline_metrics','new_metrics']:
        if label not in rows[0]:continue
        averages[label]={key:float(np.mean([row[label][key] for row in rows])) if all(row[label][key]is not None for row in rows) else None for key in ['psnr','ssim','lpips']}
    canvas=Image.new('RGB',(640*len(panels[0]),361*3),'#202020');draw=ImageDraw.Draw(canvas)
    for r,(camera,images) in enumerate(zip(CAMERAS,panels)):
        for c,image in enumerate(images):
            canvas.paste(Image.fromarray(image).resize((640,337),Image.Resampling.LANCZOS),(640*c,361*r+24))
            draw.text((640*c+4,361*r+5),camera+' '+['canonicalGT','acceptedPNG unchanged','newPQ mapped'][c],fill='white')
    canvas.save(a.output/'common_GT_predictions_contact.jpg',quality=95)
    report=dict(implementation_sha256=sha(__file__),no_field_rendering=True,no_training=True,
        native_raster=[1920,1012],GT_source='Decoded actual canonical PQ input, not oldPNG',
        display='EOTF/gain→AP1D60to709D65→positive clamp→4*2^OLDfixedEV→Reinhard→sRGB→rounduint8',
        black_floor_subtracted=False,baseline_exposure_applied_again=False,new_prediction_and_GT_same_fixed_EV=True,
        exposure_fitted=False,exposure_contract_sha256=sha(a.exposure_contract),vendor_sha256=sha(vendor_path),
        input_camera_geometry_sha256=geometries,
        color_audit_sha256=sha(a.color_audit),AP1_to_709_matrix=matrix.tolist(),signed_gain=gain,
        metric_protocol='All3 arithmetic mean, normalized quantized8bit commonSDR, SSIM data_range1, LPIPS alex normalizeTrue',
        no_GT_compositing=True,raw_prediction_buffers_preserved=True,averages=averages,rows=rows,
        limitations=['These common-display metrics supersede cross-domain comparison of oldSDR and nativePQ metrics.',
            'Old exposure corrections are fixed train-derived controls, not new eval fitting.',
            'Codec floor/peak and finite precision produce a measured GT roundtrip difference; audit separates them.',
            'Numerical PQ-code compositing remains distinct from physical radiance integration.'])
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(averages))


if __name__=='__main__':main()
