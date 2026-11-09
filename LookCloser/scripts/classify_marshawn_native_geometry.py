#!/usr/bin/env python3
"""Read-only checkpoint/saved-ray classification of SfM admission and band support."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def stats(values):
    return dict(min=float(np.min(values)),median=float(np.median(values)),
                p90=float(np.quantile(values,.9)),max=float(np.max(values)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    report=json.loads(args.audit.read_text());folder=args.audit.parent
    evidence=np.load(folder/'known_ray_evidence.npz',allow_pickle=False)
    traces=np.load(folder/'traces.npz',allow_pickle=False)
    state=torch.load(report['checkpoint'],map_location='cpu',weights_only=False)['pipeline']
    def tensor(suffix):
        values=[v for k,v in state.items()if k.endswith(suffix)]
        if not values:raise ValueError('Missing native state '+suffix)
        if not all(torch.equal(values[0],value)for value in values[1:]):
            raise ValueError('Native model/sampler shared-grid aliases disagree: '+suffix)
        return values[0]
    box=tensor('occupancy_grid.aabbs')[0];mask=tensor('occupancy_grid.binaries')[0]
    occs=tensor('occupancy_grid.occs').reshape_as(mask);threshold=min(float(occs.mean()),.01)
    xyz=torch.tensor(evidence['witness_xyz'],dtype=torch.float32)
    shape=torch.tensor(mask.shape);ijk=torch.floor((xyz-box[:3])/(box[3:]-box[:3])*shape).long()
    if not bool(((ijk>=0)&(ijk<shape)).all()):raise ValueError('Outside-AABB witness')
    admitted=mask[tuple(ijk.T)].numpy();sigma=evidence['witness_sigma']
    qualifies=sigma*.003>threshold;missing=~admitted
    bands=[]
    for index,row in enumerate(report['rows']):
        distance=row['witness_distance'];radius=max(.005,.01*distance)
        lo,hi=distance-radius,distance+radius;prefix=str(index)+'_'
        start,end,density,T,weights=[traces[prefix+k]for k in ['starts','ends','sigma','T','weights']]
        clipped_start=np.maximum(start,lo);length=np.maximum(0,np.minimum(end,hi)-clipped_start)
        mass=float((T*np.exp(-density*np.maximum(0,clipped_start-start))*(-np.expm1(-density*length))).sum())
        ns,ne,nw=[traces[prefix+'native_'+k]for k in ['starts','ends','weights']]
        native_mass=float((nw*np.maximum(0,np.minimum(ne,hi)-np.maximum(ns,lo))/np.maximum(ne-ns,1e-20)).sum())
        bands.append(dict(observation=row['observation'],camera=row['camera'],point_index=row['point_index'],
            witness_distance=distance,radius=radius,uniform_band_optical_thickness=float((density*length).sum()),
            uniform_band_weight_mass=mass,uniform_band_conditional_mass=mass/max(float(weights.sum()),1e-20),
            native_band_weight_mass_fractional_intervals=native_mass))
    fn_rows=[dict(point_index=int(i),sigma=float(sigma[i]),cell_ema=float(occs[tuple(ijk[i])]),
                  high_sigma_qualifies=bool(qualifies[i]),cell=ijk[i].tolist())for i in np.flatnonzero(missing)]
    output=dict(checkpoint=report['checkpoint'],audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),read_only=True,
        native_ema_threshold=threshold,point_count=len(xyz),default_false_free=int(missing.sum()),
        high_sigma_false_free=int((missing&qualifies).sum()),low_sigma_false_free=int((missing&~qualifies).sum()),
        all_point_sigma_stats=stats(sigma),false_free_sigma_stats=stats(sigma[missing])if missing.any()else None,
        fp32_index_admission_matches_prior=bool(np.array_equal(admitted,evidence['witness_cell_occupied'])),
        high_sigma_qualifier='sigma_at_point * render_step_size .003 > min(mean(savedEMA),.01)',
        uniform_band_mass_stats=stats([r['uniform_band_weight_mass']for r in bands]),
        uniform_conditional_band_mass_stats=stats([r['uniform_band_conditional_mass']for r in bands]),
        native_band_mass_stats=stats([r['native_band_weight_mass_fractional_intervals']for r in bands]),
        uniform_rays_band_mass_lt_05=sum(r['uniform_band_weight_mass']<.05 for r in bands),
        uniform_rays_band_mass_lt_50=sum(r['uniform_band_weight_mass']<.5 for r in bands),
        point_rows=fn_rows,band_rows=bands,
        definitions=['Uniform band mass integrates piecewise-constant density exactly inside clipped band.',
            'Native band mass weights packed intervals by fractional band overlap.',
            'Band halfwidth=max(.005,.01*witness distance); same actual observed rays as the audit.',
            'SfM witnesses are geometry evidence, not independent ground truth.'])
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),xyz=xyz.numpy(),indices_xyz=ijk.numpy(),
                       sigma=sigma,cell_ema=occs[tuple(ijk.T)].numpy(),admitted=admitted,
                       high_sigma_qualifies=qualifies)
    print(json.dumps({k:output[k]for k in ['default_false_free','high_sigma_false_free','low_sigma_false_free',
        'uniform_band_mass_stats','native_band_mass_stats','uniform_rays_band_mass_lt_05','uniform_rays_band_mass_lt_50']}))


if __name__=='__main__':main()
