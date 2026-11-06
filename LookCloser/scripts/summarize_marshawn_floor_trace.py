"""CPU-only review of a completed, bounded native-weight probe; no geometry labels."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    root=a.root;plan=json.loads((root/'plan.json').read_text());assert json.loads((root/'complete.json').read_text())['passed']
    query=json.loads((root/'queries/eval_queries.json').read_text());regions=json.loads(a.regions.read_text())
    assert sha(a.regions)in query['source_bindings'].values()
    a.output.mkdir(parents=True,exist_ok=False);results=[];parity=[];bindings={str(root/'plan.json'):sha(root/'plan.json'),str(root/'complete.json'):sha(root/'complete.json'),str(a.regions):sha(a.regions)}
    for model in plan['models']:
        for split in ['eval','train']:
            trace=root/'traces'/(model['label']+'_'+split)
            p=trace/'height_bins/partition.json';part=json.loads(p.read_text());bindings[str(p)]=sha(p)
            pp=trace/'native_parity.json';pr=json.loads(pp.read_text());bindings[str(pp)]=sha(pp)
            parity.append(dict(model=model['label'],split=split,rays=len(pr['rows']),max_alpha_error=max(r['alpha_error']for r in pr['rows']),max_RGB_error=max(r['foreground_RGB_error']for r in pr['rows']),max_PNG_float_byte_error=max(r.get('PNG_byte_float_error',0)for r in pr['rows'])))
            for r in part['rows']:
                w=np.array(r['mass_per_bin']);s=w.sum();rgb=np.array(r['RGB_contribution_per_bin']);p=r['pixel']
                results.append(dict(model=model['label'],split=split,camera=p['camera_name'],x=p['x'],y=p['y'],query_id=p['query_id'],region=p['region'],
                    alpha=float(s),mass_below_minus015=float(w[:2].sum()),mass_within015=float(w[2:8].sum()),mass_above_plus015=float(w[8:].sum()),
                    fraction_below_minus015=float(w[:2].sum()/s)if s else None,
                    fraction_within015=float(w[2:8].sum()/s)if s else None,
                    RGB_below_minus015=rgb[:2].sum(0).tolist(),RGB_within015=rgb[2:8].sum(0).tolist(),RGB_above_plus015=rgb[8:].sum(0).tolist(),
                    weighted_midpoint_Z_quantiles=r['signed_Z_weighted_midpoint_quantiles_10_50_90'],
                    bin_RGB_sum_error=r['bin_RGB_sum_max_abs_error'],sample_weight_sum_error=r['numerical']['max_sample_weight_reconstruction_error'],
                    analytic_vs_actual_weight_max_error=r['numerical']['analytic_minus_original_weight']['max_abs']))
    result=dict(rows=results,parity=parity,source_bindings=bindings,scope='27 fixed diagnostic EVAL pixels and6 TRAINshoe observations permodel; sparse selection is not a representative floor-area estimator',
        interpretation='Z is fitted floor coordinate, not height above independently known physicalfloor. No EVAL fitting, carving, source mutation or new geometry labels.',
        actual_weight_conservation_max=max(r['sample_weight_sum_error']for r in results),RGB_conservation_max=max(r['bin_RGB_sum_error']for r in results),
        analytic_vs_actual_weight_max=max(r['analytic_vs_actual_weight_max_error']for r in results))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    keys=['model','split','camera','x','y','query_id','region','alpha','mass_below_minus015','mass_within015','mass_above_plus015','fraction_below_minus015','fraction_within015']
    with(a.output/'rays.csv').open('w')as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    sheet=Image.new('RGB',(4*440,3*250),(20,20,20));d=ImageDraw.Draw(sheet)
    for row,cam in enumerate(['DC','HC','KA']):
        source=root/'queries'/('eval_'+cam+'_source_queries.png');im=Image.open(source).convert('RGB');sheet.paste(im,(0,row*250+25));d.text((4,row*250+5),cam+' SOURCE (diagnostic only)',fill='white')
        for col,model in enumerate(plan['models'],1):
            im=Image.open(model['reference_PNGs'][cam]).convert('RGB');draw=ImageDraw.Draw(im)
            for p in query['pixels']:
                if p['camera_name']!=cam:continue
                x,y=p['x'],p['y'];draw.ellipse((x-4,y-4,x+4,y+4),outline='red',width=1);draw.text((x+5,y-10),str(p['query_id']),fill='yellow')
            im=im.crop(regions[cam]['feet']);sheet.paste(im,(col*440,row*250+25));d.text((col*440+4,row*250+5),model['label'],fill='white')
    sheet.save(a.output/'source_and_models.jpg',quality=96)
    print(json.dumps({k:result[k]for k in ['parity','actual_weight_conservation_max','RGB_conservation_max','analytic_vs_actual_weight_max']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['root','regions','output']:p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
