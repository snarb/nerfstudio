#!/usr/bin/env python3
"""Strict new-gain4 frequency gate on top of raster/visual audit; no historical exceptions."""
import argparse,json,hashlib
from pathlib import Path
from types import SimpleNamespace
from review_marshawn_frequencies import run,sha

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',choices=['gain4','gain4_scalar'],default='gain4');p.add_argument('--output',type=Path,required=True);p.add_argument('--require-complete',action='store_true');a=p.parse_args();task=a.root/('frequency_'+a.variant);contract=json.loads((task/'input_contract.json').read_text());generic=run(SimpleNamespace(data=task/'data',frequency_map_dir='lookcloser_frequencies',output=a.output,require_complete=False));report=json.loads((a.output/'review.json').read_text());errors=[]
 if sha(task/'data/transforms.json')!=contract['transforms_sha256']:errors.append('Currenttransforms differsealedinputcontract')
 if a.variant=='gain4':
  if sha(a.root/'color_variants/gain4/color_gate.json')!=contract['source_color_gate_sha256']:errors.append('Gain4sourcecolor gatechanged')
 else:
  source=Path(contract['source_data'])
  if sha(source/'color_contract.json')!=contract['source_color_contract_sha256']or sha(source.parent/'complete.json')!=contract['source_complete_sha256']:errors.append('Scalarsourcecolorexposurecontract changed')
 for row in report['records']:
  if 'metadata' not in row:continue
  meta=json.loads(Path(row['metadata']).read_text());stem=Path(row['image']).stem;source=contract['source_images'].get(stem)
  if not source or row['source_sha256']!=source or meta.get('source_image_sha256')!=source:errors.append(stem+':freshgain4sourcebindingmismatch')
  for key,value in contract['recipe'].items():
   if key=='stride':actual=meta.get('stride')
   else:actual=meta.get(key)
   if actual!=value:errors.append(stem+':wrongrecipe '+key)
  if meta.get('fit_script_sha256')!=contract['fitter_sha256']:errors.append(stem+':unqualifiedfittersource')
  if Path(row['map']).resolve().parent!=(task/'maps').resolve():errors.append(stem+':mapnotnewseparateddirectory')
  if meta.get('threshold_provenance'):errors.append(stem+':historicalmapprovenanceforbidden')
 passed=bool(report['complete_gate'] and not errors);strict=dict(complete_gate=passed,complete_pairs=report['complete_pairs'],valid_maps=report['valid_maps'],strict_errors=errors,input_contract_sha256=sha(task/'input_contract.json'),raster_review_sha256=sha(a.output/'review.json'),script_sha256=sha(Path(__file__)),policy='ALL63 freshgain4sourceSHA; exactSSIM.95/1000steps/min16/max8192/16levels/2features/patch8stride8/chunk1024; fitterSHA6ef90; NOhistoricalexceptions',requires_manual_inspection_of_every_contact_page=True)
 (a.output/'strict_review.json').write_text(json.dumps(strict,indent=2)+'\n');print(json.dumps(strict));raise SystemExit(0 if passed or not a.require_complete else 1)
if __name__=='__main__':main()
