#!/usr/bin/env python3
"""Geometry-only inner-standoff candidate camera bundles; preserve original paths.
Every camera translates d along its unit OpenGL forward(-localZ), exact K/R fixed.
Sparse witness clearance does not certify collision-free unseen volume.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;source=r/'review_path_matched_intrinsics';base={k:v.copy()for k,v in np.load(source/'cameras.npz').items()};old=json.loads((source/'receipt.json').read_text());xyz=np.load(r/'geometry/train_witnesses.npz')['xyz'];witness=r/'final_pair/release_audit/radius0/body/all_witnesses.npz';w=np.load(witness);body=w['xyz'][w['subject']];assert len(xyz)==9022 and len(body)==495;trees=[cKDTree(xyz),cKDTree(body)];R=base['camera_to_worlds'][:,:,:3];origin=base['camera_to_worlds'][:,:,3];forward=-R[:,:,2];np.testing.assert_allclose(np.linalg.norm(forward,axis=1),1,rtol=0,atol=1e-12);a.output.mkdir(parents=True,exist_ok=False);summary=[]
 for offset in [0.,.1,.15,.2]:
  name='offset_'+format(offset,'.2f').replace('.','p');folder=a.output/name;folder.mkdir();arrays={k:v.copy()for k,v in base.items()};arrays['camera_to_worlds'][:,:,3]=origin+offset*forward
  assert np.array_equal(arrays['camera_to_worlds'][:,:,:3],R)
  for key in base:
   if key!='camera_to_worlds':assert np.array_equal(arrays[key],base[key])
  np.testing.assert_allclose(arrays['camera_to_worlds'][:,:,3]-origin,offset*forward,rtol=0,atol=1e-15);distances=[]
  for tree in trees:distances.append(tree.query(arrays['camera_to_worlds'][:,:,3]))
  rows=[]
  for i in range(24):rows.append(dict(index=i,anchors=old['records'][i]['anchors'],offset=offset,origin=arrays['camera_to_worlds'][i,:,3].tolist(),forward=forward[i].tolist(),nearest_scene_distance=float(distances[0][0][i]),nearest_scene_point=int(distances[0][1][i]),nearest_body_distance=float(distances[1][0][i]),nearest_body_local_point=int(distances[1][1][i])))
  np.savez_compressed(folder/'cameras.npz',**arrays);receipt=dict(old);receipt.update(bundle_sha256=sha(folder/'cameras.npz'),script_sha256=sha(__file__),source_matched_bundle_sha256=sha(source/'cameras.npz'),source_matched_receipt_sha256=sha(source/'receipt.json'),poses_identical_to_legacy_path=offset==0,source_pose_path_sha256=old['source_pose_path_sha256'],translation_policy='new_origin=exactmatched_origin +offset*(-exactmatched_rotation[:,2]); unitforward',translation_offset=offset,rotations_bitwise_unchanged=True,intrinsics_bitwise_unchanged=True,independent_exact_translation_check=True,translation_max_abs_error=float(abs(arrays['camera_to_worlds'][:,:,3]-(origin+offset*forward)).max()),history='New INNER-STANDOFF candidate; original matchedK andwide paths preserved. No claim sparsepointclearance proves physicalcollision freedom.',clearance_rows=rows)
  (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(folder/'cameras.json').write_text(json.dumps(dict(camera_type='perspective',views=24,encoding='Directfx/fy/cx/cy, Runchanged, innerstandofftranslation',frames=[dict(camera_to_world=arrays['camera_to_worlds'][i].tolist(),**old['records'][i])for i in range(24)]),indent=2)+'\n');summary.append(dict(offset=offset,bundle=str(folder/'cameras.npz'),receipt=str(folder/'receipt.json'),scene_min=float(distances[0][0].min()),scene_median=float(np.median(distances[0][0])),body_min=float(distances[1][0].min()),body_median=float(np.median(distances[1][0])),problem_views=[rows[i]for i in [3,6,17,21]],all24_rows=rows))
 report=dict(source_bundle_sha256=sha(source/'cameras.npz'),source_points_sha256=sha(r/'geometry/train_witnesses.npz'),body_witness_sha256=sha(witness),scene_points=9022,body_points=495,script_sha256=sha(__file__),candidates=summary,selection='Not automaticallyphysicalcertified; choose minimumoffsetmeetingdeclaredknownpointclearancecriterion andactualdensity/pathreview',limitations=['Scene/bodywitnesses sparse, not a solidmesh; distances are not collision certificates.','Translations are newnovel viewpoints; sameR/K doesnotmake them physicalcaptureposes.','No NN/grid changes; no rendering or optimizer steps performed.'])
 (a.output/'clearance_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([{k:x[k]for k in ['offset','scene_min','scene_median','body_min','body_median']}for x in summary]))
if __name__=='__main__':main()
