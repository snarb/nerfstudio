"""Independently verify merged masks, frozen support, monotonic grids, fresh padding."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();B=a.variant.parent
corrections={r['camera']:r for r in json.loads((B/'corrected_v2/mask_corrections.json').read_text())['records']};records=[]
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
for width in ['W64','W32']:
 m=json.loads((a.variant/width/'support_manifest.json').read_text());extra_total=clear_total=0
 for r in m['records']:
  assert sha(r['archive'])==r['archive_sha256'];assert sha(r['source_archive'])==r['source_archive_sha256']
  with np.load(r['archive'])as z:s=z['support'];u=z['unknown'];v=z['valid']
  with np.load(r['source_archive'])as old:os=old['support']if 'support'in old else old['W64']>0;ou=old['unknown']if 'unknown'in old else old['source_ineligible']
  assert np.array_equal(s,os)
  with np.load(r['mat_mask_path'])as mat:expected=ou|mat['mat_unknown']
  if r['camera']in corrections:
   with np.load(corrections[r['camera']]['corrected_archive'])as c:expected|=c['confirmed_bag']|c['contact_unknown']
  if r['camera']in ['CE','JE','NE']:expected[:]=True
  assert np.array_equal(u,expected),(width,r['camera'],'unexpectedUNKNOWN')
  extra_total+=int((u&~ou).sum());clear_total+=int((u&~ou&v&~s).sum())
 new=np.load(a.variant/width/'semantic_grid.npz');old=np.load(B/('corrected_v2'if width=='W64'else'margin_W32')/'semantic_grid.npz')
 assert np.array_equal(old['aabb'],new['aabb'])and np.array_equal(old['world_to_box'],new['world_to_box'])
 for k in ['semantic_gate0','semantic_gate_pad1']:assert not(old[k]&~new[k]).any()
 assert np.array_equal(binary_dilation(new['semantic_gate0'],np.ones((3,3,3),bool)),new['semantic_gate_pad1'])
 records.append(dict(width=width,cameras=len(m['records']),all_source_support_unchanged=True,unknown_union_exact=True,added_unknown_pixels=extra_total,removed_clear_pixels=clear_total,counts={k:dict(old=int(old[k].sum()),new=int(new[k].sum()))for k in ['semantic_gate0','semantic_gate_pad1']},grid_sha256=sha(a.variant/width/'semantic_grid.npz'),fresh_padding=True,monotone_superset=True))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(passed=True,variant=str(a.variant),records=records),indent=2)+'\n');print(json.dumps(records))
