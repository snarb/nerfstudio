"""Isolated semantic-target gate; low support is not physical empty space."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    r=json.loads((a.counts.parent/'receipt.json').read_text());assert sha(a.counts)==r['counts_sha256']
    with np.load(r['grid_path']) as z:base=z['semantic_gate0'].copy();oldpad=z['semantic_gate_pad1'].copy();bounds=z['aabb'].copy();frame=z['world_to_box'].copy()
    key='W64_overlap_no_unknown_footprint';counts=np.zeros_like(base,dtype=np.uint8)
    with np.load(a.counts) as z:counts.reshape(-1)[z['flat_indices']]=z[key]
    gate=base&(counts>=2);pad=binary_dilation(gate,structure=np.ones((3,3,3),bool))
    assert not (gate&~base).any() and not (pad&~oldpad).any()
    np.savez_compressed(a.output/'semantic_grid.npz',semantic_gate0=gate,semantic_gate_pad1=pad,aabb=bounds,world_to_box=frame,resolution=np.asarray(gate.shape))
    receipt=dict(kind='PROVISIONAL_SEMANTIC_TARGET_SUPPORT_RESTRICTION_NOT_FREE_SPACE',source_grid=r['grid_path'],source_grid_sha256=sha(r['grid_path']),counts_path=str(a.counts),counts_sha256=sha(a.counts),source_manifest_sha256=r['source_manifest_sha256'],support_key=key,minimum_views=2,gate0_rule='Original partial-occlusion W64 gate0 AND at least2 positive actor-support overlaps whose entire conservative projected cell rectangle is valid and UNKNOWN-free.',padding='One fresh26-neighbor dilation from filtered gate0; original nativeR1 unchanged.',counts=dict(gate0=int(gate.sum()),pad1=int(pad.sum()),old_gate0=int(base.sum()),old_pad1=int(oldpad.sum())),grid_sha256=sha(a.output/'semantic_grid.npz'),builder_sha256=sha(__file__),field_checkpoint_unchanged=True,no_eval_input=True,limitations=['A semantic-target restriction excludes insufficiently evidenced cells; it does not certify physical emptiness.','UNKNOWN coverage is partial and positive masks contain remaining boundary errors.','Preserved ray intersection and sparse clothing/toe depths cannot guarantee complete hair or hidden actor surfaces.'])
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['counts']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--counts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
