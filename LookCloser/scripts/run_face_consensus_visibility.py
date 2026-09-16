"""Recover from a bad old semantic source using three other valid skin witnesses.

Exact ablation of the old-source skin requirement. The selected old source must
still pass its original visibility gate; new RGB must come from eroded train
face support and a strictly better-quality exactly visible camera. No inpainting.
"""
import argparse
import shutil
from pathlib import Path
import numpy as np
import study_face_interior_visibility as base
from study_multiview_face_prior import read, save, sha

PARENT = Path('/mnt/data/dec5_face_interior_visibility90_001123')
ROOT = Path('/mnt/data/dec5_face_consensus_visibility_001123')


def consensus_proposals(old_valid, face_support, weights, chosen):
    j = np.arange(old_valid.shape[1]); safe = np.clip(chosen, 0, old_valid.shape[0]-1)
    anchor = (chosen >= 0) & (chosen < old_valid.shape[0]) & old_valid[safe, j]
    votes = (old_valid & face_support).sum(0)
    old = weights[safe, j]
    candidate = (~old_valid) & face_support & (weights > old[None]) & anchor[None] & (votes[None] >= 3)
    return candidate, votes, old


def main(stage):
    if stage == 'prepare':
        assert not ROOT.exists(); ROOT.mkdir()
        q = read(PARENT/'request.json')
        for p, h in q['input_hashes'].items(): assert sha(p) == h
        assert sha(PARENT/'face_masks.npz') == q['face_masks_sha256']
        shutil.copyfile(PARENT/'face_masks.npz', ROOT/'face_masks.npz')
        q['input_hashes'].update({str(Path(__file__).resolve()): sha(__file__),
            str(Path(base.__file__).resolve()): sha(base.__file__),
            str(PARENT/'request.json'): sha(PARENT/'request.json'),
            str(PARENT/'review/audit.json'): sha(PARENT/'review/audit.json')})
        q.update(script_sha256=sha(__file__), old_selected_source_requires_face_semantics=False,
                 ablation='Only remove old-source semantic veto; keep original old-source visibility and three face witnesses.')
        save(ROOT/'request.json', q)
    else:
        base.ROOT = ROOT; base.proposals = consensus_proposals; base.__dict__['__file__'] = __file__
        base.render()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['prepare','render']); a=p.parse_args()
    base.torch.set_num_threads(2)
    with base.torch.inference_mode(): main(a.stage)
