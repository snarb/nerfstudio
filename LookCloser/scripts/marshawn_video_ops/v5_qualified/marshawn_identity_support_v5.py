"""Actor ownership uncertainty affects identity anchors, never foreground union."""
from pathlib import Path
import numpy as np
from prepare_marshawn_tuning import sha
from crew_guard_plan import prior_actors,small

def clean_identity_actors(path):
 with np.load(path)as z:
  uncertain=z['actor_ownership_uncertain']if'actor_ownership_uncertain'in z.files else np.zeros_like(z['black'])
  actors={a:small(z[a]&~uncertain)for a in ['black','grey']}
 return actors,sha(path)

def prior_identity_actors(root,frame,camera):
 p=Path(root)/'frames'/f'{frame-5:06d}'/'native_masks'/f'{camera}.npz'
 return clean_identity_actors(p)if p.exists()else prior_actors(root,frame,camera)
