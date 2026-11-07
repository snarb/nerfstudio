"""Conserve current two-actor foreground despite uncertain actor ownership."""
import hashlib
import numpy as np
from scipy.ndimage import distance_transform_edt

def array_sha(array):
 a=np.ascontiguousarray(array);return hashlib.sha256(a.tobytes()).hexdigest()

def resolve_actor_union(pre_black,pre_grey,source_ineligible,independent_unknown=None):
 b=np.asarray(pre_black,bool);g=np.asarray(pre_grey,bool);v=np.asarray(source_ineligible,bool)
 if b.shape!=g.shape or b.shape!=v.shape:raise ValueError('Native actor/veto shapes differ')
 overlap=b&g;eligible=overlap&~v
 if independent_unknown is not None:
  u=np.asarray(independent_unknown,bool)
  if u.shape!=b.shape:raise ValueError('Independent person UNKNOWN shape differs')
  if (eligible&u).any():raise ValueError('Independent person UNKNOWN conflicts with actor union; block publication')
 exclusive_b=b&~g;exclusive_g=g&~b
 # These are labels only, not foreground decisions. Both source-positive
 # actors agree here. Keep the uncertainty independent from RGB/geometry UNKNOWN.
 if eligible.any():
  db=distance_transform_edt(~exclusive_b);dg=distance_transform_edt(~exclusive_g)
  assign_b=eligible&(db<dg);assign_g=eligible&~assign_b
 else:assign_b=np.zeros_like(b);assign_g=np.zeros_like(g)
 out_b=exclusive_b|assign_b;out_g=exclusive_g|assign_g
 expected=(b|g)&~v;actual=(out_b|out_g)&~v
 if not np.array_equal(expected,actual):raise AssertionError('Eligible current actor union lost')
 if (out_b&out_g).any():raise AssertionError('Ownership labels must remain disjoint')
 if (exclusive_b&~out_b).any()or(exclusive_g&~out_g).any():raise AssertionError('Existing actor-exclusive pixels erased')
 receipt=dict(version=5,pre_overlap_pixels=int(overlap.sum()),eligible_restored_overlap_pixels=int(eligible.sum()),overlap_excluded_by_independent_veto=int((overlap&v).sum()),pre_union_eligible_pixels=int(expected.sum()),post_union_eligible_pixels=int(actual.sum()),eligible_pre_union_sha256=array_sha(expected),eligible_post_union_sha256=array_sha(actual),union_conserved=True,ownership_uncertain_sha256=array_sha(eligible),ownership_rule='Nearest current exclusive actor support; deterministic grey tie. All restored ownership remains uncertain and excluded from identity anchors.',geometry_or_RGB_UNKNOWN_unchanged=True,old_actor_exclusive_pixels_preserved=True)
 return dict(black=out_b,grey=out_g,actor_ownership_uncertain=eligible),receipt
