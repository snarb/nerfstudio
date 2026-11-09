"""Predeclared view-independent physical export hypotheses; no automatic choice."""


def selector_memberships(probability):
    """p is a class belief; none of these definitions establishes optical alphaGT.

    Untouched inside-box grid nodes retain their explicitly configured initial
    probability. The prior, not evidence, controls such regions: p.99 survives
    hard thresholds while p.01 does not. No support gate or hidden prior is
    introduced here; neither choice establishes physical emptiness.
    """
    t = ((probability-.1)/.4).clamp(0, 1)
    return dict(soft_probability=probability,
        hard_theta01=(probability >= .1).to(probability.dtype),
        hard_theta05=(probability >= .5).to(probability.dtype),
        exactzero_smoothstep01_05=t*t*(3-2*t))
