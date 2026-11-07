"""Regression for disappearing ownership-uncertain actor fragments in 2D guidance."""
import json
from pathlib import Path
import sys
import numpy as np
import marshawn_union_tracking_proposals_v1 as m


def run():
    b = np.zeros(m.NATIVE, bool)
    b[1000:1200, 3000:3200] = True
    # A thin valid contour pixel and a whole ambiguous far-leg-like patch.
    b[1400, 3400] = True
    g = np.zeros_like(b)
    u = np.zeros_like(b)
    u[1000:1200, 3100:3200] = True
    u[1400, 3400] = True
    veto = np.zeros_like(b)
    veto[1000:1020, 3000:3020] = True
    original = [a.copy() for a in [b, g, u, veto]]
    c = m.channels(b, g, u, veto)
    assert c['complete_union'][137, 393] and not c['identity_black'][137, 393]
    assert c['complete_union'][175, 425] and not c['identity_black'][175, 425]
    assert np.array_equal(c['complete_union'], m.pool(b & ~veto))
    im = np.random.default_rng(42).integers(0, 255, (540, 1024), dtype='uint8')
    f = m.flow_channels(im, im, c)
    assert f['flowed_complete_union'][137, 393] and not f['flowed_identity_black'][137, 393]
    assert f['flowed_complete_union'][175, 425]
    assert all(np.array_equal(a, z) for a, z in zip(original, [b, g, u, veto]))
    return {'passed': True, 'full_union_retains_ownership_uncertainty': True,
            'identity_core_excludes_uncertainty': True, 'one_native_pixel_union_retained': True,
            'reliable_flow_keeps_uncertain_union_and_thin_contour': True,
            'independent_veto_respected': True, 'previous_arrays_not_modified': True,
            'proposal_helper_sha256': m.sha(m.__file__), 'test_code_sha256': m.sha(__file__)}


if __name__ == '__main__':
    result = run()
    if len(sys.argv) > 1:
        m.atomic(Path(sys.argv[1]), result)
    print(json.dumps(result))
