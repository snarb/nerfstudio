"""CPU regressions for isolated V5 observation/resolution and identity seeds."""
from pathlib import Path
from types import SimpleNamespace
import json, tempfile
import numpy as np
import marshawn_sam3_union_capture_v5 as capture


def run(resolver):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary); native = root / 'frames/001675/native_masks'; native.mkdir(parents=True)
        b = np.zeros((32, 48), bool); g = b.copy(); b[3:20, 3:25] = True; g[10:28, 16:40] = True
        veto = b & False; veto[12:14, 19:21] = True; unknown = b & False
        # Aggregate own-actor UNKNOWN must remain unchanged, not erase union.
        unknown[15, 20] = True
        module = SimpleNamespace(__file__=__file__, np=np)

        def refine(model, proc, image, coarse, banned):
            return coarse.copy(), [{'same_original_tile': True}]

        def main():
            camera = 'MC'; refined = {}; tiles = {}
            for actor, mask in [('black', b), ('grey', g)]:
                refined[actor], tiles[actor] = module.refine(None, None, None, mask, veto)
            overlap = refined['black'] & refined['grey']
            for actor in refined:
                refined[actor][overlap] = False
            path = native / (camera + '.npz')
            module.np.savez_compressed(path, **refined, source_ineligible=veto, unknown=unknown, crop_xyxy=np.array([0, 0, 48, 32]))
            capture.write(native / (camera + '.json'), dict(frame=1675, camera=camera, archive_sha256=capture.sha(path), tiles=tiles))

        module.refine = refine; module.main = main
        restore = capture.install_capture(module, root / 'pre_clear')
        module.main(); restore()
        with np.load(native / 'MC.npz') as z:
            assert np.array_equal(z['black'], b & ~(b & g))
            assert np.array_equal(z['grey'], g & ~(b & g))
        records = capture.resolve_captured(native, root / 'pre_clear', resolver, capture.sha(resolver))
        with np.load(native / 'MC.npz') as z:
            assert np.array_equal((z['black'] | z['grey']) & ~veto, (b | g) & ~veto)
            assert not (z['black'] & z['grey']).any()
            assert np.array_equal(z['unknown'], unknown)
            assert np.array_equal(z['source_ineligible'], veto)
            assert np.array_equal(z['actor_ownership_uncertain'], b & g & ~veto)
        assert records['MC']['union_conserved']
        capture.resolve_captured(native, root / 'pre_clear', resolver, capture.sha(resolver))
        # Original prior selection and source hash are retained. Uncertain
        # labels disappear only from identity anchors, not foreground masks.
        module.small = lambda mask: mask[::2, ::2]
        previous_sha = capture.sha(native / 'MC.npz')
        def seed(campaign, frame, camera):
            with np.load(native / 'MC.npz') as z:
                return {a: module.small(z[a]) for a in ['black', 'grey']}, 1675, previous_sha
        (reference, previous, digest), policy = capture.seed_with_ownership_policy(module, root, 1680, 'MC', seed)
        assert previous == 1675 and digest == previous_sha and policy['uncertainty_present']
        with np.load(native / 'MC.npz') as z:
            for actor in ['black', 'grey']:
                assert np.array_equal(reference[actor], module.small(z[actor] & ~z['actor_ownership_uncertain']))
        return dict(canonical_post_clear_replay_exact=True, union_conserved=True,
                    aggregate_unknown_preserved=True, independent_veto_preserved=True,
                    uncertain_identity_anchors_excluded=True, idempotent_resolution=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--resolver', type=Path, required=True); a = p.parse_args()
    print(json.dumps(run(a.resolver)))
