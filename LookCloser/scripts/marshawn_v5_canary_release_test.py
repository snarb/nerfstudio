"""CPU scope/hash regressions for the post1675,1680-only release."""
from pathlib import Path
import json, tempfile
import marshawn_mask_assist_canary1680_v5 as canary
import marshawn_sam3_assist as io


def run():
    with tempfile.TemporaryDirectory() as temporary:
        V = Path(temporary); release = V / 'operations/sam3_assist/v5/canary1680_release.json'
        io.write(release, dict(files_sha256={}))
        try:
            canary.guard(V, 1685)
            raise AssertionError('Out-of-scope frame accepted')
        except ValueError:
            pass
        try:
            canary.guard(V, 1680)
            raise AssertionError('Missing visual release accepted')
        except FileNotFoundError:
            pass
        native = V / 'frames/001675/native_semantic_ready.json'; prepared = V / 'frames/001675/data/prepared_ready.json'
        io.write(native, {'ready': True}); io.write(prepared, {'ready': True})
        gate = dict(kind='ROOT_REVIEWED_V5_1675_NATIVE_SEMANTICS_RELEASE', authorized=True, frame=1675, cameras=66,
                    native_semantic_ready_sha256=io.sha(native), prepared_ready_sha256=io.sha(prepared), visual_evidence=[],
                    future_scope={'canary_frame': 1680, 'continuous_authorized': False})
        path = V / 'operations/v5_1675_final_visual_release.json'; io.write(path, gate)
        assert canary.guard(V, 1680)['files_sha256'][str(path)] == io.sha(path)
        io.write(prepared, {'ready': 'changed'})
        try:
            canary.guard(V, 1680)
            raise AssertionError('Changed prepared data accepted')
        except ValueError:
            pass
    print(json.dumps(dict(passed=True, missing_gate_blocks=True, source_hash_change_blocks=True, only1680_allowed=True)))


if __name__ == '__main__':
    run()
