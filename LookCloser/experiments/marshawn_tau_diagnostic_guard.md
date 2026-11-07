# Quarter τ diagnostic orchestration

## What was tested

Prepared an external v3 guard for three fresh 12000-attempt continuations from the original quarter.01 checkpoint at step91128: control, qualified7369 clear rays and qualified10481 clear rays. This is a causal target-population experiment on that older parent; it does not claim to use the current best recipe. No job has been launched by this preparation.

The scientific snapshot, worker, specifications, source checkpoint, target populations and six evaluation gates are unchanged. The prior v2 guard and its completed zero/technical evidence remain intact. The exact technical verification is `08d4055a816311e4642f80d65e43c5e16175fe9d14a3315fffe9864ac0464309`: original Adam91087→91119→91151, actual91136 refresh and exact state/private-stream restoration. Technical checkpoints are excluded from production ancestry.

## Results

The prepared package is `C/quarter_seven_tau_candidate_v1/external_guard_v3`, where `C=/fsx/tmp/marshawn_001660_hd/tuning_20261106`. Its remote mirror uses `/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106`.

| Artifact | SHA256 |
|---|---|
| Current policy | `5d449a9a5cbaa29f9d804081b613f0abc1ae4624724462c8899fe4ddcae81cd1` |
| Guardian | `82823e7f5778208d43f212288d65a9610980a3b1a36e3a1483520e07f54cbed8` |
| CPU retention helper | `76a14dd314630616ea4edeb42ae2da60a1cdbf09ef32876123a56dd0c2eb9242` |

All1586 bound files passed remote staging verification. Twelve guard functions, including PID/start/boot/argv identity, UUID/resource checks, sticky failure, error accounting and signal escalation, are AST-identical to v2. The exact diff is packaged as `guardian.diff`. Ten new CPU tests pass, including direct replay from the packaged directory; the previously passing27 core guard tests were not repeated.

Changes are limited to diagnostic phase authorization, the exact root-bound technical proof, separate v3 state paths, a3600-second worker deadline and a dispatcher GPU allowlist0/1/2/3/5/6. GPUs4/7 remain reserved for the frequency fits. The16GiB process limit,10GiB allocator,90-second heartbeat expiry,25-second TERM grace and failed-reaper behavior remain. Existing diagnostic outputs/job claims are rejected before fresh dispatch. The three specifications each require `diagnostic_authorized=true`; a separate v3 root release must bind the policy and technical-proof hashes.

An initial unapproved v3 seal had a repository-relative test loader. It was preserved on both hosts as `external_guard_v3_review_generation1`; only the test loader and corresponding proof/policy bindings changed to support direct packaged replay. Guardian, retention helper and scientific files did not change. The superseded policy hash is recorded in the current policy.

## Retention and limits

The separate detached CPU fetcher waits until atomic `latest.json` has reached each gate before reading it. The frozen worker writes renders, metrics, the full checkpoint and grid before publishing this marker. The fetcher verifies checkpoint/grid hashes and the render metric step, then atomically publishes ten local review files: six PNGs, actor metrics, clear-target diagnostics and the grid archive/receipt. Its manifest records source/spec/worker hashes and the remote checkpoint path/hash/size. It never loads a model, uses a GPU, copies full checkpoints or prunes anything.

All six checkpoints per arm—91628,92128,93128,95128,99128,103128—remain remote. Subsequent polls check their existence, size and grid-receipt digest again. A failed transfer leaves unpublished staging evidence; three consecutive fetch errors produce an explicit failure without interrupting workers. Completion requires all18 gate manifests. This metadata check does not replace later visual and geometry review.

After root release, launch both commands from the local host using the package's `guardian.py launch --policy .../policy.json --phase diagnostic` and `retention.py launch --policy .../policy.json`. Guard state is `external_guard_v3/diagnostic`, dispatcher state is `external_guard_v3/diagnostic_controller`, and copied artifacts are under `external_guard_v3/retained_gates`. No source, frequency map, calibration, mask, accepted box or architecture is modified.

Reusable helpers: [guardian](../scripts/guard_marshawn_tau_diagnostic_resources.py), [retention](../scripts/retain_marshawn_tau_diagnostic_gates.py), [preparer](../scripts/prepare_marshawn_tau_diagnostic_guard.py), [new CPU tests](../tests/test_tau_diagnostic_orchestration.py).

A second pre-release self-review found that the fetcher reset its error counter before checking a terminal guard with incomplete retention. The counter now resets only after a successful nonterminal poll, and a tenth regression test proves failure after three attempts. The fetch lock closes on exit. The preceding unapproved seal remains in `external_guard_v3_review_generation2`; `retention_revision3.diff` records the exact change. Guardian and scientific code remain unchanged.
