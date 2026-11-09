# Reviewed Marshawn final-pair helper snapshot

These five Python files are byte-identical copies of the actual reviewed runtime
under `/fsx/tmp/marshawn_001660_hd/final_pair/code`; hashes and the successful
no-optimizer CUDA source/callback/export/reload probe are recorded in
`SOURCE_MANIFEST.json`. This directory preserves original sibling import names
so trusted campaign YAML can reconstruct the same class. Add this directory
to `PYTHONPATH` alongside the explicitly pinned native code and parser snapshot.
No native model/field source is vendored here.

`run_final_pair.py` retains absolute campaign paths and is not a generic launcher.
Its presence is for review/reproduction, not authorization for another training
run. Actual checkpoints, configs, source RGB and geometry contracts remain
immutable FSX artifacts. Read-only reproduction commands and pending acceptance
checks are documented in:

`/fsx/tmp/marshawn_001660_hd/final_pair/README_REPRODUCIBILITY.md`

Import-only validation using these repository copies and the pinned native model
passed without constructing a field or performing optimizer/GPU work; receipt:
`/fsx/tmp/marshawn_001660_hd/final_pair/repo_import_receipt.json`.

The one-halo policy is noncumulative. Stored radius-1 active masks must not be
expanded again on load. Blue native-core and amber guard labels describe
traversal admission, not certified physical occupancy. The inference-only
witness-union candidate is a separate external mask, never a native checkpoint
or replacement for its authoritative exported core.
