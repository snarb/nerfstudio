# Marshawn video native preparation operations

These are exact source copies of the operational helpers used by the released Marshawn 1660–2015 campaign. `source_snapshot.json` binds each copy to its live source SHA and the production releases. The frozen scientific preprocessing/training tree is separate; copying these files did not change running code.

The CPU producer and watchdog retain their V3 filenames. A versioned V4 release changes the native-assist command to the V4 entrypoint in `../marshawn_mask_assist_full_v4.py`, which adopts an existing frame helper and fixes the scheduling timeout for later frames. The V4 controller times genuine absence of its own work/progress; it does not turn normal camera waves into LOCAL fallback after 60 seconds of total frame age. `../marshawn_mask_v4_cpu_test.py` reproduces that scheduling regression without CUDA.

The producer waits for the exclusive source decoder, invokes the native assist without holding a GPU lease, checks every current person and qualified fragment, builds soft W64 with the frozen DATA helper, and publishes only `native_semantic_ready.json`. The visibility dispatcher subsequently finishes all 66 frequency maps, seals the inventory, builds current mat/bag UNKNOWN admission, and publishes field dispatch. No model, optimizer, learned occupancy or frequency state crosses frames.

`crew_guard_autonomous.py` preserves all existing actor pixels and only adds qualified components from the current NN output. Independent crew/unknown-person conflicts block publication. Its real positive regression is the two black fighter legs in JE1670; the negative regression is the bald crew member in MC1665. The policy and source-bound regression receipt are included. `crew_guard_apply.py` and `crew_track_native.py` preserve the initial qualified crew-repair procedure and its rejected-candidate guards.

Deploy the flat helper files into a campaign's `operations` directory, retain the policy/regression relative paths, and supply its original frozen scripts on `PYTHONPATH`. Production releases enumerate exact helper, source, calibration, photometry, source-owner and box hashes; the campaign root supplies these data-specific inputs. The entrypoint is:

```bash
PYTHONPATH="$CAMPAIGN_ROOT/operations:$CAMPAIGN_ROOT/code/LookCloser/scripts" \
  "$CONDA_PYTHON" "$CAMPAIGN_ROOT/operations/producer_v3_watchdog.py" \
  --root "$CAMPAIGN_ROOT" --release "$CAMPAIGN_ROOT/operations/producer_v4_release.json"
```

Actual preparation checks, source limitations and visual evidence are documented in `../../experiments/marshawn_video_preparation_20261107.md`.
